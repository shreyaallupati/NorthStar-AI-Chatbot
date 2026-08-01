"""Local vector store / retriever for the product catalog and FAQ corpus.

Two offline scorers are fused:

* **TF-IDF cosine similarity** (scikit-learn) over word 1-2 grams.
* **BM25 Okapi** (``rank_bm25``, pure Python).

The fitted vectorizer and document matrix are cached to
``backend/app/data/retriever_index.joblib`` and keyed by a fingerprint of the
corpus, so a changed catalog rebuilds the index automatically.

Everything is exposed through a real ``langchain_core.retrievers.BaseRetriever``
so the recommendation chain is a genuine LCEL pipeline. If scikit-learn or
rank_bm25 are missing, the layer silently degrades to the original token-overlap
scorer -- it never raises and never touches the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any, Callable, Iterable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.config import RETRIEVER_INDEX_PATH, settings
from app.db import list_faqs, list_products

logger = logging.getLogger(__name__)

BACKEND_KEYWORD = "keyword"
_TOKEN_RE = re.compile(r"[a-z0-9]+")

COLD_TOKENS = {"cold", "winter", "zero", "sub", "freezing", "snow", "frigid"}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# --------------------------------------------------------------------------- #
# Corpus construction
# --------------------------------------------------------------------------- #
def _product_document(product: dict) -> Document:
    parts: list[str] = [
        product.get("name", ""),
        product.get("category", "").replace("_", " "),
        product.get("best_for", ""),
        product.get("fill", "") or "",
        product.get("capacity", "") or "",
        product.get("season", "") or "",
        " ".join(product.get("tags", [])),
    ]
    temp = product.get("temp_rating_f")
    if temp is not None:
        parts.append(f"{temp} degrees fahrenheit rating")
        if temp <= 0:
            parts.append("sub-zero below zero freezing cold winter")
    return Document(
        page_content=" ".join(p for p in parts if p).strip(),
        metadata={"kind": "product", "id": product.get("id", ""), "payload": product},
    )


def _faq_document(faq: dict) -> Document:
    content = f"{faq['question']} {faq['answer']} {' '.join(faq.get('tags', []))}"
    return Document(
        page_content=content,
        metadata={"kind": "faq", "id": faq.get("id", ""), "payload": faq},
    )


def product_boost(query_tokens: set[str], product: dict) -> float:
    """Catalog-specific re-ranking signal (kept from the original scorer)."""
    boost = 0.0
    category = product.get("category")
    if {"sleep", "sleeping", "bag", "bags"} & query_tokens and category == "sleeping_bags":
        boost += 0.5
    if {"tent", "tents", "shelter"} & query_tokens and category == "tents":
        boost += 0.5
    if {"boot", "boots", "shoe", "shoes", "footwear"} & query_tokens and category == "footwear":
        boost += 0.5
    if {"jacket", "apparel", "clothing", "layer"} & query_tokens and category == "apparel":
        boost += 0.5
    if {"stove", "cooking", "kitchen", "cook"} & query_tokens and category == "camp_kitchen":
        boost += 0.5
    temp = product.get("temp_rating_f")
    if COLD_TOKENS & query_tokens and temp is not None and temp <= 0:
        boost += 0.8
    return boost


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class LocalIndex:
    """Fused TF-IDF cosine + BM25 index over a small document corpus."""

    def __init__(self, name: str, documents: list[Document]) -> None:
        self.name = name
        self.documents = documents
        self.corpus = [d.page_content for d in documents]
        self.corpus_tokens = [tokenize(text) for text in self.corpus]
        self.fingerprint = hashlib.sha256(
            (name + "\u0000" + "\u0000".join(self.corpus)).encode("utf-8")
        ).hexdigest()
        self.vectorizer: Any = None
        self.matrix: Any = None
        self.bm25: Any = None
        self.backends: list[str] = []
        self._build()

    # -- build ------------------------------------------------------------
    def _build(self) -> None:
        self._build_tfidf()
        self._build_bm25()
        if not self.backends:
            self.backends = [BACKEND_KEYWORD]

    def _build_tfidf(self) -> None:
        if not self.corpus:
            return
        try:
            import joblib
            import sklearn
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception:
            logger.warning("scikit-learn unavailable, TF-IDF retrieval disabled")
            return

        cache_key = f"{self.fingerprint}:{sklearn.__version__}"
        if RETRIEVER_INDEX_PATH.exists():
            try:
                cached = joblib.load(RETRIEVER_INDEX_PATH)
                entry = cached.get(self.name)
                if entry and entry.get("cache_key") == cache_key:
                    self.vectorizer = entry["vectorizer"]
                    self.matrix = entry["matrix"]
                    self.backends.append("tfidf-cosine")
                    logger.info("loaded cached %s retriever index", self.name)
                    return
            except Exception:
                logger.warning("retriever index cache unreadable, rebuilding", exc_info=True)

        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r"[a-z0-9]+",
            )
            matrix = vectorizer.fit_transform(self.corpus)
        except Exception:
            logger.warning("could not fit TF-IDF index for %s", self.name, exc_info=True)
            return

        self.vectorizer = vectorizer
        self.matrix = matrix
        self.backends.append("tfidf-cosine")
        self._persist(cache_key)

    def _persist(self, cache_key: str) -> None:
        try:
            import joblib

            payload: dict[str, Any] = {}
            if RETRIEVER_INDEX_PATH.exists():
                try:
                    payload = joblib.load(RETRIEVER_INDEX_PATH) or {}
                except Exception:
                    payload = {}
            payload[self.name] = {
                "cache_key": cache_key,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
            }
            RETRIEVER_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(payload, RETRIEVER_INDEX_PATH)
        except Exception:
            logger.warning("could not persist retriever index", exc_info=True)

    def _build_bm25(self) -> None:
        if not self.corpus_tokens:
            return
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
            self.backends.append("bm25")
        except Exception:
            logger.warning("rank_bm25 unavailable, BM25 retrieval disabled")

    # -- scoring ----------------------------------------------------------
    def _tfidf_scores(self, query: str) -> list[float]:
        if self.vectorizer is None or self.matrix is None:
            return [0.0] * len(self.documents)
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            vector = self.vectorizer.transform([query])
            return [float(x) for x in cosine_similarity(vector, self.matrix)[0]]
        except Exception:
            logger.warning("TF-IDF scoring failed", exc_info=True)
            return [0.0] * len(self.documents)

    def _bm25_scores(self, query_tokens: list[str]) -> list[float]:
        if self.bm25 is None or not query_tokens:
            return [0.0] * len(self.documents)
        try:
            raw = [float(x) for x in self.bm25.get_scores(query_tokens)]
        except Exception:
            logger.warning("BM25 scoring failed", exc_info=True)
            return [0.0] * len(self.documents)
        top = max(raw) if raw else 0.0
        if top <= 0:
            return [0.0] * len(self.documents)
        return [max(0.0, x) / top for x in raw]

    def _keyword_scores(self, query_tokens: list[str]) -> list[float]:
        unique = set(query_tokens)
        if not unique:
            return [0.0] * len(self.documents)
        scores: list[float] = []
        for tokens in self.corpus_tokens:
            counts = Counter(tokens)
            overlap = sum(counts[t] for t in unique if t in counts)
            scores.append(overlap / max(len(unique), 1))
        return scores

    def score(
        self,
        query: str,
        boost: Callable[[set[str], dict], float] | None = None,
    ) -> list[tuple[float, Document]]:
        tokens = tokenize(query)
        unique = set(tokens)

        if "tfidf-cosine" in self.backends or "bm25" in self.backends:
            tfidf = self._tfidf_scores(query)
            bm25 = self._bm25_scores(tokens)
            fused = [0.6 * a + 0.4 * b for a, b in zip(tfidf, bm25)]
        else:
            fused = self._keyword_scores(tokens)

        results: list[tuple[float, Document]] = []
        for score, document in zip(fused, self.documents):
            if boost is not None:
                score += boost(unique, document.metadata.get("payload", {}))
            results.append((score, document))
        results.sort(key=lambda pair: pair[0], reverse=True)
        return results


# --------------------------------------------------------------------------- #
# LangChain retriever
# --------------------------------------------------------------------------- #
class LocalHybridRetriever(BaseRetriever):
    """LangChain retriever over a `LocalIndex`, usable directly in LCEL chains."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any = None
    k: int = 3
    score_floor: float = 0.0
    use_product_boost: bool = False

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        if self.index is None:
            return []
        boost = product_boost if self.use_product_boost else None
        scored = self.index.score(query, boost=boost)
        hits: list[Document] = []
        for score, document in scored:
            if score <= self.score_floor:
                continue
            enriched = Document(
                page_content=document.page_content,
                metadata={**document.metadata, "score": round(score, 6)},
            )
            hits.append(enriched)
            if len(hits) >= self.k:
                break
        return hits

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        return self._get_relevant_documents(query)


# --------------------------------------------------------------------------- #
# Singletons
# --------------------------------------------------------------------------- #
_product_index: LocalIndex | None = None
_faq_index: LocalIndex | None = None
_product_signature: str | None = None
_faq_signature: str | None = None


def _signature(rows: Iterable[dict]) -> str:
    return hashlib.sha256(
        json.dumps(list(rows), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def product_index() -> LocalIndex:
    global _product_index, _product_signature
    products = list_products()
    signature = _signature(products)
    if _product_index is None or _product_signature != signature:
        _product_index = LocalIndex("products", [_product_document(p) for p in products])
        _product_signature = signature
    return _product_index


def faq_index() -> LocalIndex:
    global _faq_index, _faq_signature
    faqs = list_faqs()
    signature = _signature(faqs)
    if _faq_index is None or _faq_signature != signature:
        _faq_index = LocalIndex("faqs", [_faq_document(f) for f in faqs])
        _faq_signature = signature
    return _faq_index


def product_retriever(k: int = 3) -> LocalHybridRetriever:
    return LocalHybridRetriever(
        index=product_index(),
        k=k,
        score_floor=settings.retrieval_score_floor,
        use_product_boost=True,
    )


def faq_retriever(k: int = 2) -> LocalHybridRetriever:
    return LocalHybridRetriever(
        index=faq_index(),
        k=k,
        score_floor=settings.retrieval_score_floor,
        use_product_boost=False,
    )


def retriever_backend() -> str:
    try:
        backends = product_index().backends
    except Exception:
        logger.warning("retriever backend probe failed", exc_info=True)
        return BACKEND_KEYWORD
    return "+".join(backends) if backends else BACKEND_KEYWORD


def warmup() -> str:
    """Build both indexes eagerly (called from FastAPI startup)."""
    try:
        product_index()
        faq_index()
    except Exception:
        logger.warning("retriever warmup failed", exc_info=True)
    return retriever_backend()
