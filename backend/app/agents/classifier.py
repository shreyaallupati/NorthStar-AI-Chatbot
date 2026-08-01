"""Local intent classifier: TF-IDF + multinomial LogisticRegression.

Trained entirely offline from `app.agents.training_data`. The fitted pipeline is
cached to `backend/app/data/intent_model.joblib` and keyed by a fingerprint of
the training corpus plus the scikit-learn version, so the model is rebuilt
automatically whenever either changes.

If scikit-learn is not importable the module degrades to `available() == False`
and the caller falls back to the keyword router. Nothing here ever raises.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from threading import Lock

from app.agents.training_data import INTENT_LABELS, flatten

logger = logging.getLogger(__name__)

BACKEND_SKLEARN = "sklearn-tfidf-logreg"
BACKEND_KEYWORD = "keyword-fallback"

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float
    known_tokens: int


class _IntentModel:
    def __init__(self) -> None:
        self._lock = Lock()
        self._pipeline = None
        self._vocabulary: frozenset[str] = frozenset()
        self._loaded = False
        self._backend = BACKEND_KEYWORD

    # -- lifecycle ---------------------------------------------------------
    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                self._build()
                self._backend = BACKEND_SKLEARN
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "intent classifier unavailable, degrading to keyword router",
                    exc_info=True,
                )
                self._pipeline = None
                self._backend = BACKEND_KEYWORD

    def _build(self) -> None:
        import joblib
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from app.config import INTENT_MODEL_PATH

        texts, labels = flatten()
        fingerprint = hashlib.sha256(
            ("\n".join(texts) + "\n#" + "\n".join(labels) + "\n#" + sklearn.__version__).encode(
                "utf-8"
            )
        ).hexdigest()

        if INTENT_MODEL_PATH.exists():
            try:
                cached = joblib.load(INTENT_MODEL_PATH)
                if cached.get("fingerprint") == fingerprint:
                    self._pipeline = cached["pipeline"]
                    self._vocabulary = frozenset(cached["vocabulary"])
                    logger.info("loaded cached intent model from %s", INTENT_MODEL_PATH)
                    return
            except Exception:
                logger.warning("cached intent model unreadable, retraining", exc_info=True)

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                        token_pattern=r"[a-z0-9']+",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=8.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=0,
                    ),
                ),
            ]
        )
        pipeline.fit(texts, labels)

        vocabulary = set(pipeline.named_steps["tfidf"].vocabulary_.keys())
        # Keep unigrams only for the out-of-vocabulary check.
        unigrams = {term for term in vocabulary if " " not in term}

        self._pipeline = pipeline
        self._vocabulary = frozenset(unigrams)

        try:
            INTENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "fingerprint": fingerprint,
                    "pipeline": pipeline,
                    "vocabulary": sorted(unigrams),
                },
                INTENT_MODEL_PATH,
            )
            logger.info("trained and cached intent model at %s", INTENT_MODEL_PATH)
        except Exception:
            logger.warning("could not persist intent model", exc_info=True)

    # -- inference ---------------------------------------------------------
    @property
    def backend(self) -> str:
        self.ensure_loaded()
        return self._backend

    def available(self) -> bool:
        self.ensure_loaded()
        return self._pipeline is not None

    def predict(self, message: str) -> Prediction | None:
        self.ensure_loaded()
        if self._pipeline is None:
            return None
        text = (message or "").strip().lower()
        if not text:
            return None
        tokens = _TOKEN_RE.findall(text)
        known = sum(1 for t in tokens if t in self._vocabulary)
        try:
            probabilities = self._pipeline.predict_proba([text])[0]
            classes = list(self._pipeline.named_steps["clf"].classes_)
        except Exception:
            logger.warning("intent prediction failed", exc_info=True)
            return None
        best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        intent = str(classes[best_index])
        if intent not in INTENT_LABELS:
            return None
        return Prediction(
            intent=intent,
            confidence=float(probabilities[best_index]),
            known_tokens=known,
        )


model = _IntentModel()


def warmup() -> str:
    """Train/load the model eagerly (called from FastAPI startup)."""
    model.ensure_loaded()
    return model.backend
