"""Retrieval tools.

Thin, stable wrappers over the offline hybrid retriever in `app.rag.store`.
Public signatures are unchanged from the original keyword implementation so the
graph, tests, and any callers keep working.
"""

from __future__ import annotations

import logging

from app.db import get_policies
from app.rag.store import faq_retriever, product_retriever, retriever_backend

logger = logging.getLogger(__name__)


def search_products(query: str, limit: int = 3) -> list[dict]:
    try:
        docs = product_retriever(k=limit).invoke(query or "")
    except Exception:
        logger.warning("product retrieval failed", exc_info=True)
        return []
    return [doc.metadata["payload"] for doc in docs if doc.metadata.get("payload")]


def search_faqs(query: str, limit: int = 2) -> list[dict]:
    try:
        docs = faq_retriever(k=limit).invoke(query or "")
    except Exception:
        logger.warning("faq retrieval failed", exc_info=True)
        return []
    return [doc.metadata["payload"] for doc in docs if doc.metadata.get("payload")]


def policy_snippet(kind: str) -> str:
    """Grounded policy text rendered straight from the catalog/SQLite data."""
    policies = get_policies()
    if kind == "returns":
        r = policies["returns"]
        return (
            f"{r['summary']} Items must be unused with original packaging. "
            f"Returns link: {r['returns_link']}"
        )
    if kind == "shipping":
        s = policies["shipping"]
        return f"Standard shipping: {s['standard']}. Expedited shipping: {s['expedited']}."
    return ""


def backend_name() -> str:
    return retriever_backend()
