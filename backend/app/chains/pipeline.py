"""LCEL chains that turn retrieved, grounded facts into user-facing replies.

Three composed `Runnable` pipelines:

* ``response_chain`` - template selection -> rendering -> grounding guard.
* ``recommendation_chain`` - retriever -> product docs -> formatted picks.
* ``faq_chain`` - retriever -> best FAQ answer (falls back to the policy text).

The "generation" step is deterministic template rendering. Nothing is
paraphrased, so a business fact can only appear if it came out of SQLite or the
seeded catalog.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from app.chains.prompts import TEMPLATES
from app.rag.store import faq_retriever, product_retriever
from app.tools.retrieval import policy_snippet

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Response chain
# --------------------------------------------------------------------------- #
def _render(payload: dict[str, Any]) -> dict[str, Any]:
    key = payload.get("template", "")
    facts = dict(payload.get("facts") or {})
    template = TEMPLATES.get(key)
    if template is None:
        logger.error("unknown response template %r, using fallback copy", key)
        return {**payload, "text": TEMPLATES["fallback_first"].format(), "degraded": True}

    scoped = {name: facts.get(name, "") for name in template.input_variables}
    try:
        text = template.format(**scoped)
    except Exception:
        logger.error("template %r failed to render", key, exc_info=True)
        return {**payload, "text": TEMPLATES["fallback_first"].format(), "degraded": True}
    return {**payload, "text": text, "degraded": False}


def _guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Refuse any reply that drops or contradicts a retrieved ground truth fact."""
    grounding = payload.get("grounding") or {}
    must_include = [str(x) for x in grounding.get("must_include", []) if str(x)]
    if not must_include:
        return payload

    text = payload.get("text", "")
    lowered = text.lower()
    missing = [fact for fact in must_include if fact.lower() not in lowered]
    if not missing:
        return payload

    safe_text = grounding.get("safe_text")
    logger.warning(
        "grounding guard tripped for template %r; missing facts=%s; using deterministic reply",
        payload.get("template"),
        missing,
    )
    if safe_text:
        return {**payload, "text": safe_text, "degraded": True, "guard_tripped": True}
    return {**payload, "degraded": True, "guard_tripped": True}


response_chain: Runnable = (
    RunnableLambda(_render, name="render_template")
    | RunnableLambda(_guard, name="grounding_guard")
    | RunnableLambda(lambda payload: payload["text"], name="emit_reply")
)


def render_reply(
    template: str,
    facts: dict[str, Any] | None = None,
    grounding: dict[str, Any] | None = None,
) -> str:
    return response_chain.invoke(
        {"template": template, "facts": facts or {}, "grounding": grounding or {}}
    )


# --------------------------------------------------------------------------- #
# Recommendation chain
# --------------------------------------------------------------------------- #
def _docs_to_products(docs: list[Document]) -> list[dict]:
    return [d.metadata["payload"] for d in docs if d.metadata.get("payload")]


def format_picks(products: list[dict]) -> str:
    lines: list[str] = []
    for index, product in enumerate(products, 1):
        bits = [
            "**{name}** (${price}) - {best}".format(
                name=product.get("name", ""),
                price=product.get("price", ""),
                best=product.get("best_for", product.get("category", "")),
            )
        ]
        if product.get("temp_rating_f") is not None:
            bits.append("rated to {t}\u00b0F".format(t=product["temp_rating_f"]))
        if product.get("fill"):
            bits.append(product["fill"])
        if product.get("capacity"):
            bits.append(product["capacity"])
        lines.append(f"{index}. " + "; ".join(bits))
    return "\n".join(lines)


def _retrieve_products(payload: dict[str, Any]) -> list[Document]:
    query = payload.get("query", "")
    limit = int(payload.get("limit", 3))
    try:
        return product_retriever(k=limit).invoke(query)
    except Exception:
        logger.warning("product retriever failed", exc_info=True)
        return []


recommendation_chain: Runnable = (
    RunnableLambda(_retrieve_products, name="retrieve_products")
    | RunnableLambda(_docs_to_products, name="docs_to_products")
    | RunnableLambda(
        lambda products: {"products": products, "picks": format_picks(products)},
        name="format_picks",
    )
)


# --------------------------------------------------------------------------- #
# FAQ chain
# --------------------------------------------------------------------------- #
def _retrieve_faqs(payload: dict[str, Any]) -> list[Document]:
    query = payload.get("query", "")
    limit = int(payload.get("limit", 2))
    try:
        return faq_retriever(k=limit).invoke(query)
    except Exception:
        logger.warning("faq retriever failed", exc_info=True)
        return []


def _best_faq_answer(payload: dict[str, Any]) -> str:
    docs = payload["docs"]
    kind = payload.get("policy_kind", "returns")
    for doc in docs:
        answer = (doc.metadata.get("payload") or {}).get("answer")
        if answer:
            return str(answer)
    return policy_snippet(kind)


faq_chain: Runnable = RunnableLambda(
    lambda payload: {
        "docs": _retrieve_faqs(payload),
        "policy_kind": payload.get("policy_kind", "returns"),
    },
    name="retrieve_faqs",
) | RunnableLambda(_best_faq_answer, name="select_faq_answer")
