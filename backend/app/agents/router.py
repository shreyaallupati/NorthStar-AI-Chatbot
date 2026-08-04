"""Hybrid intent router.

Layer 1 - high-precision deterministic overrides (explicit order number,
explicit "talk to a human", explicit "main menu", active slot-filling context).
Layer 2 - the locally trained scikit-learn classifier, gated by a confidence
threshold so unfamiliar input becomes `fallback` instead of a wrong guess.
Unrelated "where is my <noun>?" phrasing without order-domain language is also
forced to `fallback` so the classifier cannot over-generalise from
"where is my order" into the tracking flow.
Layer 3 - the original keyword/regex router, used only when scikit-learn is not
importable so the bot never crashes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.agents.classifier import BACKEND_KEYWORD, model
from app.config import settings

logger = logging.getLogger(__name__)

Intent = Literal[
    "order_tracking",
    "returns",
    "shipping",
    "recommendations",
    "handoff",
    "menu",
    "fallback",
]


HANDOFF_PATTERNS = [
    r"talk to (a )?(human|person|agent|someone)",
    r"live agent",
    r"real person",
    r"customer service",
    r"speak to",
    r"human please",
    r"representative",
]

ORDER_PATTERNS = [
    r"where('?s| is) my (order|package)",
    r"track( my)? (order|package)",
    r"order status",
    r"shipping status",
    r"package status",
    r"my order",
    r"order #?\d+",
]

RETURN_PATTERNS = [
    r"return",
    r"exchange",
    r"refund",
    r"send (it )?back",
]

SHIPPING_PATTERNS = [
    r"shipping (info|information|time|policy)",
    r"how long (does|do|is) ship",
    r"delivery time",
    r"expedited",
    r"standard shipping",
]

RECOMMEND_PATTERNS = [
    r"recommend",
    r"suggestion",
    r"what('?s| is) the best",
    r"looking for",
    r"need a ",
    r"sleeping bag",
    r"tent",
    r"boots?",
    r"jacket",
    r"gear for",
    r"product",
]

MENU_PATTERNS = [
    r"main menu",
    r"start over",
    r"go back",
    r"return to (the )?bot",
    r"back to (the )?bot",
    r"menu",
    r"^hi$",
    r"^hello$",
    r"^hey$",
    r"help$",
]

# Narrower set used as a *deterministic override* ahead of the classifier.
MENU_OVERRIDE_PATTERNS = [
    r"\bmain menu\b",
    r"^\s*menu\s*[.!?]*$",
    r"\bstart over\b",
    r"\bgo back\b",
    r"\breturn to (the )?bot\b",
    r"\bback to (the )?bot\b",
    r"^\s*restart\s*[.!?]*$",
    r"^\s*reset\s*[.!?]*$",
]

# "#111", "order 111", "package no. 222", "tracking number 333"
EXPLICIT_ORDER_RE = re.compile(
    r"(?:^|\s)#\s*\d{3,}"
    r"|\b(?:order|package|parcel|shipment|tracking)\b\s*"
    r"(?:number|no\.?|num|id)?\s*#?\s*\d{3,}",
    re.IGNORECASE,
)
BARE_ORDER_RE = re.compile(r"^\s*#?\s*\d{3,}\s*$")
RETURN_LANGUAGE_RE = re.compile(
    r"\b(returns?|returning|refund(?:s|ed)?|exchange|money back)\b|send (?:it|this|them) back",
    re.IGNORECASE,
)
# "where is my X" / "where's my X" without an order-domain noun is out of scope
# (e.g. "where is my cat?", "where is my <abc>?") — the classifier otherwise
# over-generalises from "where is my order" training examples.
WHERE_IS_MY_RE = re.compile(
    r"\bwhere(?:'s|\s+is|\s+has|\s+did)\s+my\b",
    re.IGNORECASE,
)
ORDER_DOMAIN_RE = re.compile(
    r"\b("
    r"orders?|packages?|parcels?|shipments?|deliver(?:y|ies)|"
    r"tracking|track|stuff|box(?:es)?|purchases?|"
    r"ship(?:ped|ping)?|dispatch(?:ed)?|warehouse|"
    r"arrive[ds]?|arriving"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouterDecision:
    intent: Intent
    confidence: float
    source: str


def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def keyword_intent(message: str, awaiting: str | None = None) -> Intent:
    """The original deterministic router, kept as the no-scikit-learn fallback."""
    text = message.strip()
    lower = text.lower()

    if awaiting == "order_id":
        if re.search(r"\d{3,}", text):
            return "order_tracking"
        if _matches(lower, HANDOFF_PATTERNS):
            return "handoff"
        if _matches(lower, MENU_PATTERNS):
            return "menu"

    if awaiting in {"rec_use_case", "rec_preference"}:
        if _matches(lower, HANDOFF_PATTERNS):
            return "handoff"
        if _matches(lower, MENU_PATTERNS):
            return "menu"
        return "recommendations"

    if _matches(lower, HANDOFF_PATTERNS):
        return "handoff"
    if _matches(lower, MENU_PATTERNS):
        return "menu"
    if _matches(lower, ORDER_PATTERNS):
        return "order_tracking"
    if _matches(lower, RETURN_PATTERNS):
        return "returns"
    if _matches(lower, SHIPPING_PATTERNS):
        return "shipping"
    if _matches(lower, RECOMMEND_PATTERNS):
        return "recommendations"
    return "fallback"


def route(message: str, awaiting: str | None = None) -> RouterDecision:
    text = (message or "").strip()
    lower = text.lower()

    if not text:
        return RouterDecision("fallback", 1.0, "override:empty")

    # --- Layer 1: deterministic overrides -------------------------------
    if awaiting == "order_id" and re.search(r"\d{3,}", text):
        return RouterDecision("order_tracking", 1.0, "override:slot_order_id")

    if _matches(lower, HANDOFF_PATTERNS):
        return RouterDecision("handoff", 1.0, "override:handoff")

    if _matches(lower, MENU_OVERRIDE_PATTERNS):
        return RouterDecision("menu", 1.0, "override:menu")

    if BARE_ORDER_RE.match(text):
        return RouterDecision("order_tracking", 1.0, "override:order_number")

    if EXPLICIT_ORDER_RE.search(text) and not RETURN_LANGUAGE_RE.search(text):
        return RouterDecision("order_tracking", 1.0, "override:order_number")

    if awaiting in {"rec_use_case", "rec_preference"}:
        if _matches(lower, MENU_PATTERNS):
            return RouterDecision("menu", 1.0, "override:menu")
        return RouterDecision("recommendations", 1.0, "override:slot_recommendations")

    # --- Layer 2: local ML classifier ------------------------------------
    prediction = model.predict(text)
    if prediction is None:
        return RouterDecision(keyword_intent(message, awaiting), 0.0, BACKEND_KEYWORD)

    if prediction.known_tokens == 0:
        logger.debug("no in-vocabulary tokens for %r, routing to fallback", text)
        return RouterDecision("fallback", prediction.confidence, "classifier:out_of_vocab")

    if prediction.confidence < settings.intent_confidence_threshold:
        logger.debug(
            "low confidence %.3f for %r (top=%s), routing to fallback",
            prediction.confidence,
            text,
            prediction.intent,
        )
        return RouterDecision("fallback", prediction.confidence, "classifier:low_confidence")

    if (
        prediction.intent == "order_tracking"
        and WHERE_IS_MY_RE.search(text)
        and not ORDER_DOMAIN_RE.search(text)
    ):
        logger.debug(
            "unrelated 'where is my …' phrasing for %r, routing to fallback",
            text,
        )
        return RouterDecision(
            "fallback", prediction.confidence, "classifier:unrelated_tracking"
        )

    return RouterDecision(prediction.intent, prediction.confidence, "classifier")


def classify_intent(message: str, awaiting: str | None = None) -> Intent:
    """Stable public entry point used by the graph and tests."""
    return route(message, awaiting).intent


def router_backend() -> str:
    return model.backend
