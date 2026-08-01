from __future__ import annotations

from typing import Any, Literal, TypedDict


Mode = Literal["bot", "live_agent"]


class ChatState(TypedDict, total=False):
    """LangGraph channel schema for a single support conversation."""

    session_id: str
    messages: list[dict[str, str]]
    user_message: str
    intent: str
    awaiting: str | None
    order_id: str | None
    rec_answers: list[str]
    fallback_count: int
    mode: Mode
    reply: str
    suggestions: list[str]
    context: dict[str, Any]

    # Set by the router node, consumed by the conditional edge.
    intent_confidence: float
    route_source: str

    # Set by the agent nodes, consumed by the response node.
    template: str
    facts: dict[str, Any]
    grounding: dict[str, Any]
