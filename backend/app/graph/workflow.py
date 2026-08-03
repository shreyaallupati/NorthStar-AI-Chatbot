"""LangGraph conversation workflow for North Star Support Bot.

Node flow (all edges below are real `StateGraph` edges):

    START
      -> ingest
           |- (mode == live_agent) -> live_agent -+-> menu_agent -> respond
           |                                      \\-> respond
           \\- (mode == bot)       -> router
                                        |-> order_agent          -> respond
                                        |-> returns_agent        -> respond
                                        |-> shipping_agent       -> respond
                                        |-> recommendation_agent -> respond
                                        |-> escalation_agent     -> respond
                                        |-> menu_agent           -> respond
                                        \\-> fallback_agent       -> respond
                                                                     -> END

`router` is the local scikit-learn intent classifier behind deterministic
overrides. Each agent node retrieves grounded facts (SQLite orders, catalog
policies, hybrid TF-IDF/BM25 retrieval) and picks a LangChain `PromptTemplate`.
`respond` renders that template through the LCEL response chain and runs the
anti-hallucination grounding guard. There is no LLM anywhere in the graph.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.router import route, router_backend
from app.chains.pipeline import faq_chain, recommendation_chain, render_reply, response_chain
from app.chains.prompts import TEMPLATES
from app.db import get_policies
from app.graph.state import ChatState
from app.tools.orders import extract_order_id, lookup_order
from app.tools.retrieval import policy_snippet

logger = logging.getLogger(__name__)

DEFAULT_SUGGESTIONS = [
    "Track my order",
    "Return policy",
    "Product recommendations",
    "Talk to a human",
]

# While a live agent "has" the chat, only a clear request should hand control
# back to the bot -- an ambiguous message must not silently end the handoff.
LIVE_AGENT_EXIT_CONFIDENCE = 0.55

REC_HINT_TOKENS = (
    "sleep",
    "tent",
    "boot",
    "jacket",
    "cold",
    "winter",
    "zero",
    "camp",
)


def initial_state(session_id: str) -> ChatState:
    welcome = render_reply("welcome")
    return {
        "session_id": session_id,
        "messages": [{"role": "assistant", "content": welcome}],
        "user_message": "",
        "intent": "menu",
        "awaiting": None,
        "order_id": None,
        "rec_answers": [],
        "fallback_count": 0,
        "mode": "bot",
        "reply": welcome,
        "suggestions": DEFAULT_SUGGESTIONS,
        "context": {},
        "intent_confidence": 1.0,
        "route_source": "init",
        "template": "welcome",
        "facts": {},
        "grounding": {},
    }


def _menu_reply() -> tuple[str, list[str]]:
    """Kept for backwards compatibility with earlier callers/tests."""
    return render_reply("menu"), list(DEFAULT_SUGGESTIONS)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def ingest_node(state: ChatState) -> dict[str, Any]:
    message = (state.get("user_message") or "").strip()
    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": message})
    return {
        "user_message": message,
        "messages": messages,
        "template": "",
        "facts": {},
        "grounding": {},
    }


def live_agent_node(state: ChatState) -> dict[str, Any]:
    decision = route(state.get("user_message") or "")
    wants_menu = decision.intent == "menu" and (
        decision.source.startswith("override")
        or decision.confidence >= LIVE_AGENT_EXIT_CONFIDENCE
    )
    if wants_menu:
        return {
            "intent": "menu",
            "mode": "bot",
            "fallback_count": 0,
            "intent_confidence": decision.confidence,
            "route_source": decision.source,
        }
    return {
        "intent": "handoff",
        "intent_confidence": decision.confidence,
        "route_source": decision.source,
        "template": "live_agent_ack",
        "facts": {},
        "suggestions": ["Main menu", "Return to bot"],
    }


def router_node(state: ChatState) -> dict[str, Any]:
    decision = route(state.get("user_message") or "", state.get("awaiting"))
    updates: dict[str, Any] = {
        "intent": decision.intent,
        "intent_confidence": decision.confidence,
        "route_source": decision.source,
        "context": {
            "router": {
                "backend": router_backend(),
                "intent": decision.intent,
                "confidence": round(decision.confidence, 4),
                "source": decision.source,
            }
        },
    }
    if decision.intent != "fallback":
        updates["fallback_count"] = 0
    return updates


def menu_node(state: ChatState) -> dict[str, Any]:
    return {
        "intent": "menu",
        "awaiting": None,
        "order_id": None,
        "rec_answers": [],
        "template": "menu",
        "facts": {},
        "suggestions": list(DEFAULT_SUGGESTIONS),
    }


def order_agent_node(state: ChatState) -> dict[str, Any]:
    message = state.get("user_message") or ""
    # Only a number in *this* message starts a lookup. The order context is
    # cleared once a status is shown, so a fresh "where is my order?" must
    # re-ask for a number instead of replaying the previous order.
    order_id = extract_order_id(message)

    if not order_id:
        return {
            "awaiting": "order_id",
            "template": "order_ask",
            "facts": {},
            "suggestions": ["#111", "#222", "#333", "Main menu"],
        }

    order = lookup_order(order_id)
    if not order:
        return {
            "order_id": None,
            "awaiting": "order_id",
            "template": "order_not_found",
            "facts": {"order_id": order_id},
            "suggestions": ["#111", "#222", "#333", "Main menu"],
        }

    status = order["status"]
    facts = {
        "order_id": order["order_id"],
        "item": order["item"],
        "status": status,
        "status_lower": status.lower(),
        "detail": order["detail"],
    }
    # Deterministic, fully grounded reply used if the guard trips.
    safe_text = TEMPLATES["order_generic"].format(
        order_id=order["order_id"], status=status, detail=order["detail"]
    )

    if status == "Shipped":
        template = "order_shipped"
        must_include = [order["order_id"], status.lower(), order["detail"]]
        suggestions = ["Track another order", "Main menu", "Talk to a human"]
    elif status == "Processing":
        template = "order_processing"
        must_include = [order["order_id"], status.lower(), order["detail"]]
        suggestions = ["Track another order", "Main menu", "Talk to a human"]
    elif status == "Delivered":
        template = "order_delivered"
        must_include = [order["order_id"], status.lower()]
        suggestions = ["Return policy", "Product recommendations", "Main menu"]
    else:
        template = "order_generic"
        must_include = [order["order_id"], status, order["detail"]]
        suggestions = ["Track another order", "Main menu", "Talk to a human"]

    return {
        # Clear the active order context: the next tracking request starts a
        # new lookup and asks for a new order number.
        "order_id": None,
        "awaiting": None,
        "template": template,
        "facts": facts,
        "grounding": {"must_include": must_include, "safe_text": safe_text},
        "suggestions": suggestions,
    }


def returns_agent_node(state: ChatState) -> dict[str, Any]:
    message = state.get("user_message") or ""
    policies = get_policies()["returns"]
    # Order context is cleared after every completed lookup, so eligibility
    # checks need an order number in the current message.
    order_id = extract_order_id(message)
    faq_bit = faq_chain.invoke({"query": message or "return policy", "policy_kind": "returns"})

    facts: dict[str, Any] = {
        "faq_bit": faq_bit,
        "window_days": policies["window_days"],
        "returns_link": policies["returns_link"],
    }
    grounding = {
        "must_include": [
            f"{policies['window_days']}-day",
            "unused",
            policies["returns_link"],
        ],
        "safe_text": policy_snippet("returns"),
    }

    if order_id:
        order = lookup_order(order_id)
        if order and order["within_return_window"]:
            facts.update({"order_id": order_id, "item": order["item"]})
            return {
                "awaiting": None,
                "template": "returns_eligible",
                "facts": facts,
                "grounding": grounding,
                "suggestions": ["Shipping info", "Track my order", "Main menu"],
            }
        if order:
            facts.update({"order_id": order_id, "item": order["item"]})
            return {
                "awaiting": None,
                "template": "returns_expired",
                "facts": facts,
                "grounding": grounding,
                "suggestions": ["Talk to a human", "Main menu"],
            }
        facts.update({"order_id": order_id})
        return {
            "awaiting": None,
            "template": "returns_unknown_order",
            "facts": facts,
            "grounding": grounding,
            "suggestions": ["#111", "#333", "Main menu"],
        }

    return {
        "awaiting": None,
        "template": "returns_policy_only",
        "facts": facts,
        "grounding": grounding,
        "suggestions": ["Shipping info", "Track my order", "Main menu"],
    }


def shipping_agent_node(state: ChatState) -> dict[str, Any]:
    shipping = get_policies()["shipping"]
    return {
        "awaiting": None,
        "template": "shipping",
        "facts": {"shipping_snippet": policy_snippet("shipping")},
        "grounding": {
            "must_include": [shipping["standard"], shipping["expedited"]],
            "safe_text": policy_snippet("shipping"),
        },
        "suggestions": ["Track my order", "Return policy", "Main menu"],
    }


def _rec_result(query: str, limit: int = 3) -> dict[str, Any]:
    return recommendation_chain.invoke({"query": query, "limit": limit})


def _rec_updates(result: dict[str, Any], suggestions: list[str]) -> dict[str, Any]:
    products = result.get("products") or []
    if not products:
        return {
            "awaiting": None,
            "rec_answers": [],
            "template": "recommendations_empty",
            "facts": {},
            "grounding": {},
            "suggestions": suggestions,
        }
    return {
        "awaiting": None,
        "rec_answers": [],
        "template": "recommendations",
        "facts": {"picks": result["picks"]},
        "grounding": {
            "must_include": [products[0]["name"]],
            "safe_text": result["picks"],
        },
        "suggestions": suggestions,
    }


def recommendation_agent_node(state: ChatState) -> dict[str, Any]:
    message = state.get("user_message") or ""
    awaiting = state.get("awaiting")
    answers = list(state.get("rec_answers") or [])

    if awaiting is None:
        result = _rec_result(message)
        lowered = message.lower()
        if len(result.get("products") or []) >= 2 and any(
            token in lowered for token in REC_HINT_TOKENS
        ):
            return _rec_updates(
                result, ["Another recommendation", "Track my order", "Main menu"]
            )
        return {
            "awaiting": "rec_use_case",
            "rec_answers": [],
            "template": "rec_ask_use_case",
            "facts": {},
            "suggestions": ["Sleeping bags", "Tents", "Footwear", "Main menu"],
        }

    if awaiting == "rec_use_case":
        answers.append(message)
        return {
            "rec_answers": answers,
            "awaiting": "rec_preference",
            "template": "rec_ask_preference",
            "facts": {},
            "suggestions": [
                "Sub-zero winter",
                "Weekend backpacking",
                "Family camping",
                "Main menu",
            ],
        }

    # awaiting == "rec_preference"
    answers.append(message)
    result = _rec_result(" ".join(answers))
    if not result.get("products"):
        result = _rec_result(message)
    return _rec_updates(result, ["Another recommendation", "Return policy", "Main menu"])


def escalation_agent_node(state: ChatState) -> dict[str, Any]:
    return {
        "mode": "live_agent",
        "awaiting": None,
        "fallback_count": 0,
        "template": "handoff",
        "facts": {},
        "suggestions": ["Main menu", "Return to bot"],
    }


def fallback_agent_node(state: ChatState) -> dict[str, Any]:
    count = int(state.get("fallback_count") or 0) + 1
    if count >= 2:
        return {
            "fallback_count": count,
            "template": "fallback_escalate",
            "facts": {},
            "suggestions": ["Talk to a human", "Track my order", "Return policy", "Main menu"],
        }
    return {
        "fallback_count": count,
        "template": "fallback_first",
        "facts": {},
        "suggestions": list(DEFAULT_SUGGESTIONS),
    }


def respond_node(state: ChatState) -> dict[str, Any]:
    template = state.get("template") or "fallback_first"
    reply = response_chain.invoke(
        {
            "template": template,
            "facts": state.get("facts") or {},
            "grounding": state.get("grounding") or {},
        }
    )
    messages = list(state.get("messages") or [])
    messages.append({"role": "assistant", "content": reply})
    return {"reply": reply, "messages": messages}


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #
def route_after_ingest(state: ChatState) -> str:
    return "live_agent" if state.get("mode") == "live_agent" else "router"


def route_after_live_agent(state: ChatState) -> str:
    return "menu_agent" if state.get("intent") == "menu" else "respond"


INTENT_TO_NODE = {
    "menu": "menu_agent",
    "handoff": "escalation_agent",
    "order_tracking": "order_agent",
    "returns": "returns_agent",
    "shipping": "shipping_agent",
    "recommendations": "recommendation_agent",
    "fallback": "fallback_agent",
}


def route_after_router(state: ChatState) -> str:
    return INTENT_TO_NODE.get(str(state.get("intent")), "fallback_agent")


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
AGENT_NODES = (
    "menu_agent",
    "order_agent",
    "returns_agent",
    "shipping_agent",
    "recommendation_agent",
    "escalation_agent",
    "fallback_agent",
)


def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("ingest", ingest_node)
    builder.add_node("live_agent", live_agent_node)
    builder.add_node("router", router_node)
    builder.add_node("menu_agent", menu_node)
    builder.add_node("order_agent", order_agent_node)
    builder.add_node("returns_agent", returns_agent_node)
    builder.add_node("shipping_agent", shipping_agent_node)
    builder.add_node("recommendation_agent", recommendation_agent_node)
    builder.add_node("escalation_agent", escalation_agent_node)
    builder.add_node("fallback_agent", fallback_agent_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "ingest")
    builder.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {"live_agent": "live_agent", "router": "router"},
    )
    builder.add_conditional_edges(
        "live_agent",
        route_after_live_agent,
        {"menu_agent": "menu_agent", "respond": "respond"},
    )
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {name: name for name in AGENT_NODES},
    )
    for name in AGENT_NODES:
        builder.add_edge(name, "respond")
    builder.add_edge("respond", END)

    return builder.compile()


graph = build_graph()


def describe_graph() -> dict[str, Any]:
    return {
        "framework": "langgraph",
        "nodes": [
            "ingest",
            "live_agent",
            "router",
            *AGENT_NODES,
            "respond",
        ],
    }


def run_turn(state: ChatState, message: str) -> ChatState:
    """Stable entry point: run one conversation turn through the LangGraph app."""
    incoming = deepcopy(dict(state))
    incoming["user_message"] = (message or "").strip()
    try:
        result = graph.invoke(incoming)
    except Exception:
        logger.exception("graph execution failed, emitting deterministic fallback")
        messages = list(incoming.get("messages") or [])
        messages.append({"role": "user", "content": incoming["user_message"]})
        reply = render_reply("fallback_first")
        messages.append({"role": "assistant", "content": reply})
        degraded = dict(incoming)
        degraded.update(
            {
                "messages": messages,
                "intent": "fallback",
                "reply": reply,
                "suggestions": list(DEFAULT_SUGGESTIONS),
            }
        )
        return degraded  # type: ignore[return-value]

    merged = dict(incoming)
    merged.update(result)
    return merged  # type: ignore[return-value]
