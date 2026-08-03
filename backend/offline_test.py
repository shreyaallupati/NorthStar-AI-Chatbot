"""Offline test suite for the local agentic stack.

Everything runs with no API keys, no network access, and no external services.

    cd backend
    .venv311\\Scripts\\python offline_test.py

Coverage:
  1. The scikit-learn intent router generalises across phrasings that the old
     regex router could not match at all.
  2. Deterministic overrides (order number / "talk to a human" / "main menu")
     win regardless of the classifier.
  3. Low-confidence and out-of-vocabulary input routes to `fallback`.
  4. The hybrid TF-IDF + BM25 retriever ranks a sub-zero sleeping-bag query
     above unrelated gear.
  5. The LangGraph app has the expected nodes and conditional edges.
  6. The grounding guard replaces any order-tracking reply that drops or
     contradicts the retrieved status with the deterministic reply.
  7. All four required use cases plus fallback, end to end through `run_turn`.
  8. The router still works when scikit-learn is unavailable.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import PromptTemplate

from app.agents.classifier import model
from app.agents.router import classify_intent, keyword_intent, route
from app.chains import prompts
from app.config import settings
from app.db import init_db
from app.graph.workflow import graph, initial_state, run_turn
from app.rag.store import product_index
from app.tools.retrieval import search_products

logging.disable(logging.CRITICAL)


class Failure(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


# --------------------------------------------------------------------------- #
# 1. Classifier generalisation
# --------------------------------------------------------------------------- #
# Phrasings the original keyword router scored as `fallback`, but that a real
# support bot must understand.
PARAPHRASES: list[tuple[str, str]] = [
    ("order_tracking", "my parcel hasn't turned up"),
    ("order_tracking", "any news on my delivery"),
    ("order_tracking", "has my stuff been dispatched"),
    ("returns", "I'd like my money back"),
    ("returns", "can i swap these boots for a bigger size"),
    ("shipping", "how quickly can you get it to me"),
    ("shipping", "how many business days"),
    ("recommendations", "i need something for freezing nights"),
    ("handoff", "put me through to someone"),
    ("handoff", "get me support staff"),
]


def test_classifier_generalises_beyond_regex() -> None:
    expect(model.available(), "scikit-learn classifier should be available in this venv")
    missed_by_regex = 0
    for expected, text in PARAPHRASES:
        actual = classify_intent(text)
        expect(
            actual == expected,
            f"classifier: {text!r} -> {actual!r}, expected {expected!r}",
        )
        if keyword_intent(text) == "fallback":
            missed_by_regex += 1
    expect(
        missed_by_regex >= 8,
        f"expected the keyword router to miss most paraphrases, missed {missed_by_regex}",
    )


def test_classifier_covers_every_label() -> None:
    seen = {
        classify_intent("where has my package got to"),
        classify_intent("i want a refund on this jacket"),
        classify_intent("what are your delivery times"),
        classify_intent("recommend a tent for two"),
        classify_intent("connect me with an agent"),
        classify_intent("main menu"),
        classify_intent("asdfgh qwerty"),
    }
    expect(
        seen
        == {
            "order_tracking",
            "returns",
            "shipping",
            "recommendations",
            "handoff",
            "menu",
            "fallback",
        },
        f"all seven intent labels should be reachable, got {sorted(seen)}",
    )


# --------------------------------------------------------------------------- #
# 2. Deterministic overrides
# --------------------------------------------------------------------------- #
def test_deterministic_overrides_win() -> None:
    cases = [
        ("#111", None, "order_tracking", "override:order_number"),
        ("222", None, "order_tracking", "override:order_number"),
        ("order 999", None, "order_tracking", "override:order_number"),
        ("Track package 222", None, "order_tracking", "override:order_number"),
        ("Talk to a human", None, "handoff", "override:handoff"),
        ("i want to speak to a live agent", None, "handoff", "override:handoff"),
        ("main menu", None, "menu", "override:menu"),
        ("Return to bot", None, "menu", "override:menu"),
        ("111", "order_id", "order_tracking", "override:slot_order_id"),
        ("something warm", "rec_use_case", "recommendations", "override:slot_recommendations"),
        ("sub-zero winter", "rec_preference", "recommendations", "override:slot_recommendations"),
    ]
    for text, awaiting, intent, source in cases:
        decision = route(text, awaiting)
        expect(
            decision.intent == intent and decision.source == source,
            f"override for {text!r} (awaiting={awaiting}) -> "
            f"{decision.intent}/{decision.source}, expected {intent}/{source}",
        )

    # Explicit returns language beats the bare order-number override.
    decision = route("i want to return order 111")
    expect(
        decision.intent == "returns",
        f"explicit returns language should win, got {decision.intent}",
    )


# --------------------------------------------------------------------------- #
# 3. Low confidence / out of vocabulary
# --------------------------------------------------------------------------- #
def test_low_confidence_routes_to_fallback() -> None:
    text = "the moon is quite bright tonight friend"
    prediction = model.predict(text)
    expect(prediction is not None, "classifier should return a prediction")
    expect(
        prediction.confidence < settings.intent_confidence_threshold,
        f"expected low confidence for {text!r}, got {prediction.confidence:.3f}",
    )
    decision = route(text)
    expect(
        decision.intent == "fallback" and decision.source == "classifier:low_confidence",
        f"low-confidence input should fall back, got {decision.intent}/{decision.source}",
    )

    decision = route("qwoiejfoiwej zzzplork")
    expect(
        decision.intent == "fallback" and decision.source == "classifier:out_of_vocab",
        f"out-of-vocabulary input should fall back, got {decision.intent}/{decision.source}",
    )


# --------------------------------------------------------------------------- #
# 4. Retrieval quality
# --------------------------------------------------------------------------- #
def test_retriever_ranks_subzero_bag_first() -> None:
    scored = product_index().score("sleeping bag for sub-zero temperatures")
    names = [doc.metadata["payload"]["name"] for _, doc in scored]
    expect(
        names[0] == "Summit Down Sleeping Bag",
        f"sub-zero query should rank the -20F bag first, got {names[:3]}",
    )

    by_name = {doc.metadata["payload"]["name"]: score for score, doc in scored}
    for unrelated in ("GlowForge Camp Stove", "Trailblazer Hiking Boots", "Basecamp Family Dome"):
        expect(
            by_name["Summit Down Sleeping Bag"] > by_name[unrelated],
            f"Summit Down should outrank {unrelated}",
        )

    top = [p["name"] for p in search_products("sleeping bag for sub-zero temperatures", limit=3)]
    expect("GlowForge Camp Stove" not in top, f"camp stove should not be a sleeping-bag hit: {top}")

    expect(
        search_products("family car camping tent", limit=1)[0]["name"] == "Basecamp Family Dome",
        "family car camping should retrieve the 4-person dome",
    )
    expect(
        search_products("hiking boots", limit=1)[0]["name"] == "Trailblazer Hiking Boots",
        "boots query should retrieve the hiking boots",
    )


# --------------------------------------------------------------------------- #
# 5. LangGraph structure
# --------------------------------------------------------------------------- #
def test_graph_structure() -> None:
    drawable = graph.get_graph()
    nodes = set(drawable.nodes)
    for name in (
        "ingest",
        "router",
        "live_agent",
        "order_agent",
        "returns_agent",
        "shipping_agent",
        "recommendation_agent",
        "escalation_agent",
        "fallback_agent",
        "respond",
    ):
        expect(name in nodes, f"graph should contain node {name!r}, got {sorted(nodes)}")

    conditional = [edge for edge in drawable.edges if edge.conditional]
    expect(len(conditional) >= 9, f"expected conditional edges, found {len(conditional)}")


# --------------------------------------------------------------------------- #
# 6. Grounding guard
# --------------------------------------------------------------------------- #
def test_grounding_guard_overrides_ungrounded_order_reply() -> None:
    original = prompts.TEMPLATES["order_shipped"]
    try:
        # Simulate a rewrite that silently drops the retrieved status/detail.
        prompts.TEMPLATES["order_shipped"] = PromptTemplate.from_template(
            "Great news about order #{order_id} for your {item} - it is out for delivery today!"
        )
        state = initial_state("guard")
        state = run_turn(state, "#111")
        expect(
            state["reply"] == "Order #111: Shipped - arriving tomorrow.",
            f"guard should emit the deterministic reply, got {state['reply']!r}",
        )
    finally:
        prompts.TEMPLATES["order_shipped"] = original

    # Sanity check: with the real template the guard stays out of the way.
    state = initial_state("guard-ok")
    state = run_turn(state, "#111")
    expect("shipped" in state["reply"].lower(), "normal shipped reply should survive the guard")
    expect("arriving tomorrow" in state["reply"], "normal shipped reply keeps the detail")


# --------------------------------------------------------------------------- #
# 7. End-to-end use cases
# --------------------------------------------------------------------------- #
def test_use_case_order_tracking() -> None:
    state = initial_state("e2e-orders")

    state = run_turn(state, "my parcel hasn't turned up")
    expect(state["intent"] == "order_tracking", f"intent was {state['intent']}")
    expect(state["awaiting"] == "order_id", "should ask for the order number")

    state = run_turn(state, "#111")
    expect("shipped" in state["reply"].lower(), f"#111 must be Shipped: {state['reply']!r}")
    expect("arriving tomorrow" in state["reply"], "#111 must arrive tomorrow")
    expect(state["awaiting"] is None, "slot should be cleared once resolved")

    state = run_turn(state, "Track package 222")
    expect("processing" in state["reply"].lower(), f"#222 must be Processing: {state['reply']!r}")
    expect("ships in 24 hours" in state["reply"], "#222 must ship in 24 hours")

    state = run_turn(state, "order 333")
    expect("delivered" in state["reply"].lower(), f"#333 must be Delivered: {state['reply']!r}")
    expect("?" in state["reply"], "#333 should ask a follow-up question")

    state = run_turn(state, "order 999")
    expect("couldn't find" in state["reply"].lower(), f"invalid order: {state['reply']!r}")
    expect(state["awaiting"] == "order_id", "invalid order should re-ask")


def test_order_context_cleared_after_status() -> None:
    state = initial_state("e2e-order-context")

    state = run_turn(state, "Where is my order?")
    expect(state["awaiting"] == "order_id", "should ask for the order number")

    state = run_turn(state, "111")
    expect("shipped" in state["reply"].lower(), f"#111 must be Shipped: {state['reply']!r}")
    expect(state.get("order_id") is None, "order context must be cleared once status is shown")

    # A fresh tracking request must start a new lookup, not re-display #111.
    state = run_turn(state, "Where is my order?")
    expect(state["intent"] == "order_tracking", f"intent was {state['intent']}")
    expect(state["awaiting"] == "order_id", "should prompt for a new order number")
    expect("order number" in state["reply"].lower(), f"should ask for a number: {state['reply']!r}")
    expect("shipped" not in state["reply"].lower(), "must not re-show the previous status")

    # The new lookup then proceeds normally.
    state = run_turn(state, "#222")
    expect("processing" in state["reply"].lower(), f"#222 must be Processing: {state['reply']!r}")

    # "Main menu" also clears any in-progress order slot.
    state = run_turn(state, "Where is my order?")
    expect(state["awaiting"] == "order_id", "should ask for the order number again")
    state = run_turn(state, "main menu")
    expect(state.get("order_id") is None, "menu must clear the order context")
    expect(state["awaiting"] is None, "menu must clear the pending slot")


def test_use_case_returns() -> None:
    state = initial_state("e2e-returns")

    state = run_turn(state, "I'd like my money back")
    expect(state["intent"] == "returns", f"intent was {state['intent']}")
    reply = state["reply"]
    expect("30-day" in reply, f"30-day window: {reply!r}")
    expect("unused" in reply.lower(), "unused items required")
    expect("original packaging" in reply.lower(), "original packaging required")
    expect("https://northstar.example/returns" in reply, "returns link required")

    state = initial_state("e2e-returns-eligible")
    state = run_turn(state, "can i return order 111")
    expect("within the return window" in state["reply"], f"#111 eligible: {state['reply']!r}")

    state = initial_state("e2e-returns-expired")
    state = run_turn(state, "i want to return order 333")
    expect("outside the 30-day window" in state["reply"], f"#333 expired: {state['reply']!r}")


def test_use_case_shipping() -> None:
    state = initial_state("e2e-shipping")
    state = run_turn(state, "how quickly can you get it to me")
    expect(state["intent"] == "shipping", f"intent was {state['intent']}")
    expect("3-5 business days" in state["reply"], f"standard: {state['reply']!r}")
    expect("1-2 business days" in state["reply"], f"expedited: {state['reply']!r}")


def test_use_case_recommendations() -> None:
    # Vague opener: the bot must ask clarifying questions before recommending.
    state = initial_state("e2e-recs")
    state = run_turn(state, "can you recommend something")
    expect(state["intent"] == "recommendations", f"intent was {state['intent']}")
    expect(state["awaiting"] == "rec_use_case", "should ask what they're shopping for")

    state = run_turn(state, "sleeping bags")
    expect(state["awaiting"] == "rec_preference", "should ask one more clarifying question")

    state = run_turn(state, "sub-zero winter camping")
    expect("Summit Down Sleeping Bag" in state["reply"], f"picks: {state['reply']!r}")
    expect("-20" in state["reply"], "sub-zero bag rating should be shown")
    expect(state["awaiting"] is None, "slots cleared after recommending")

    # Detailed opener: enough signal to recommend immediately.
    state = initial_state("e2e-recs-direct")
    state = run_turn(state, "What's the best sleeping bag for sub-zero temperatures?")
    expect("Summit Down Sleeping Bag" in state["reply"], f"direct picks: {state['reply']!r}")
    expect(
        "GlowForge Camp Stove" not in state["reply"],
        "unrelated gear should not be recommended",
    )


def test_use_case_handoff_and_menu() -> None:
    state = initial_state("e2e-handoff")
    state = run_turn(state, "put me through to someone")
    expect(state["mode"] == "live_agent", f"mode was {state['mode']}")
    expect("Live Agent" in state["reply"], f"handoff copy: {state['reply']!r}")

    state = run_turn(state, "are you there")
    expect(state["mode"] == "live_agent", "should stay with the live agent")
    expect("Live Agent is still with you" in state["reply"], f"ack copy: {state['reply']!r}")

    state = run_turn(state, "main menu")
    expect(state["mode"] == "bot", f"mode was {state['mode']}")
    expect("Pick a trail" in state["reply"], f"menu copy: {state['reply']!r}")

    # A paraphrased request also gets the user back to the main menu.
    state = run_turn(state, "live agent")
    expect(state["mode"] == "live_agent", "should be handed off again")
    state = run_turn(state, "i want the bot again")
    expect(state["mode"] == "bot", f"paraphrased menu request should return control: {state!r}")

    # The bot can also escalate on its own after repeated misunderstandings.
    state = initial_state("e2e-escalate")
    state = run_turn(state, "asdfgh qwerty")
    expect(state["fallback_count"] == 1, "first miss")
    state = run_turn(state, "zxcvbn hjkl")
    expect(state["fallback_count"] == 2, "second miss")
    expect("live agent" in state["reply"].lower(), f"should offer escalation: {state['reply']!r}")
    expect("Talk to a human" in state["suggestions"], "escalation should be one click away")


def test_use_case_fallback() -> None:
    state = initial_state("e2e-fallback")
    state = run_turn(state, "asdfgh qwerty")
    expect(state["intent"] == "fallback", f"intent was {state['intent']}")
    expect("didn't quite get that" in state["reply"], f"fallback copy: {state['reply']!r}")
    expect(len(state["suggestions"]) >= 3, "fallback must offer options")

    # A recognised intent resets the counter.
    state = run_turn(state, "how long does shipping take")
    expect(state["fallback_count"] == 0, "recognised intent resets fallback_count")


# --------------------------------------------------------------------------- #
# 8. Degradation when scikit-learn is missing
# --------------------------------------------------------------------------- #
def test_keyword_router_fallback() -> None:
    original = model.predict
    try:
        model.predict = lambda message: None  # type: ignore[assignment]
        decision = route("Where is my order?")
        expect(
            decision.intent == "order_tracking" and decision.source == "keyword-fallback",
            f"keyword fallback should handle routing, got {decision.intent}/{decision.source}",
        )
        state = initial_state("no-sklearn")
        state = run_turn(state, "What is your return policy?")
        expect("30-day" in state["reply"], f"keyword path still grounded: {state['reply']!r}")
    finally:
        model.predict = original  # type: ignore[assignment]


TESTS = [
    test_classifier_generalises_beyond_regex,
    test_classifier_covers_every_label,
    test_deterministic_overrides_win,
    test_low_confidence_routes_to_fallback,
    test_retriever_ranks_subzero_bag_first,
    test_graph_structure,
    test_grounding_guard_overrides_ungrounded_order_reply,
    test_use_case_order_tracking,
    test_order_context_cleared_after_status,
    test_use_case_returns,
    test_use_case_shipping,
    test_use_case_recommendations,
    test_use_case_handoff_and_menu,
    test_use_case_fallback,
    test_keyword_router_fallback,
]


def run() -> int:
    init_db(force=True)
    failures = 0
    for test in TESTS:
        name = test.__name__
        try:
            test()
        except Failure as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected error
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    total = len(TESTS)
    if failures:
        print(f"\n{failures}/{total} offline tests FAILED.")
    else:
        print(f"\nAll {total} offline tests passed (no API keys, no network).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
