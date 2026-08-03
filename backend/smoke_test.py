"""Quick smoke test for required use cases (no API keys)."""

from __future__ import annotations

from app.db import init_db
from app.graph.workflow import initial_state, run_turn


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run() -> None:
    init_db(force=True)
    state = initial_state("smoke")

    # Order tracking
    state = run_turn(state, "Where is my order?")
    expect(state["awaiting"] == "order_id", "should ask for order id")
    state = run_turn(state, "111")
    expect("Shipped" in state["reply"] or "shipped" in state["reply"], "111 shipped")
    expect("tomorrow" in state["reply"], "111 arriving tomorrow")

    # Order context is cleared after a status reply: a fresh tracking request
    # must ask for a new number instead of re-showing #111.
    state = run_turn(state, "Where is my order?")
    expect(state["awaiting"] == "order_id", "should re-ask for order id after a completed lookup")
    expect("shipped" not in state["reply"].lower(), "must not re-display the previous status")

    state = run_turn(state, "Track package 222")
    expect("Processing" in state["reply"] or "processing" in state["reply"], "222 processing")

    state = run_turn(state, "order 333")
    expect("Delivered" in state["reply"] or "delivered" in state["reply"], "333 delivered")

    state = run_turn(state, "order 999")
    expect("couldn't find" in state["reply"].lower() or "invalid" in state["reply"].lower() or "double-check" in state["reply"].lower(), "invalid order")

    # Returns
    state = run_turn(state, "What is your return policy?")
    expect("30-day" in state["reply"] or "30-day" in state["reply"].lower() or "30 day" in state["reply"].lower(), "30-day returns")
    expect("unused" in state["reply"].lower(), "unused required")
    expect("northstar.example/returns" in state["reply"], "returns link")

    # Shipping
    state = run_turn(state, "How long is standard shipping?")
    expect("3-5" in state["reply"], "standard shipping")
    expect("1-2" in state["reply"], "expedited shipping")

    # Recommendations
    state = initial_state("smoke-rec")
    state = run_turn(state, "What's the best sleeping bag for sub-zero temperatures?")
    expect("Summit" in state["reply"] or "sleeping" in state["reply"].lower(), "sleeping bag rec")

    # Handoff + return
    state = run_turn(state, "Talk to a human")
    expect(state["mode"] == "live_agent", "live agent mode")
    state = run_turn(state, "main menu")
    expect(state["mode"] == "bot", "return to bot")

    # Fallback
    state = run_turn(state, "asdfgh qwerty")
    expect("didn't" in state["reply"].lower() or "not catching" in state["reply"].lower(), "fallback")

    print("All smoke tests passed.")


if __name__ == "__main__":
    run()
