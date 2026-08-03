# Changelog

## Review fix — order tracking context is cleared after each lookup (Aug 3, 2026)

### The feedback

> Once a user enters a valid order number (e.g., 111) and receives their order status, the
> chatbot retains that order number in its session memory. Subsequent queries like "Where is
> my order?" immediately re-display the status for order 111 without starting a new lookup.
> The chatbot should clear the active order context after displaying the status and prompt
> for a new order number whenever a fresh order tracking request is initiated.

### What was happening

After a successful lookup, the bot kept the order number in session memory as a convenience,
so a follow-up question with no number in it ("Where is my order?") silently reused the
previous order instead of asking for a new one.

**Before:**

```
USER: Where is my order?
BOT : Happy to track that for you. What's your order number? (Try #111, #222, or #333 for a demo.)

USER: 111
BOT : Order #111 (Alpine Ridge Tent (2-person)) is shipped - arriving tomorrow. Safe travels for your gear!

USER: Where is my order?
BOT : Order #111 (Alpine Ridge Tent (2-person)) is shipped - arriving tomorrow. ...   <- stale replay
```

**After:**

```
USER: Where is my order?
BOT : Happy to track that for you. What's your order number? (Try #111, #222, or #333 for a demo.)

USER: 111
BOT : Order #111 (Alpine Ridge Tent (2-person)) is shipped - arriving tomorrow. Safe travels for your gear!

USER: Where is my order?
BOT : Happy to track that for you. What's your order number? (Try #111, #222, or #333 for a demo.)   <- fresh lookup

USER: 222
BOT : Order #222 (Trailblazer Hiking Boots) is still processing - ships in 24 hours. ...
```

### What changed

All changes are in the conversation engine (`backend/app/graph/workflow.py`):

- **Order context is cleared once a status is shown.** A fresh tracking request now always
  asks for an order number and starts a new lookup.
- **Lookups only use the number in the current message.** The order agent no longer falls
  back to a previously stored order number.
- **Failed lookups don't linger either.** An invalid order number is not kept in memory; the
  bot simply re-asks (retrying by typing a number still works exactly as before).
- **"Main menu" fully resets the conversation**, including any in-progress order prompt.
- **Returns eligibility checks now require an explicit order number** (e.g. "I want to
  return order 111"), consistent with the demo script. The general return-policy answer is
  unchanged.

Side effect worth noting: the **"Track another order"** suggestion chip now correctly asks
for a new order number instead of replaying the same order's status.

### What did not change

- All four demo orders (#111 shipped, #222 processing, #333 delivered, unknown numbers
  rejected) behave the same.
- The slot-filling flow is intact: ask "Where is my order?", then reply with just `111` and
  the lookup proceeds.
- Returns, shipping, recommendations, human handoff, and fallback flows are untouched, and
  every step of `DEMO_SCRIPT.md` still works as written.

### Testing

- `backend/offline_test.py` — new regression test `test_order_context_cleared_after_status`
  covers the full cycle: lookup #111 -> fresh "Where is my order?" must re-ask (not replay
  the status) -> new lookup #222 succeeds -> "main menu" clears any pending order prompt.
  **All 15 offline tests pass.**
- `backend/smoke_test.py` — added the same re-ask check to the quick smoke suite. **Passes.**

Run them with:

```bash
cd backend
.venv311\Scripts\python offline_test.py
.venv311\Scripts\python smoke_test.py
```
