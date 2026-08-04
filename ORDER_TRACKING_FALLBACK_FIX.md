# Review fix — unrelated "Where is my …?" no longer enters order tracking (Aug 4, 2026)

### The feedback

> Refine fallback handling for invalid tracking inputs: During the order tracking phase, if a
> user types a query containing random or unrelated text like "Where is my \<abc\>?", the
> chatbot currently misinterprets it as a valid order tracking request and prompts for an
> order number. Please update the intent recognition logic so these arbitrary inputs are
> recognized as out-of-scope and trigger appropriate fallback handling rather than
> proceeding into the order tracking flow.

### What was happening

The local TF-IDF + LogisticRegression classifier learned strong weights for "where" / "is" /
"my" from training examples like "where is my order". Any "Where is my \<noun\>?" message
cleared the confidence threshold and was routed to the order-tracking agent, which then
asked for an order number — even when the noun was nonsense or unrelated.

**Before:**

```
USER: Where is my <abc>?
BOT : Happy to track that for you. What's your order number? (Try #111, #222, or #333 for a demo.)

USER: Where is my cat?
BOT : Happy to track that for you. What's your order number? ...
```

**After:**

```
USER: Where is my <abc>?
BOT : I didn't quite get that. I can help with order tracking, returns, shipping info, ...

USER: Where is my cat?
BOT : I didn't quite get that. I can help with order tracking, returns, shipping info, ...

USER: Where is my order?
BOT : Happy to track that for you. What's your order number? (Try #111, #222, or #333 for a demo.)
```

### What changed

- **`backend/app/agents/router.py`** — After the classifier predicts `order_tracking`, a
  guard checks for "where is/has/did my …" phrasing. If the message has no order-domain
  noun (`order`, `package`, `parcel`, `shipment`, `delivery`, `tracking`, `stuff`, `box`,
  etc.), the turn is forced to `fallback` (`classifier:unrelated_tracking`) instead of
  entering the tracking flow.
- **`backend/app/agents/training_data.py`** — Added negative/fallback examples such as
  "where is my cat", "where is my \<abc\>", "where's my dog", "where is my pizza" so the
  model learns that the object noun matters, not just the "where is my" shell.
- **`backend/offline_test.py`** — New regression test
  `test_unrelated_where_is_my_routes_to_fallback` covers unrelated phrasing (must fall
  back, must not ask for an order number) and confirms valid tracking phrasing still
  routes to `order_tracking`. The low-confidence fixture phrase was updated because the
  expanded fallback corpus slightly shifted classifier scores.

### What did not change

- Valid order-tracking phrasing ("Where is my order?", "where's my package", "where's my
  stuff", paraphrases with parcel/delivery/etc.) still enters the tracking flow and asks
  for an order number.
- Deterministic overrides (bare numbers, `#111`, "order 222", slot-filling while awaiting
  an order id) are unchanged.
- Returns, shipping, recommendations, human handoff, menu, and existing fallback behaviour
  are untouched. `DEMO_SCRIPT.md` still works as written.

### Testing

- `backend/offline_test.py` — **All 16 offline tests pass**, including the new unrelated-
  tracking regression.
- `backend/smoke_test.py` — **All smoke tests pass.**
