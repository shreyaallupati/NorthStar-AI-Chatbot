"""Response templates expressed as LangChain `PromptTemplate` objects.

These are the *only* place user-facing copy lives. Every variable is filled from
SQLite/catalog data or from retrieved catalog documents, so no business fact can
be invented: there is no generative model in the loop at all.

Persona: North Star Support Bot - friendly, outdoorsy, concise, North American.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate


def _template(text: str) -> PromptTemplate:
    return PromptTemplate.from_template(text, template_format="f-string")


WELCOME = (
    "Hey there - I'm North Star Support Bot. "
    "I can help with order tracking, returns, product picks, or connect you to a live agent. "
    "What can I help with today?"
)

MENU_TEXT = (
    "You're back with North Star Support Bot. Pick a trail:\n"
    "\u2022 Track an order\n"
    "\u2022 Returns & exchanges\n"
    "\u2022 Product recommendations\n"
    "\u2022 Talk to a live agent"
)

RETURNS_BASE = (
    "{faq_bit}\n\n"
    "Quick rules: {window_days}-day window, unused items, original packaging required. "
    "Start here: {returns_link}"
)


TEMPLATES: dict[str, PromptTemplate] = {
    "welcome": _template(WELCOME),
    "menu": _template(MENU_TEXT),
    # -- order tracking ---------------------------------------------------
    "order_ask": _template(
        "Happy to track that for you. What's your order number? "
        "(Try #111, #222, or #333 for a demo.)"
    ),
    "order_not_found": _template(
        "I couldn't find order #{order_id}. "
        "Please double-check the number, or try #111, #222, or #333."
    ),
    "order_shipped": _template(
        "Order #{order_id} ({item}) is {status_lower} - {detail}. Safe travels for your gear!"
    ),
    "order_processing": _template(
        "Order #{order_id} ({item}) is still {status_lower} - {detail}. "
        "We'll notify you as soon as it ships."
    ),
    "order_delivered": _template(
        "Order #{order_id} ({item}) was {status_lower}. "
        "Everything look good, or do you need help with a return or something else?"
    ),
    "order_generic": _template("Order #{order_id}: {status} - {detail}."),
    # -- returns ----------------------------------------------------------
    "returns_policy_only": _template(
        RETURNS_BASE
        + "\n\nIf you share an order number, I can check whether that item is still eligible."
    ),
    "returns_eligible": _template(
        RETURNS_BASE
        + "\n\nGood news - order #{order_id} ({item}) is still within the return window. "
        "You can generate a mock return label at {returns_link}?order={order_id}."
    ),
    "returns_expired": _template(
        RETURNS_BASE
        + "\n\nOrder #{order_id} ({item}) is outside the {window_days}-day window, "
        "so it isn't eligible for a standard return. "
        "I can connect you to a live agent if you'd like."
    ),
    "returns_unknown_order": _template(
        RETURNS_BASE
        + "\n\nI couldn't find order #{order_id}. You can still review the policy above, "
        "or share a valid order number."
    ),
    # -- shipping ---------------------------------------------------------
    "shipping": _template("{shipping_snippet} Tracking updates appear once your order ships."),
    # -- recommendations --------------------------------------------------
    "rec_ask_use_case": _template(
        "Happy to gear you up. What are you shopping for - sleeping bags, tents, apparel, "
        "footwear, or camp kitchen?"
    ),
    "rec_ask_preference": _template(
        "Got it. One more: any conditions or preferences? "
        "(e.g. sub-zero temps, weekend backpacking, family car camping, budget-friendly)"
    ),
    "recommendations": _template(
        "Based on what you shared, here are solid North Star picks:\n{picks}\n"
        "Want different options, or shall I help with something else?"
    ),
    "recommendations_empty": _template(
        "I couldn't find a strong match in the demo catalog. "
        "Try asking about sleeping bags, tents, boots, or jackets - "
        "or talk to a live agent for more options."
    ),
    # -- escalation -------------------------------------------------------
    "handoff": _template(
        "Understood - connecting you to a Live Agent now. "
        "A teammate will join this chat shortly (simulated). "
        'You can keep typing here, or say "main menu" anytime to return to the bot.'
    ),
    "live_agent_ack": _template(
        "A Live Agent is still with you (simulated). "
        "Thanks for your message - they'll follow up shortly. "
        'Say "main menu" to return to the bot anytime.'
    ),
    # -- fallback ---------------------------------------------------------
    "fallback_first": _template(
        "I didn't quite get that. I can help with order tracking, returns, shipping info, "
        "product recommendations - or connect you to a live agent."
    ),
    "fallback_escalate": _template(
        "I'm still not catching that. I can connect you to a live agent, "
        "or you can pick one of the options below."
    ),
}
