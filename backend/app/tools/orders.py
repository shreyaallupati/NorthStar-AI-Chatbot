from __future__ import annotations

import re

from app.db import get_order

ORDER_PATTERN = re.compile(r"(?:order\s*)?#?\s*(\d{3,})", re.IGNORECASE)


def extract_order_id(text: str) -> str | None:
    match = ORDER_PATTERN.search(text)
    if not match:
        return None
    return match.group(1)


def lookup_order(order_id: str) -> dict | None:
    return get_order(order_id)
