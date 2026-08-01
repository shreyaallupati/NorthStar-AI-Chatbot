from __future__ import annotations

import json
import sqlite3

from app.config import DATA_DIR, DB_PATH

CATALOG_PATH = DATA_DIR / "catalog.json"


def load_catalog() -> dict:
    with CATALOG_PATH.open(encoding="utf-8-sig") as f:
        return json.load(f)


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(force: bool = False) -> None:
    catalog = load_catalog()
    conn = get_connection()
    try:
        if force:
            conn.executescript(
                "DROP TABLE IF EXISTS orders; DROP TABLE IF EXISTS products; DROP TABLE IF EXISTS faqs;"
            )

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                detail TEXT NOT NULL,
                item TEXT NOT NULL,
                ordered_at TEXT NOT NULL,
                within_return_window INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faqs (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                tags TEXT NOT NULL
            );
            """
        )

        order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        if order_count == 0 or force:
            conn.execute("DELETE FROM orders")
            for order in catalog["orders"]:
                conn.execute(
                    """
                    INSERT INTO orders (order_id, status, detail, item, ordered_at, within_return_window)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order["order_id"],
                        order["status"],
                        order["detail"],
                        order["item"],
                        order["ordered_at"],
                        1 if order["within_return_window"] else 0,
                    ),
                )

        product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if product_count == 0 or force:
            conn.execute("DELETE FROM products")
            for product in catalog["products"]:
                conn.execute(
                    "INSERT INTO products (id, name, category, payload) VALUES (?, ?, ?, ?)",
                    (product["id"], product["name"], product["category"], json.dumps(product)),
                )

        faq_count = conn.execute("SELECT COUNT(*) FROM faqs").fetchone()[0]
        if faq_count == 0 or force:
            conn.execute("DELETE FROM faqs")
            for faq in catalog["faqs"]:
                conn.execute(
                    "INSERT INTO faqs (id, question, answer, tags) VALUES (?, ?, ?, ?)",
                    (faq["id"], faq["question"], faq["answer"], json.dumps(faq["tags"])),
                )

        conn.commit()
    finally:
        conn.close()


def get_order(order_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id.strip().lstrip("#"),),
        ).fetchone()
        if not row:
            return None
        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "detail": row["detail"],
            "item": row["item"],
            "ordered_at": row["ordered_at"],
            "within_return_window": bool(row["within_return_window"]),
        }
    finally:
        conn.close()


def list_products() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT payload FROM products").fetchall()
        return [json.loads(r["payload"]) for r in rows]
    finally:
        conn.close()


def list_faqs() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT id, question, answer, tags FROM faqs").fetchall()
        return [
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "tags": json.loads(r["tags"]),
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_policies() -> dict:
    return load_catalog()["policies"]
