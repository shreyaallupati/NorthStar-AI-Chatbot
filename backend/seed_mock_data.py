"""Populate SQLite with outdoor gear mock data."""

from app.db import init_db


if __name__ == "__main__":
    init_db(force=True)
    print("Seeded orders #111/#222/#333, products, and FAQs into SQLite.")
