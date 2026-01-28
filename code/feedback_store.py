import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "reviews.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT NOT NULL
            )
        """)
        conn.commit()

def save_review(rating: int, comment: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO reviews(created_at, rating, comment) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(), rating, comment.strip())
        )
        conn.commit()

def get_recent_reviews(limit: int = 10):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT created_at, rating, comment FROM reviews ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return rows