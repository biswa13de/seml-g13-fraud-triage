"""SQLite-backed case store. Kept dead simple (one table) since this is a
demo-scale system; a production PSP would use Postgres here."""
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data_runtime/cases.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    txn_id TEXT NOT NULL UNIQUE,
    priority REAL NOT NULL,
    amount REAL NOT NULL,
    risk_score REAL NOT NULL,
    decision TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    verdict TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS labels (
    txn_id TEXT PRIMARY KEY,
    verdict TEXT NOT NULL,
    resolved_at TEXT NOT NULL
);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def create_case(conn: sqlite3.Connection, *, txn_id: str, amount: float, risk_score: float,
                 decision: str, reason_codes: list[str]) -> str | None:
    """Idempotent on txn_id: re-publishing the same event (at-least-once
    delivery) never creates a duplicate case."""
    case_id = f"CASE-{uuid.uuid4().hex[:10]}"
    priority = risk_score * amount
    try:
        conn.execute(
            "INSERT INTO cases (case_id, txn_id, priority, amount, risk_score, decision, "
            "reason_codes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (case_id, txn_id, priority, amount, risk_score, decision,
             "|".join(reason_codes), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return case_id
    except sqlite3.IntegrityError:
        return None  # already exists, duplicate event, fine


def list_cases(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    query = "SELECT * FROM cases"
    params = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY priority DESC"
    rows = conn.execute(query, params).fetchall()
    return [dict(r) | {"reason_codes": r["reason_codes"].split("|")} for r in rows]


def resolve_case(conn: sqlite3.Connection, case_id: str, verdict: str) -> dict | None:
    row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE cases SET status='RESOLVED', verdict=?, resolved_at=? WHERE case_id=?",
        (verdict, now, case_id),
    )
    # The feedback loop (FR6): analyst verdicts become training labels.
    conn.execute(
        "INSERT OR REPLACE INTO labels (txn_id, verdict, resolved_at) VALUES (?, ?, ?)",
        (row["txn_id"], verdict, now),
    )
    conn.commit()
    return dict(conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone())
