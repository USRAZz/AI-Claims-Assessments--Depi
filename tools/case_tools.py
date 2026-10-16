import sqlite3
from pathlib import Path

from claims_agent.models import ClaimSummary, NeedsManual, HumanReviewNeeded

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cases.db"

# Protected tool: only these statuses can ever be saved. There is no
# "approved" or "rejected": that decision belongs to a human adjuster.
ALLOWED_STATUSES = ("PENDING-ADJUSTER", "AI-ASSESSED")


class CaseSaveError(Exception):
    """The case could not be saved (invalid arguments or the database failed)."""


def _init() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS cases(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
            )
    finally:
        conn.close()


def _validate(result) -> None:
    """Check the arguments BEFORE touching the database."""
    if not isinstance(result, (ClaimSummary, NeedsManual, HumanReviewNeeded)):
        raise CaseSaveError(f"Unsupported result type: {type(result).__name__}")
    if not str(getattr(result, "claim_id", "")).strip():
        raise CaseSaveError("claim_id is required")


def create_claim_case(result: ClaimSummary | NeedsManual | HumanReviewNeeded) -> int:
    """Save a case for a human adjuster. Never approves/rejects anything.

    Returns the case id ONLY when the database confirmed the write.
    Retries once; raises CaseSaveError if both attempts fail."""
    _validate(result)
    status = result.status if isinstance(result, ClaimSummary) else "PENDING-ADJUSTER"
    if status not in ALLOWED_STATUSES:
        status = "PENDING-ADJUSTER"

    last_error = None
    for _attempt in (1, 2):
        try:
            _init()
            conn = sqlite3.connect(DB_PATH)
            try:
                with conn:
                    cur = conn.execute(
                        "INSERT INTO cases(claim_id, status, payload) VALUES (?,?,?)",
                        (result.claim_id, status, result.model_dump_json()),
                    )
                case_id = cur.lastrowid
            finally:
                conn.close()
            if case_id:
                return case_id               # confirmed by the database
            last_error = "insert was not confirmed"
        except (sqlite3.Error, OSError) as exc:
            last_error = str(exc)
    raise CaseSaveError(f"Could not save the case: {last_error}")