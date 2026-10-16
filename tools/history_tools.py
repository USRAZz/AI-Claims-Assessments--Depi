"""
history_tools.py: claim-history tool used by the History worker.

Plain Python (no LLM). The "suspicious" rule is deterministic and auditable:
SUSPICIOUS_CLAIMS_IN_90_DAYS or more claims in the last 90 days.
Keys of the returned dict match the fields of HistoryEvidence.
"""

from datetime import date, timedelta

from tools.common import ToolDataError, get_customer

SUSPICIOUS_CLAIMS_IN_90_DAYS = 3


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ToolDataError(f"{label} must be an ISO date (YYYY-MM-DD), got '{value}'") from exc


def check_claim_history(customer_id: str, reference_date: str = "") -> dict:
    """Return how many claims the customer made before, and in the last 90 days.

    reference_date (YYYY-MM-DD) defaults to today; tests pass a fixed date so
    results do not change over time.
    """
    customer = get_customer(customer_id)
    today = _parse_date(reference_date, "reference_date") if reference_date else date.today()
    window_start = today - timedelta(days=90)

    history = customer.get("claim_history", [])
    recent = [
        c for c in history
        if window_start < _parse_date(c["date"], "claim date") <= today
    ]
    suspicious = len(recent) >= SUSPICIOUS_CLAIMS_IN_90_DAYS

    if not history:
        notes = "No previous claims on record."
    else:
        notes = f"{len(history)} previous claim(s), {len(recent)} in the last 90 days."
        if suspicious:
            notes += f" Flagged: {SUSPICIOUS_CLAIMS_IN_90_DAYS}+ claims in 90 days."

    return {
        "previous_claims_count": len(history),
        "claims_last_90_days": len(recent),
        "suspicious_pattern": suspicious,
        "notes": notes,
    }
