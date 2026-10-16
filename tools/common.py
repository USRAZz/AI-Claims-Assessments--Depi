"""
common.py: helpers shared by policy_tools, history_tools and docs_tools.

Tools read the synthetic data in data/mock_data.json. If anything is missing
or wrong they raise ToolDataError (never invent data). The workers/merge step
catch it and record a "gap".
"""

import json
from functools import lru_cache
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "mock_data.json"
VALID_CLAIM_TYPES = {"auto", "health", "property", "travel", "other"}


class ToolDataError(Exception):
    """Raised when data cannot be found or an argument is invalid."""


@lru_cache(maxsize=1)
def load_data() -> dict:
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolDataError(f"Cannot read mock data file: {exc}") from exc


def get_customer(customer_id: str) -> dict:
    if not isinstance(customer_id, str) or not customer_id.strip():
        raise ToolDataError("customer_id must be a non-empty string")
    customer = load_data().get("customers", {}).get(customer_id.strip())
    if customer is None:
        raise ToolDataError(f"Customer '{customer_id}' was not found")
    return customer


def check_claim_type(claim_type: str) -> str:
    if claim_type not in VALID_CLAIM_TYPES:
        raise ToolDataError(
            f"claim_type must be one of {sorted(VALID_CLAIM_TYPES)}, got '{claim_type}'"
        )
    return claim_type
