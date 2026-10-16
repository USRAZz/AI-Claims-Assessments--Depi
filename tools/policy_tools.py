"""
policy_tools.py: the three policy tools used by the Policy worker.

All three are plain Python (no LLM). The deductible math is deterministic.
In workers.py they are wrapped with function_tool(...) so the agent can call them.
They return plain dicts whose keys match the fields of PolicyEvidence.
"""

from tools.common import ToolDataError, check_claim_type, get_customer


def get_customer_policy(customer_id: str) -> dict:
    """Return the customer's policy number, status and dates."""
    policy = get_customer(customer_id)["policy"]
    return {
        "policy_number": policy["policy_number"],
        "policy_status": policy["status"],  # active / expired / cancelled
        "policy_start_date": policy["start_date"],
        "policy_end_date": policy["end_date"],
    }


def get_policy_coverage(customer_id: str, claim_type: str, description: str = "") -> dict:
    """Return coverage for a claim type.

    is_covered only says "this claim type is part of the policy". It is NOT a
    coverage decision: exclusions that match the description are listed
    separately in exclusions_matched, and a human decides.
    """
    check_claim_type(claim_type)
    coverage = get_customer(customer_id)["policy"]["coverage"].get(claim_type)
    if coverage is None:
        return {
            "is_covered": False,
            "coverage_limit": 0.0,
            "deductible": 0.0,
            "exclusions_matched": [],
        }
    text = (description or "").lower()
    matched = [e for e in coverage.get("exclusions", []) if e.lower() in text]
    return {
        "is_covered": bool(coverage["covered"]),
        "coverage_limit": float(coverage["limit"]),
        "deductible": float(coverage["deductible"]),
        "exclusions_matched": matched,
    }


def calculate_deductible(claimed_amount: float, deductible: float, coverage_limit: float) -> dict:
    """Pure-Python estimate of who pays what. It is an estimate, not a promise."""
    for name, value in (
        ("claimed_amount", claimed_amount),
        ("deductible", deductible),
        ("coverage_limit", coverage_limit),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ToolDataError(f"{name} must be a number")
    if claimed_amount <= 0:
        raise ToolDataError("claimed_amount must be greater than 0")
    if deductible < 0 or coverage_limit < 0:
        raise ToolDataError("deductible and coverage_limit cannot be negative")

    insurer_share = min(max(claimed_amount - deductible, 0.0), coverage_limit)
    customer_share = claimed_amount - insurer_share
    return {
        "claimed_amount": float(claimed_amount),
        "estimated_insurer_share": round(insurer_share, 2),
        "estimated_customer_share": round(customer_share, 2),
    }