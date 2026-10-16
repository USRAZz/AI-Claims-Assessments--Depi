"""
router.py: the ROUTING step (code-orchestrated, plain Python, no LLM).

Why it exists: the course requires a routing decision that is auditable and
safety-critical, so it is written as explicit rules, not left to a model.

risk_route() looks at the claim facts (from the Intake agent) and returns:
  - HumanReviewNeeded -> stop the automatic flow, a human must review
  - None              -> routine claim, continue to the orchestrator and workers
"""

from datetime import date
from typing import Optional

from claims_agent.models import ClaimFacts, HumanReviewNeeded
from tools.common import ToolDataError
from tools.history_tools import check_claim_history
from tools.policy_tools import get_customer_policy

# Claims above this amount always go to a human. Change it here if needed.
HIGH_AMOUNT_THRESHOLD = 100_000


def risk_route(facts: ClaimFacts, reference_date: str = "") -> Optional[HumanReviewNeeded]:
    """Apply the routing rules. Collects every reason, then decides once."""
    reasons: list[str] = []
    flags: list[str] = []

    # Rule 1: the Intake agent already raised risk flags (e.g. manipulation).
    if facts.risk_flags:
        flags.append("INTAKE_FLAGGED")
        reasons.append(f"Intake flagged: {', '.join(facts.risk_flags)}")

    # Rule 2: very large amount.
    if facts.claimed_amount > HIGH_AMOUNT_THRESHOLD:
        flags.append("HIGH_AMOUNT")
        reasons.append(
            f"Claimed amount {facts.claimed_amount:,.0f} is above the "
            f"{HIGH_AMOUNT_THRESHOLD:,} threshold"
        )

    # Rules 3-5: policy checks (we never guess: if it cannot be verified, a human looks).
    try:
        policy = get_customer_policy(facts.customer_id)
    except ToolDataError as exc:
        flags.append("POLICY_NOT_VERIFIED")
        reasons.append(f"Policy could not be verified: {exc}")
    else:
        if policy["policy_status"] != "active":
            flags.append("POLICY_NOT_ACTIVE")
            reasons.append(f"Policy status is '{policy['policy_status']}'")

        start = date.fromisoformat(policy["policy_start_date"])
        end = date.fromisoformat(policy["policy_end_date"])
        if facts.incident_date < start:
            flags.append("INCIDENT_BEFORE_POLICY_START")
            reasons.append(f"Incident date {facts.incident_date} is before the policy start {start}")
        elif facts.incident_date > end:
            flags.append("INCIDENT_AFTER_POLICY_END")
            reasons.append(f"Incident date {facts.incident_date} is after the policy end {end}")

    # Rule 6: suspicious claim history (3+ claims in the last 90 days).
    try:
        history = check_claim_history(facts.customer_id, reference_date)
    except ToolDataError:
        # Unknown customer is already flagged by the policy check above.
        history = None
    if history and history["suspicious_pattern"]:
        flags.append("SUSPICIOUS_HISTORY")
        reasons.append(history["notes"])

    if not reasons:
        return None  # routine claim: continue automatically

    return HumanReviewNeeded(
        claim_id=facts.claim_id,
        reason="; ".join(reasons),
        risk_flags=flags,
    )