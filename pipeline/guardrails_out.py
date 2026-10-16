import re
from claims_agent.models import ClaimSummary, FORBIDDEN_PHRASES

EXTRA_PATTERNS = [r"\bwill be (approved|paid|refunded)\b", r"\bguarantee"]


def check_output(summary: ClaimSummary) -> list[str]:
    """Return a list of problems. Empty list = safe."""
    texts = [summary.summary, summary.coverage_evidence, *summary.gaps, *summary.risk_flags, *summary.missing_documents]
    problems = []
    for text in texts:
        low = text.lower()
        problems += [f"Forbidden phrase: '{p}'" for p in FORBIDDEN_PHRASES if p in low]
        problems += [f"Promise pattern: {p}" for p in EXTRA_PATTERNS if re.search(p, low)]
    if summary.status not in ("PENDING-ADJUSTER", "AI-ASSESSED"):
        problems.append("Invalid status")
    return problems