"""
merge.py: combine the worker results (plain Python, no LLM).

merge_evidence() builds the EvidenceBundle. If a worker failed (its result is an
Exception) or was not requested, the field stays None and a sentence is added to
`gaps`. We record a gap instead of inventing data.
"""

from claims_agent.models import (
    ClaimFacts,
    DocsEvidence,
    EvidenceBundle,
    HistoryEvidence,
    PolicyEvidence,
)

_EVIDENCE_TYPES = {
    "policy": PolicyEvidence,
    "history": HistoryEvidence,
    "docs": DocsEvidence,
}


def merge_evidence(facts: ClaimFacts, results: dict) -> EvidenceBundle:
    """results maps a worker name ("policy", "history", "docs") to its evidence
    object, or to an Exception if that worker failed. A missing key means the
    orchestrator did not request that worker."""
    gaps: list[str] = []
    found = {}

    for name, evidence_type in _EVIDENCE_TYPES.items():
        if name not in results:
            gaps.append(f"{name} check was not requested by the orchestrator")
        elif isinstance(results[name], evidence_type):
            found[name] = results[name]
        else:
            gaps.append(f"{name} check failed: {results[name]}")

    if facts.missing_fields:
        gaps.append(
            "Details not provided by the customer: " + ", ".join(facts.missing_fields)
        )

    return EvidenceBundle(
        facts=facts,
        policy=found.get("policy"),
        history=found.get("history"),
        docs=found.get("docs"),
        gaps=gaps,
    )