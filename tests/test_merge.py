from datetime import date

from claims_agent.models import ClaimFacts, DocsEvidence, HistoryEvidence, PolicyEvidence
from pipeline.merge import merge_evidence


def make_facts(missing_fields=None):
    return ClaimFacts(
        claim_id="CL-TEST", customer_id="C001", claim_type="auto",
        incident_date=date(2026, 10, 1), claimed_amount=8000,
        description="Rear-ended at a traffic light",
        missing_fields=missing_fields or [],
    )


POLICY = PolicyEvidence(
    policy_number="P-1001", policy_status="active", policy_start_date=date(2025, 1, 1),
    is_covered=True, coverage_limit=50000, deductible=1000,
)
HISTORY = HistoryEvidence(previous_claims_count=1, claims_last_90_days=0)
DOCS = DocsEvidence(required_documents=["photos"], received_documents=["photos"], is_complete=True)


def test_all_workers_ok_means_no_gaps():
    b = merge_evidence(make_facts(), {"policy": POLICY, "history": HISTORY, "docs": DOCS})
    assert b.gaps == []
    assert b.policy is POLICY and b.history is HISTORY and b.docs is DOCS


def test_failed_worker_becomes_a_gap_not_invented_data():
    b = merge_evidence(
        make_facts(),
        {"policy": POLICY, "history": Exception("tool error"), "docs": DOCS},
    )
    assert b.history is None
    assert b.policy is POLICY
    assert any("history check failed" in g for g in b.gaps)


def test_worker_not_requested_becomes_a_gap():
    b = merge_evidence(make_facts(), {"policy": POLICY, "docs": DOCS})
    assert b.history is None
    assert any("history check was not requested" in g for g in b.gaps)


def test_missing_customer_details_are_reported():
    b = merge_evidence(
        make_facts(["location"]), {"policy": POLICY, "history": HISTORY, "docs": DOCS}
    )
    assert any("location" in g for g in b.gaps)
