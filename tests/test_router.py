from datetime import date

from claims_agent.models import ClaimFacts
from pipeline.router import risk_route

REF = "2026-10-13"  # fixed date so results never change


def make_facts(customer_id="C001", amount=8000, incident=date(2026, 10, 1),
               claim_type="auto", risk_flags=None):
    return ClaimFacts(
        claim_id="CL-TEST",
        customer_id=customer_id,
        claim_type=claim_type,
        incident_date=incident,
        claimed_amount=amount,
        description="Test claim description for routing",
        risk_flags=risk_flags or [],
    )


def test_routine_claim_continues():
    assert risk_route(make_facts(), REF) is None


def test_high_amount_goes_to_human():
    r = risk_route(make_facts("C002", amount=480000), REF)
    assert r is not None
    assert "HIGH_AMOUNT" in r.risk_flags


def test_expired_policy_goes_to_human():
    r = risk_route(make_facts("C003", amount=6000), REF)
    assert "POLICY_NOT_ACTIVE" in r.risk_flags


def test_cancelled_policy_goes_to_human():
    r = risk_route(make_facts("C010", amount=5000, claim_type="travel"), REF)
    assert "POLICY_NOT_ACTIVE" in r.risk_flags


def test_incident_before_policy_start():
    r = risk_route(make_facts("C006", amount=3000, incident=date(2026, 8, 20)), REF)
    assert "INCIDENT_BEFORE_POLICY_START" in r.risk_flags


def test_incident_after_policy_end():
    r = risk_route(make_facts("C003", amount=6000, incident=date(2026, 10, 1)), REF)
    assert "INCIDENT_AFTER_POLICY_END" in r.risk_flags


def test_suspicious_history_goes_to_human():
    r = risk_route(make_facts("C005", amount=9000), REF)
    assert "SUSPICIOUS_HISTORY" in r.risk_flags


def test_unknown_customer_goes_to_human():
    r = risk_route(make_facts("C999"), REF)
    assert "POLICY_NOT_VERIFIED" in r.risk_flags


def test_intake_flags_go_to_human():
    r = risk_route(make_facts(risk_flags=["possible_manipulation"]), REF)
    assert "INTAKE_FLAGGED" in r.risk_flags


def test_several_reasons_are_all_reported():
    r = risk_route(make_facts("C003", amount=480000), REF)
    assert "HIGH_AMOUNT" in r.risk_flags and "POLICY_NOT_ACTIVE" in r.risk_flags
