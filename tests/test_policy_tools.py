import pytest

from tools.common import ToolDataError
from tools.policy_tools import calculate_deductible, get_customer_policy, get_policy_coverage


def test_policy_found():
    p = get_customer_policy("C001")
    assert p["policy_number"] == "P-1001"
    assert p["policy_status"] == "active"


def test_policy_expired():
    assert get_customer_policy("C003")["policy_status"] == "expired"


def test_unknown_customer_raises():
    with pytest.raises(ToolDataError):
        get_customer_policy("C999")


def test_empty_customer_id_raises():
    with pytest.raises(ToolDataError):
        get_customer_policy("  ")


def test_coverage_auto_covered():
    c = get_policy_coverage("C001", "auto")
    assert c["is_covered"] is True
    assert c["coverage_limit"] == 50000


def test_claim_type_not_in_policy():
    c = get_policy_coverage("C001", "health")
    assert c["is_covered"] is False
    assert c["coverage_limit"] == 0


def test_exclusion_matched():
    c = get_policy_coverage("C007", "auto", "Crashed during a track racing event")
    assert c["exclusions_matched"] == ["racing"]


def test_no_exclusion_when_not_mentioned():
    c = get_policy_coverage("C001", "auto", "Rear-ended at a traffic light")
    assert c["exclusions_matched"] == []


def test_invalid_claim_type_raises():
    with pytest.raises(ToolDataError):
        get_policy_coverage("C001", "pets")


def test_deductible_normal():
    r = calculate_deductible(8000, 1000, 50000)
    assert r["estimated_insurer_share"] == 7000
    assert r["estimated_customer_share"] == 1000


def test_deductible_over_limit():
    r = calculate_deductible(100000, 1000, 50000)
    assert r["estimated_insurer_share"] == 50000
    assert r["estimated_customer_share"] == 50000


def test_amount_below_deductible():
    r = calculate_deductible(500, 1000, 50000)
    assert r["estimated_insurer_share"] == 0
    assert r["estimated_customer_share"] == 500


@pytest.mark.parametrize("amount", [0, -10, "abc"])
def test_bad_amount_raises(amount):
    with pytest.raises(ToolDataError):
        calculate_deductible(amount, 1000, 50000)