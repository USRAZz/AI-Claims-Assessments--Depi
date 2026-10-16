import pytest

from tools.common import ToolDataError
from tools.history_tools import check_claim_history

REF = "2026-10-13"  # fixed date so results never change


def test_no_history():
    r = check_claim_history("C002", REF)
    assert r["previous_claims_count"] == 0
    assert r["suspicious_pattern"] is False


def test_old_claim_not_in_last_90_days():
    r = check_claim_history("C001", REF)
    assert r["previous_claims_count"] == 1
    assert r["claims_last_90_days"] == 0
    assert r["suspicious_pattern"] is False


def test_suspicious_pattern():
    r = check_claim_history("C005", REF)
    assert r["previous_claims_count"] == 3
    assert r["claims_last_90_days"] == 3
    assert r["suspicious_pattern"] is True


def test_unknown_customer_raises():
    with pytest.raises(ToolDataError):
        check_claim_history("C999", REF)


def test_bad_date_raises():
    with pytest.raises(ToolDataError):
        check_claim_history("C001", "13/10/2026")