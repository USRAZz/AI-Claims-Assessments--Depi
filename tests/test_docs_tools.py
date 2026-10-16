import pytest

from tools.common import ToolDataError
from tools.docs_tools import check_required_documents


def test_all_documents_present():
    r = check_required_documents("auto", ["police_report", "photos", "repair_estimate"])
    assert r["is_complete"] is True
    assert r["missing_documents"] == []


def test_missing_documents():
    r = check_required_documents("property", ["photos"])
    assert r["is_complete"] is False
    assert r["received_documents"] == ["photos"]
    assert r["missing_documents"] == ["repair_estimate", "ownership_proof"]


def test_names_are_normalized():
    r = check_required_documents("auto", ["Police Report", " PHOTOS ", "repair-estimate"])
    assert r["is_complete"] is True


def test_empty_list_means_everything_missing():
    r = check_required_documents("health", [])
    assert r["missing_documents"] == ["medical_report", "invoice", "prescription"]


def test_extra_documents_are_ignored():
    r = check_required_documents("auto", ["police_report", "photos", "repair_estimate", "selfie"])
    assert r["is_complete"] is True


def test_invalid_claim_type_raises():
    with pytest.raises(ToolDataError):
        check_required_documents("pets", [])


def test_submitted_documents_must_be_a_list():
    with pytest.raises(ToolDataError):
        check_required_documents("auto", "police_report")
