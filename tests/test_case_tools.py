import sqlite3

import pytest

from claims_agent.models import ClaimSummary, HumanReviewNeeded, NeedsManual
from tools import case_tools


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db = tmp_path / "cases.db"
    monkeypatch.setattr(case_tools, "DB_PATH", db)
    return db


def make_summary(**overrides):
    data = dict(
        claim_id="CL-TEST", summary="The incident is pending adjuster review.",
        coverage_evidence="Policy data was collected.", estimated_deductible=500,
    )
    data.update(overrides)
    return ClaimSummary(**data)


def saved_rows(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT id, claim_id, status, payload FROM cases ORDER BY id").fetchall()


def test_saves_claim_summary_and_returns_confirmed_id(isolated_db):
    summary = make_summary(status="AI-ASSESSED")
    case_id = case_tools.create_claim_case(summary)
    rows = saved_rows(isolated_db)
    assert case_id > 0
    assert rows[0][:3] == (case_id, "CL-TEST", "AI-ASSESSED")
    assert ClaimSummary.model_validate_json(rows[0][3]) == summary


@pytest.mark.parametrize("result", [
    NeedsManual(claim_id="CL-MANUAL", reason="Too many revisions"),
    HumanReviewNeeded(claim_id="CL-HUMAN", reason="Risk detected"),
])
def test_escalations_are_saved_as_pending(isolated_db, result):
    case_tools.create_claim_case(result)
    assert saved_rows(isolated_db)[0][2] == "PENDING-ADJUSTER"


def test_disallowed_status_is_never_persisted(isolated_db):
    summary = make_summary().model_copy(update={"status": "APPROVED"})
    case_tools.create_claim_case(summary)
    assert saved_rows(isolated_db)[0][2] == "PENDING-ADJUSTER"


@pytest.mark.parametrize("result", [None, "claim", {"claim_id": "CL-1"}, 123])
def test_unsupported_result_is_rejected_before_database_access(isolated_db, result):
    with pytest.raises(case_tools.CaseSaveError, match="Unsupported result type"):
        case_tools.create_claim_case(result)
    assert not isolated_db.exists()


def test_blank_claim_id_is_rejected_before_database_access(isolated_db):
    with pytest.raises(case_tools.CaseSaveError, match="claim_id is required"):
        case_tools.create_claim_case(NeedsManual(claim_id="  ", reason="Review"))
    assert not isolated_db.exists()


def test_each_successful_save_creates_a_distinct_row(isolated_db):
    first = case_tools.create_claim_case(make_summary())
    second = case_tools.create_claim_case(make_summary())
    assert first != second
    assert len(saved_rows(isolated_db)) == 2


def test_database_error_retries_once_and_then_raises(isolated_db, monkeypatch):
    attempts = []

    def fail_init():
        attempts.append(1)
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(case_tools, "_init", fail_init)
    with pytest.raises(case_tools.CaseSaveError, match="Could not save the case"):
        case_tools.create_claim_case(make_summary())
    assert len(attempts) == 2


def test_transient_database_error_recovers_on_second_attempt(isolated_db, monkeypatch):
    original_init = case_tools._init
    attempts = []

    def flaky_init():
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("temporary failure")
        original_init()

    monkeypatch.setattr(case_tools, "_init", flaky_init)
    assert case_tools.create_claim_case(make_summary()) > 0
    assert len(attempts) == 2
    assert len(saved_rows(isolated_db)) == 1
