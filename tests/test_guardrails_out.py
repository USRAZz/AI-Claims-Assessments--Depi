import pytest
from pydantic import ValidationError

from claims_agent.models import ClaimSummary
from pipeline.guardrails_out import check_output


def make_summary(**overrides):
    values = dict(
        claim_id="CL-TEST",
        summary="The reported incident requires adjuster review.",
        coverage_evidence="Policy information is available for review.",
        estimated_deductible=1000,
    )
    values.update(overrides)
    return ClaimSummary(**values)


def test_safe_summary_has_no_output_problems():
    assert check_output(make_summary()) == []


@pytest.mark.parametrize("field", ["summary", "coverage_evidence"])
def test_model_rejects_forbidden_promises_in_main_text(field):
    with pytest.raises(ValidationError, match="Must not promise"):
        make_summary(**{field: "The claim will be paid after review."})


@pytest.mark.parametrize("field", ["gaps", "risk_flags", "missing_documents"])
def test_forbidden_phrases_in_list_fields_are_detected(field):
    result = check_output(make_summary(**{field: ["The claim is approved"]}))
    assert any("Forbidden phrase" in problem for problem in result)


@pytest.mark.parametrize("field", ["summary", "coverage_evidence", "gaps", "risk_flags", "missing_documents"])
def test_additional_promise_patterns_are_detected_in_every_field(field):
    text = "We guarantee a favorable result"  # not the exact forbidden word 'guaranteed'
    value = [text] if field in {"gaps", "risk_flags", "missing_documents"} else text
    result = check_output(make_summary(**{field: value}))
    assert any("Promise pattern" in problem for problem in result)


def test_case_insensitive_matching():
    problems = check_output(make_summary(gaps=["WE WILL REFUND the fee"]))
    assert any("Forbidden phrase" in problem for problem in problems)


@pytest.mark.parametrize("status", ["PENDING-ADJUSTER", "AI-ASSESSED"])
def test_allowed_statuses(status):
    assert check_output(make_summary(status=status)) == []


def test_invalid_status_is_detected_defensively():
    # Bypass Pydantic's Literal validation to exercise the output guardrail itself.
    summary = make_summary().model_copy(update={"status": "APPROVED"})
    assert "Invalid status" in check_output(summary)


def test_multiple_problems_are_returned():
    summary = make_summary().model_copy(update={
        "status": "REJECTED", "gaps": ["will be paid", "guarantee payment"]
    })
    problems = check_output(summary)
    assert "Invalid status" in problems
    assert len(problems) >= 3
