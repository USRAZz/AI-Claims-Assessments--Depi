from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from claims_agent.context import ClaimContext
from claims_agent.models import (
    ClaimFacts, ClaimSummary, DocsEvidence, EvaluationResult,
    EvidenceBundle, NeedsManual, PolicyEvidence,
)

# The revision module requires the OpenAI Agents SDK and pipeline.evaluator.
# The evaluator module is not included in this project snapshot, so isolate
# that import for unit tests without modifying application code.
@pytest.fixture
def revision(monkeypatch):
    import importlib
    import sys
    import types

    pytest.importorskip("agents")
    if "pipeline.evaluator" not in sys.modules:
        fake = types.ModuleType("pipeline.evaluator")
        fake.claim_evaluator = object()
        monkeypatch.setitem(sys.modules, "pipeline.evaluator", fake)
    module = importlib.import_module("pipeline.revision_loop")
    return module


@pytest.fixture
def bundle():
    return EvidenceBundle(
        facts=ClaimFacts(
            claim_id="CL-REAL", customer_id="C-1", claim_type="auto",
            incident_date=date(2026, 10, 1), claimed_amount=8000,
            description="Vehicle collision at a junction",
        ),
        policy=PolicyEvidence(
            policy_number="P-1", policy_status="active",
            policy_start_date=date(2025, 1, 1), is_covered=True,
            coverage_limit=50000, deductible=1200,
        ),
        docs=DocsEvidence(
            required_documents=["photos", "report"],
            received_documents=["photos"], missing_documents=["report"],
            is_complete=False,
        ),
    )


def make_draft():
    return ClaimSummary(
        claim_id="CL-HALLUCINATED", summary="The incident needs an adjuster review.",
        coverage_evidence="Policy details are available.",
        estimated_deductible=9999, missing_documents=["invented.pdf"],
    )


def test_ground_draft_uses_evidence_not_writer_values(revision, bundle):
    original = make_draft()
    grounded = revision.ground_draft(original, bundle)
    assert grounded.claim_id == "CL-REAL"
    assert grounded.estimated_deductible == 1200
    assert grounded.missing_documents == ["report"]
    assert original.claim_id == "CL-HALLUCINATED"


def test_ground_draft_handles_missing_policy_and_docs(revision, bundle):
    bundle = bundle.model_copy(update={"policy": None, "docs": None})
    grounded = revision.ground_draft(make_draft(), bundle)
    assert grounded.estimated_deductible == 0.0
    assert grounded.missing_documents == []


def mock_runner(monkeypatch, revision, outputs):
    responses = iter(outputs)

    async def run(agent, prompt, context):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(final_output=value)

    runner = AsyncMock(side_effect=run)
    monkeypatch.setattr(revision.Runner, "run", runner)
    return runner


@pytest.mark.asyncio
async def test_accepted_first_draft_returns_grounded_summary(revision, bundle, monkeypatch):
    runner = mock_runner(monkeypatch, revision, [
        make_draft(), EvaluationResult(is_acceptable=True),
    ])
    ctx = ClaimContext(claim_id="CL-REAL")
    result = await revision.run_revision_loop(bundle, ctx)
    assert isinstance(result, ClaimSummary)
    assert result.claim_id == "CL-REAL"
    assert result.estimated_deductible == 1200
    assert result.missing_documents == ["report"]
    assert ctx.revision_count == 1
    assert "draft accepted" in ctx.audit_log[-1]
    assert runner.await_count == 2


@pytest.mark.asyncio
async def test_rejected_draft_is_revised_with_feedback(revision, bundle, monkeypatch):
    runner = mock_runner(monkeypatch, revision, [
        make_draft(), EvaluationResult(is_acceptable=False, feedback="Add evidence"),
        make_draft(), EvaluationResult(is_acceptable=True),
    ])
    ctx = ClaimContext(claim_id="CL-REAL")
    result = await revision.run_revision_loop(bundle, ctx)
    assert isinstance(result, ClaimSummary)
    assert ctx.revision_count == 2
    assert "Add evidence" in runner.await_args_list[2].args[1]
    assert any("rejected" in entry for entry in ctx.audit_log)


@pytest.mark.asyncio
async def test_all_rejections_escalate_to_manual(revision, bundle, monkeypatch):
    outputs = []
    for _ in range(revision.MAX_REVISIONS):
        outputs.extend([make_draft(), EvaluationResult(is_acceptable=False, feedback="Unsupported detail")])
    runner = mock_runner(monkeypatch, revision, outputs)
    ctx = ClaimContext(claim_id="CL-REAL")
    result = await revision.run_revision_loop(bundle, ctx)
    assert isinstance(result, NeedsManual)
    assert result.claim_id == "CL-REAL"
    assert result.last_feedback == "Unsupported detail"
    assert ctx.revision_count == revision.MAX_REVISIONS
    assert runner.await_count == 2 * revision.MAX_REVISIONS


@pytest.mark.asyncio
async def test_writer_errors_do_not_crash_loop(revision, bundle, monkeypatch):
    runner = mock_runner(monkeypatch, revision, [RuntimeError("offline")] * revision.MAX_REVISIONS)
    ctx = ClaimContext(claim_id="CL-REAL")
    result = await revision.run_revision_loop(bundle, ctx)
    assert isinstance(result, NeedsManual)
    assert result.last_feedback == "Model error: RuntimeError"
    assert runner.await_count == revision.MAX_REVISIONS


@pytest.mark.asyncio
async def test_wrong_writer_output_is_retried(revision, bundle, monkeypatch):
    mock_runner(monkeypatch, revision, [
        "not a summary", make_draft(), EvaluationResult(is_acceptable=True),
    ])
    ctx = ClaimContext(claim_id="CL-REAL")
    result = await revision.run_revision_loop(bundle, ctx)
    assert isinstance(result, ClaimSummary)
    assert ctx.revision_count == 2
    assert any("writer output invalid" in entry for entry in ctx.audit_log)


@pytest.mark.asyncio
async def test_evaluator_errors_are_retried(revision, bundle, monkeypatch):
    mock_runner(monkeypatch, revision, [
        make_draft(), RuntimeError("offline"),
        make_draft(), EvaluationResult(is_acceptable=True),
    ])
    ctx = ClaimContext(claim_id="CL-REAL")
    result = await revision.run_revision_loop(bundle, ctx)
    assert isinstance(result, ClaimSummary)
    assert ctx.revision_count == 2
    assert any("evaluator failed" in entry for entry in ctx.audit_log)


@pytest.mark.asyncio
async def test_wrong_evaluator_output_is_retried(revision, bundle, monkeypatch):
    mock_runner(monkeypatch, revision, [
        make_draft(), "invalid evaluation", make_draft(), EvaluationResult(is_acceptable=True),
    ])
    ctx = ClaimContext(claim_id="CL-REAL")
    result = await revision.run_revision_loop(bundle, ctx)
    assert isinstance(result, ClaimSummary)
    assert any("evaluator output invalid" in entry for entry in ctx.audit_log)
