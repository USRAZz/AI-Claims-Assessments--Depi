from claims_agent.models import ClaimSummary, NeedsManual, HumanReviewNeeded
from claims_agent.context import ClaimContext
from pipeline.guardrails_out import check_output
from tools.case_tools import CaseSaveError, create_claim_case


def send_to_human(result: ClaimSummary | NeedsManual | HumanReviewNeeded, ctx: ClaimContext) -> dict:
    if isinstance(result, ClaimSummary):
        problems = check_output(result)
        if problems:
            ctx.log(f"Output guardrail blocked summary: {problems}")
            result = NeedsManual(claim_id=result.claim_id,
                                 reason="Output guardrail blocked the summary",
                                 last_feedback="; ".join(problems))

    try:
        case_id = create_claim_case(result)
    except CaseSaveError as exc:
        # Never claim the case was saved unless the tool confirmed it.
        ctx.log(f"Case NOT saved: {exc}")
        return {"case_id": None, "result": result, "audit_log": ctx.audit_log, "error": str(exc)}

    ctx.log(f"Case {case_id} saved for human adjuster")
    return {"case_id": case_id, "result": result, "audit_log": ctx.audit_log, "error": None}
