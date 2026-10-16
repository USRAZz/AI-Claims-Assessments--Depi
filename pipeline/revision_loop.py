from agents import Runner
from agents.exceptions import ModelBehaviorError

from claims_agent.context import ClaimContext
from claims_agent.models import ClaimSummary, EvaluationResult, EvidenceBundle, NeedsManual
from pipeline.evaluator import claim_evaluator
from pipeline.writer import claim_writer

MAX_REVISIONS = 3


def ground_draft(draft: ClaimSummary, bundle: EvidenceBundle) -> ClaimSummary:
    """Python decides the hard facts; the LLM only writes the text.
    claim_id, estimated_deductible and missing_documents always come from the evidence."""
    missing = list(bundle.docs.missing_documents) if bundle.docs else []
    deductible = bundle.policy.deductible if bundle.policy else 0.0
    return draft.model_copy(update={
        "claim_id": bundle.facts.claim_id,
        "estimated_deductible": deductible,
        "missing_documents": missing,
    })


async def run_revision_loop(bundle: EvidenceBundle, ctx: ClaimContext) -> ClaimSummary | NeedsManual:
    evidence = bundle.model_dump_json(indent=2)
    prompt = f"Evidence:\n{evidence}"
    last_feedback = ""

    for attempt in range(1, MAX_REVISIONS + 1):
        ctx.revision_count = attempt
        try:
            draft = (await Runner.run(claim_writer, prompt, context=ctx)).final_output
            if not isinstance(draft, ClaimSummary):
                raise ModelBehaviorError("writer did not return a ClaimSummary")
        except ModelBehaviorError as e:      # model validators rejected the draft
            last_feedback = f"Your output was invalid: {e}"
            ctx.log(f"Attempt {attempt}: writer output invalid")
            prompt = f"Evidence:\n{evidence}\n\nFix this: {last_feedback}"
            continue
        except Exception as e:               # rate limit, network, ... (never crash the app)
            last_feedback = f"Model error: {type(e).__name__}"
            ctx.log(f"Attempt {attempt}: writer failed ({type(e).__name__})")
            continue

        draft = ground_draft(draft, bundle)

        try:                                  # protects the evaluator call
            review = (await Runner.run(
                claim_evaluator,
                f"Evidence:\n{evidence}\n\nDraft:\n{draft.model_dump_json(indent=2)}",
                context=ctx,
            )).final_output
            if not isinstance(review, EvaluationResult):
                raise ModelBehaviorError("evaluator did not return an EvaluationResult")
        except ModelBehaviorError as e:
            last_feedback = f"Evaluator output was invalid: {e}"
            ctx.log(f"Attempt {attempt}: evaluator output invalid")
            continue
        except Exception as e:
            last_feedback = f"Model error: {type(e).__name__}"
            ctx.log(f"Attempt {attempt}: evaluator failed ({type(e).__name__})")
            continue

        if review.is_acceptable:
            ctx.log(f"Attempt {attempt}: draft accepted")
            return draft

        last_feedback = review.feedback
        ctx.log(f"Attempt {attempt}: rejected - {last_feedback}")
        prompt = (f"Evidence:\n{evidence}\n\nYour previous draft:\n{draft.model_dump_json(indent=2)}"
                  f"\n\nRevise it. Fix this: {last_feedback}")

    return NeedsManual(
        claim_id=bundle.facts.claim_id,
        reason=f"Draft still rejected after {MAX_REVISIONS} attempts",
        last_feedback=last_feedback,
    )
