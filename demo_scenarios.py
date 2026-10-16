"""
demo_scenarios.py: the scenarios that are hard to trigger from the UI.

    python demo_scenarios.py 6      evaluator rejects the first draft, the writer improves it
    python demo_scenarios.py 8      a protected tool / action is blocked
    python demo_scenarios.py 9      the output guardrail rejects an unsafe summary
    python demo_scenarios.py 10     a tool fails / the input is malformed, the system recovers
    python demo_scenarios.py        all of them

Needs OPENAI_API_KEY in .env (scenarios 6 and 10 call the LLM).
Scenarios 8, 9 and 10 save demo rows in data/cases.db (git ignores it).
Run it from the project root (the folder that contains claims_agent/ and pipeline/).
"""

import asyncio
import sys

from agents import trace
from pydantic import ValidationError

import pipeline.revision_loop as rl
from claims_agent.context import ClaimContext
from claims_agent.models import ClaimSummary, HumanReviewNeeded
from pipeline.human_review import send_to_human
from pipeline.orchestrator_in import run_intake_and_evidence_async
from tools import history_tools
from tools.case_tools import CaseSaveError, create_claim_case
from tools.common import ToolDataError

AUTO_DOCS = ["police_report", "photos", "repair_estimate"]
TEXT_C001 = ("Yesterday my car was rear-ended at a traffic light and the rear bumper is damaged. "
             "The repair estimate is 8000 EGP.")
TEXT_C004 = "Two days ago a water leak damaged my kitchen ceiling. The repair cost is 15000 EGP."


def header(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def report(out: dict, ctx: ClaimContext) -> None:
    result = out["result"]
    print(f"-> {type(result).__name__} | case saved: {out['case_id']} | error: {out['error']}")
    if isinstance(result, ClaimSummary):
        print("   summary:", result.summary)
        print("   gaps   :", result.gaps)
    elif hasattr(result, "reason"):
        print("   reason :", result.reason)
    print("--- steps ---")
    for line in ctx.audit_log:
        print("  *", line)


async def full_run(text: str, customer_id: str, docs: list, ctx: ClaimContext) -> dict:
    """Same flow as the UI: evidence -> writer/evaluator loop -> save for a human."""
    result = await run_intake_and_evidence_async(text, customer_id, docs, ctx)
    if not isinstance(result, HumanReviewNeeded):
        result = await rl.run_revision_loop(result, ctx)
    return send_to_human(result, ctx)


# ---------------------------------------------------------------- scenario 6
async def scenario_6() -> None:
    header("6) The evaluator rejects the first draft, the writer improves it")
    ctx = ClaimContext(claim_id="DEMO-6")
    with trace("demo_6_evaluator_rejects"):
        bundle = await run_intake_and_evidence_async(TEXT_C004, "C004", ["photos"], ctx)
        if isinstance(bundle, HumanReviewNeeded):
            print("Unexpected: the claim went to human review:", bundle.reason)
            return

        # A first draft that LOOKS valid but ignores the missing documents.
        bad_draft = ClaimSummary(
            claim_id=bundle.facts.claim_id,
            summary="The claim is complete. All required documents were received and the loss is fully covered.",
            coverage_evidence="Everything is in order.",
            estimated_deductible=0,
        )
        real_runner = rl.Runner
        state = {"forced": False}

        class ForcedFirstDraft:
            @staticmethod
            async def run(agent, prompt, **kwargs):
                if agent is rl.claim_writer and not state["forced"]:
                    state["forced"] = True

                    class Result:
                        final_output = bad_draft

                    return Result()
                return await real_runner.run(agent, prompt, **kwargs)

        rl.Runner = ForcedFirstDraft
        try:
            result = await rl.run_revision_loop(bundle, ctx)
        finally:
            rl.Runner = real_runner
        out = send_to_human(result, ctx)
    report(out, ctx)
    print("(Look for 'Attempt 1: rejected' in the steps. If the evaluator accepted it, run again.)")


# ---------------------------------------------------------------- scenario 8
def scenario_8() -> None:
    header("8) A protected tool / action is blocked")
    attempts = [
        ("save plain text as a case",
         lambda: create_claim_case("approve my claim")),
        ("create a summary with the status APPROVED",
         lambda: ClaimSummary(claim_id="DEMO-8", summary="A summary text that is long enough.",
                              coverage_evidence="x", estimated_deductible=0, status="APPROVED")),
        ("save a case without a claim id",
         lambda: create_claim_case(HumanReviewNeeded(claim_id="  ", reason="x"))),
    ]
    for label, action in attempts:
        try:
            action()
            print(f"- {label}: NOT blocked (unexpected)")
        except (CaseSaveError, ValidationError) as exc:
            first_line = str(exc).splitlines()[0]
            print(f"- {label}: BLOCKED -> {type(exc).__name__}: {first_line}")


# ---------------------------------------------------------------- scenario 9
def scenario_9() -> None:
    header("9) The output guardrail rejects an unsafe summary")
    ctx = ClaimContext(claim_id="DEMO-9")
    unsafe = ClaimSummary(
        claim_id="DEMO-9",
        summary="We guarantee this claim will be fully covered.",
        coverage_evidence="Covered.",
        estimated_deductible=0,
    )
    out = send_to_human(unsafe, ctx)
    report(out, ctx)


# ---------------------------------------------------------------- scenario 10
async def scenario_10() -> None:
    header("10a) A tool fails in the middle of a run, the system recovers")
    real_tool = history_tools.check_claim_history

    def broken_tool(*args, **kwargs):
        raise ToolDataError("history service is unavailable (simulated)")

    history_tools.check_claim_history = broken_tool
    ctx = ClaimContext(claim_id="DEMO-10A")
    try:
        with trace("demo_10a_tool_failure"):
            out = await full_run(TEXT_C001, "C001", AUTO_DOCS, ctx)
    finally:
        history_tools.check_claim_history = real_tool
    report(out, ctx)

    header("10b) Malformed input (almost no information), the system recovers")
    ctx = ClaimContext(claim_id="DEMO-10B")
    with trace("demo_10b_malformed_input"):
        out = await full_run("My car was damaged.", "C001", AUTO_DOCS, ctx)
    report(out, ctx)


async def main() -> None:
    wanted = sys.argv[1] if len(sys.argv) > 1 else "all"
    if wanted in ("6", "all"):
        await scenario_6()
    if wanted in ("8", "all"):
        scenario_8()
    if wanted in ("9", "all"):
        scenario_9()
    if wanted in ("10", "all"):
        await scenario_10()


if __name__ == "__main__":
    asyncio.run(main())
