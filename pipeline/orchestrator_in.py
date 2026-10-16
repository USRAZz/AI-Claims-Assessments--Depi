"""
orchestrator_in.py: the ORCHESTRATOR (LLM plans, Python runs) and the main entry
point run_intake_and_evidence().

Flow (who decides what):
  1. check_input()      Python  -> block unsafe text
  2. extract_claim_facts() LLM  -> ClaimFacts (intake agent)
  3. risk_route()       Python  -> high-risk claims stop here (human review)
  4. plan_workers()     LLM     -> orchestrator agent chooses the workers
     sanitize_plan()    Python  -> makes sure the plan is allowed
  5. asyncio.gather()   Python  -> chosen workers run at the same time
  6. merge_evidence()   Python  -> EvidenceBundle (+ gaps)

Returns EvidenceBundle (continue to the writer) or HumanReviewNeeded (stop).
"""

import asyncio
import uuid
from typing import Literal, Optional

from agents import Agent, Runner
from pydantic import BaseModel

from claims_agent.context import ClaimContext
from claims_agent.models import ClaimFacts, EvidenceBundle, HumanReviewNeeded
from pipeline.guardrails_in import check_input
from pipeline.intake import extract_claim_facts
from pipeline.llm_config import MODEL, MODEL_SETTINGS, output_schema
from pipeline.merge import merge_evidence
from pipeline.router import risk_route
from pipeline.workers import WORKERS

ALL_WORKERS = ["policy", "history", "docs"]


class WorkerPlan(BaseModel):
    workers: list[Literal["policy", "history", "docs"]]
    reasoning: str


orchestrator_agent = Agent(
    name="Claim orchestrator",
    instructions=(
        "You plan the evidence gathering for one insurance claim. Available workers:\n"
        "- policy: policy status, coverage, limits, exclusions, deductible estimate\n"
        "- history: the customer's previous claims and suspicious patterns\n"
        "- docs: which required documents were submitted and which are missing\n"
        "Return the workers that are needed and a one-sentence reasoning. "
        "If you are unsure whether a worker is needed, include it."
    ),
    model=MODEL,
    model_settings=MODEL_SETTINGS,
    output_type=output_schema(WorkerPlan),
)


def sanitize_plan(planned: list) -> list[str]:
    """Python decides what is allowed: only known workers, no duplicates, the policy
    worker always runs, and an empty or invalid plan falls back to all workers."""
    valid = []
    for name in planned or []:
        if name in ALL_WORKERS and name not in valid:
            valid.append(name)
    if not valid:
        return list(ALL_WORKERS)
    if "policy" not in valid:
        valid.insert(0, "policy")
    return valid


async def plan_workers(facts: ClaimFacts, ctx: ClaimContext) -> list[str]:
    prompt = (
        f"Claim facts: {facts.model_dump_json()}\n"
        f"Submitted documents: {ctx.attached_documents}"
    )
    planned, reasoning = [], "fallback plan (orchestrator unavailable)"
    try:
        run = await Runner.run(orchestrator_agent, prompt, context=ctx)
        if isinstance(run.final_output, WorkerPlan):
            planned, reasoning = run.final_output.workers, run.final_output.reasoning
    except Exception as exc:
        ctx.log(f"orchestrator failed: {type(exc).__name__}")
    plan = sanitize_plan(planned)
    ctx.log(f"orchestrator plan: {plan} ({reasoning})")
    return plan


async def run_intake_and_evidence_async(
    raw_text: str,
    customer_id: str,
    attached_documents: Optional[list[str]] = None,
    ctx: Optional[ClaimContext] = None,
) -> EvidenceBundle | HumanReviewNeeded:
    if ctx is None:
        ctx = ClaimContext(claim_id=f"CL-{uuid.uuid4().hex[:8].upper()}")
    ctx.customer_id = customer_id
    ctx.raw_claim_text = raw_text
    ctx.attached_documents = list(attached_documents or [])

    # 1) input guardrail (Python)
    reason = check_input(raw_text)
    if reason:
        ctx.log(f"input guardrail blocked: {reason}")
        return HumanReviewNeeded(
            claim_id=ctx.claim_id, reason=f"Input blocked: {reason}", risk_flags=["INPUT_BLOCKED"]
        )

    # 2) intake agent (LLM)
    facts = await extract_claim_facts(ctx)
    if isinstance(facts, HumanReviewNeeded):
        ctx.log(f"stopped after intake: {facts.risk_flags}")
        return facts

    # 3) routing (Python)
    decision = risk_route(facts)
    if decision is not None:
        ctx.log(f"router: human review needed {decision.risk_flags}")
        return decision
    ctx.log("router: routine claim, continuing")

    # 4) orchestrator plans (LLM), Python checks the plan
    plan = await plan_workers(facts, ctx)

    # 5) workers run in parallel (Python)
    outcomes = await asyncio.gather(
        *(WORKERS[name](facts, ctx) for name in plan), return_exceptions=True
    )

    # 6) merge (Python)
    bundle = merge_evidence(facts, dict(zip(plan, outcomes)))
    ctx.log(f"merge done, gaps: {bundle.gaps}")
    return bundle


def run_intake_and_evidence(
    raw_text: str,
    customer_id: str,
    attached_documents: Optional[list[str]] = None,
    ctx: Optional[ClaimContext] = None,
) -> EvidenceBundle | HumanReviewNeeded:
    """Normal (non-async) version. Use this one from Gradio unless your handler is async."""
    return asyncio.run(
        run_intake_and_evidence_async(raw_text, customer_id, attached_documents, ctx)
    )