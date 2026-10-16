"""
workers.py: the three WORKER agents (LLM + tools): policy, history, docs.

Each worker is an agent that calls its tool(s) and returns structured evidence.
Safety built in (the LLM decides what to call, Python decides what is true):
  - tool arguments are checked (the customer_id must match the claim)
  - every tool result is recorded; a tool error makes the worker FAIL (-> a "gap")
  - the LLM's output is compared with the tool results and corrected if different
"""

import json
from datetime import date

from agents import Agent, RunContextWrapper, Runner, function_tool

from claims_agent.context import ClaimContext
from claims_agent.models import ClaimFacts, DocsEvidence, HistoryEvidence, PolicyEvidence
from pipeline.llm_config import MODEL, MODEL_SETTINGS, output_schema
from tools import docs_tools, history_tools, policy_tools
from tools.common import ToolDataError


class WorkerFailed(Exception):
    """A worker could not produce trustworthy evidence."""


# ---------------------------------------------------------------- helpers
def _call(ctx, results, key, fn, *args, customer_id=None):
    """Run one plain tool function, record the result, never raise to the LLM."""
    ctx.context.log(f"tool {fn.__name__}{args}")
    try:
        if customer_id is not None and customer_id != ctx.context.customer_id:
            raise ToolDataError("customer_id does not match the claim being processed")
        data = fn(*args)
    except ToolDataError as exc:
        results.setdefault("errors", []).append(str(exc))
        return json.dumps({"error": str(exc)})
    results[key] = data
    return json.dumps(data)


def _same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return sorted(a) == sorted(b)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 0.01
        except (TypeError, ValueError):
            return False
    return a == b


def _ground(evidence, truth: dict, ctx: ClaimContext, who: str):
    """Replace any field that differs from the tool results (tool results win)."""
    wrong = [k for k, v in truth.items() if not _same(getattr(evidence, k, None), v)]
    if wrong:
        ctx.log(f"{who}: output corrected from tool results: {', '.join(wrong)}")
    return evidence.model_copy(update=truth)


async def _run(agent, prompt, ctx, results, truth_fn, evidence_type):
    try:
        run = await Runner.run(agent, prompt, context=ctx)
    except Exception as exc:
        raise WorkerFailed(f"{agent.name} failed ({type(exc).__name__})") from exc
    if results.get("errors"):
        raise WorkerFailed(f"{agent.name} tool error: {results['errors'][0]}")
    if not isinstance(run.final_output, evidence_type):
        raise WorkerFailed(f"{agent.name} returned invalid output")
    evidence = _ground(run.final_output, truth_fn(results), ctx, agent.name)
    ctx.log(f"{agent.name} ok")
    return evidence


def _need(results, *keys):
    missing = [k for k in keys if not results.get(k)]
    if missing:
        raise WorkerFailed(f"tools did not return data: {', '.join(missing)}")


# ---------------------------------------------------------------- policy worker
def _policy_truth(r: dict) -> dict:
    _need(r, "policy", "coverage", "deductible")
    p, c, d = r["policy"], r["coverage"], r["deductible"]
    return {
        "policy_number": p["policy_number"],
        "policy_status": p["policy_status"],
        "policy_start_date": date.fromisoformat(p["policy_start_date"]),
        "is_covered": c["is_covered"],
        "coverage_limit": c["coverage_limit"],
        "deductible": c["deductible"],
        "exclusions_matched": c["exclusions_matched"],
        "estimated_insurer_share": d["estimated_insurer_share"],
        "estimated_customer_share": d["estimated_customer_share"],
    }


async def run_policy_worker(facts: ClaimFacts, ctx: ClaimContext) -> PolicyEvidence:
    results: dict = {}

    @function_tool
    def get_customer_policy(ctx: RunContextWrapper[ClaimContext], customer_id: str) -> str:
        """Get the customer's policy number, status and start/end dates."""
        return _call(ctx, results, "policy", policy_tools.get_customer_policy,
                     customer_id, customer_id=customer_id)

    @function_tool
    def get_policy_coverage(ctx: RunContextWrapper[ClaimContext], customer_id: str,
                            claim_type: str, description: str) -> str:
        """Get coverage, limit, deductible and matched exclusions for a claim type."""
        return _call(ctx, results, "coverage", policy_tools.get_policy_coverage,
                     customer_id, claim_type, description, customer_id=customer_id)

    @function_tool
    def calculate_deductible(ctx: RunContextWrapper[ClaimContext], claimed_amount: float,
                             deductible: float, coverage_limit: float) -> str:
        """Estimate the insurer and customer shares. This is an estimate, not a promise."""
        return _call(ctx, results, "deductible", policy_tools.calculate_deductible,
                     claimed_amount, deductible, coverage_limit)

    agent = Agent(
        name="Policy worker",
        instructions=(
            "You gather policy evidence for one insurance claim.\n"
            "1) Call get_customer_policy. 2) Call get_policy_coverage with the claim type "
            "and the claim description. 3) Call calculate_deductible using the claimed "
            "amount, and the deductible and coverage limit returned by get_policy_coverage.\n"
            "Fill PolicyEvidence ONLY with values returned by the tools. Never invent data. "
            "If a tool returns an error, do not guess; the system will record the failure."
        ),
        model=MODEL,
        model_settings=MODEL_SETTINGS,
        tools=[get_customer_policy, get_policy_coverage, calculate_deductible],
        output_type=output_schema(PolicyEvidence),
    )
    return await _run(agent, f"Claim facts: {facts.model_dump_json()}", ctx, results,
                      _policy_truth, PolicyEvidence)


# ---------------------------------------------------------------- history worker
def _history_truth(r: dict) -> dict:
    _need(r, "history")
    h = r["history"]
    return {
        "previous_claims_count": h["previous_claims_count"],
        "claims_last_90_days": h["claims_last_90_days"],
        "suspicious_pattern": h["suspicious_pattern"],
        "notes": h["notes"],
    }


async def run_history_worker(facts: ClaimFacts, ctx: ClaimContext) -> HistoryEvidence:
    results: dict = {}

    @function_tool
    def check_claim_history(ctx: RunContextWrapper[ClaimContext], customer_id: str) -> str:
        """Get the customer's previous claims and whether the pattern looks suspicious."""
        return _call(ctx, results, "history", history_tools.check_claim_history,
                     customer_id, customer_id=customer_id)

    agent = Agent(
        name="History worker",
        instructions=(
            "You check a customer's claim history. Call check_claim_history once, then "
            "fill HistoryEvidence ONLY with values returned by the tool. Never invent data. "
            "If the tool returns an error, do not guess; the system will record the failure."
        ),
        model=MODEL,
        model_settings=MODEL_SETTINGS,
        tools=[check_claim_history],
        output_type=output_schema(HistoryEvidence),
    )
    return await _run(agent, f"Claim facts: {facts.model_dump_json()}", ctx, results,
                      _history_truth, HistoryEvidence)


# ---------------------------------------------------------------- docs worker
def _docs_truth(r: dict) -> dict:
    _need(r, "docs")
    d = r["docs"]
    return {
        "required_documents": d["required_documents"],
        "received_documents": d["received_documents"],
        "missing_documents": d["missing_documents"],
        "is_complete": d["is_complete"],
    }


async def run_docs_worker(facts: ClaimFacts, ctx: ClaimContext) -> DocsEvidence:
    results: dict = {}

    @function_tool
    def check_required_documents(ctx: RunContextWrapper[ClaimContext], claim_type: str,
                                 submitted_documents: list[str]) -> str:
        """Compare the submitted documents with the documents required for the claim type."""
        return _call(ctx, results, "docs", docs_tools.check_required_documents,
                     claim_type, submitted_documents)

    agent = Agent(
        name="Docs worker",
        instructions=(
            "You check whether a claim has all required documents. Call "
            "check_required_documents with the claim type and the submitted documents "
            "listed in the input, then fill DocsEvidence ONLY with values returned by the "
            "tool. Never invent data. If the tool returns an error, do not guess; the "
            "system will record the failure."
        ),
        model=MODEL,
        model_settings=MODEL_SETTINGS,
        tools=[check_required_documents],
        output_type=output_schema(DocsEvidence),
    )
    prompt = (
        f"Claim facts: {facts.model_dump_json()}\n"
        f"Submitted documents: {json.dumps(ctx.attached_documents)}"
    )
    return await _run(agent, prompt, ctx, results, _docs_truth, DocsEvidence)


# name -> function, used by the orchestrator
WORKERS = {
    "policy": run_policy_worker,
    "history": run_history_worker,
    "docs": run_docs_worker,
}