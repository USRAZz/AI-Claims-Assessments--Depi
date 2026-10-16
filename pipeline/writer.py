from agents import Agent

from claims_agent.models import ClaimSummary
from pipeline.llm_config import MODEL, MODEL_SETTINGS, output_schema

claim_writer = Agent(
    name="Claim Writer",
    instructions=(
        "You prepare a case summary for a HUMAN insurance adjuster from the evidence provided. "
        "The evidence has these parts: facts (claim details and risk_flags), policy, history, "
        "docs and gaps. "
        "Use ONLY facts in the evidence. Never decide coverage and never promise approval, "
        "payment or refund. Do not use phrases like 'will be approved' or 'will be paid'. "
        "Describe coverage evidence neutrally (e.g. 'policy is active and the loss type appears covered'). "
        "If policy.is_covered is false, say neutrally that the claim type does not appear in the policy. "
        "If policy.exclusions_matched is not empty, say neutrally that an exclusion may apply and the "
        "adjuster must review it. Do not decide. "
        "Copy risk flags from facts.risk_flags and missing documents from docs.missing_documents. "
        "Set estimated_deductible from the policy deductible; if there is no policy evidence use 0 "
        "and add a note to gaps. Copy any evidence gaps into gaps. "
        "If you receive revision feedback, fix exactly those problems."
    ),
    model=MODEL,
    model_settings=MODEL_SETTINGS,
    output_type=output_schema(ClaimSummary),
)