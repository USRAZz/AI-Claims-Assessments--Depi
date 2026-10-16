from agents import Agent

from claims_agent.models import EvaluationResult
from pipeline.llm_config import MODEL, MODEL_SETTINGS, output_schema

claim_evaluator = Agent(
    name="Claim Evaluator",
    instructions=(
        "Compare the draft summary to the evidence. Set has_unsupported_promise if it promises "
        "or implies approval/payment/refund, or states anything the evidence does not support. "
        "Set has_missing_evidence if evidence shows missing documents or gaps that the draft ignores. "
        "Set has_contradiction if the draft contradicts the evidence or itself "
        "(e.g. says policy is active when it is expired, or says the loss is covered when the "
        "evidence shows policy.is_covered is false or an exclusion matched). "
        "is_acceptable is True ONLY if all three are False. "
        "If not acceptable, write specific feedback telling the writer what to fix."
    ),
    model=MODEL,
    model_settings=MODEL_SETTINGS,
    output_type=output_schema(EvaluationResult),
)