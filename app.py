"""
app.py: Gradio interface for the AI Insurance Claims Assessment Agent.

Run from the project root (the folder that contains claims_agent/, pipeline/, tools/):
    python app.py
Then open the local address Gradio prints (usually http://127.0.0.1:7860).

One click on "Assess claim" runs the whole pipeline:
  1. run_intake_and_evidence_async()  (A) -> EvidenceBundle or HumanReviewNeeded
  2. run_revision_loop()              (B) -> ClaimSummary or NeedsManual
  3. send_to_human()                  (B) -> output guardrail + saves the case

The app never approves or rejects a claim. A human adjuster always decides.
"""

from pipeline import llm_config  # noqa: F401

import json
import os
import uuid
from pathlib import Path

import gradio as gr
from agents import trace

from claims_agent.context import ClaimContext
from claims_agent.models import ClaimSummary, EvidenceBundle, HumanReviewNeeded, NeedsManual
from pipeline.human_review import send_to_human
from pipeline.orchestrator_in import run_intake_and_evidence_async
from pipeline.revision_loop import run_revision_loop

# ---------------------------------------------------------------------------
# Form choices (read from the mock data, with a safe fallback)
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).resolve().parent / "data" / "mock_data.json"


def _load_mock_data() -> dict:
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_MOCK = _load_mock_data()

CUSTOMER_CHOICES = [
    (f"{cid} - {info.get('name', 'customer')}", cid)
    for cid, info in sorted(_MOCK.get("customers", {}).items())
] or [(f"C{n:03d}", f"C{n:03d}") for n in range(1, 11)]

DOCUMENT_CHOICES: list[str] = []
for _docs in _MOCK.get("required_documents", {}).values():
    for _doc in _docs:
        if _doc not in DOCUMENT_CHOICES:
            DOCUMENT_CHOICES.append(_doc)
if not DOCUMENT_CHOICES:
    DOCUMENT_CHOICES = ["police_report", "photos", "repair_estimate"]

AUTO_DOCS = ["police_report", "photos", "repair_estimate"]

# Same demo cases as run_demo.py: [customer_id, claim text, attached documents]
EXAMPLES = [['C001', 'On 2026-10-10 another driver rear-ended my car at a traffic light. The rear bumper repair estimate is 8000 EGP. I attached the police report, photos, and repair estimate.', ['police_report', 'photos', 'repair_estimate']],
    ['C004', 'On 2026-10-09 an indoor water supply pipe leaked and damaged my kitchen ceiling. Repairs are estimated at 15000 EGP. I attached photos, a repair estimate, and ownership proof.', ['photos', 'repair_estimate', 'ownership_proof']],
    ['C009', 'On 2026-10-07 I underwent a medically necessary outpatient procedure. The bill was 12000 EGP. I attached the medical report, invoice, and prescription.', ['medical_report', 'invoice', 'prescription']],
    ['C011', 'On 2026-10-08 another vehicle hit my parked car, damaging the rear door. The repair estimate is 6500 EGP. Police report, photos, and repair estimate are attached.', ['police_report', 'photos', 'repair_estimate']],
    ['C012', 'On 2026-10-06 a bathroom supply pipe leaked and damaged the bedroom wall. Repairs are estimated at 11000 EGP. Photos, repair estimate, and ownership proof are attached.', ['photos', 'repair_estimate', 'ownership_proof']],
    ['C001', 'On 2026-10-11 a shopping cart scratched my parked car door. The repair estimate is 3200 EGP. Police report, photos, and repair estimate are attached.', ['police_report', 'photos', 'repair_estimate']],
    ['C004', 'On 2026-10-05 an interior plumbing connector broke and damaged the hallway plaster. The repair estimate is 9500 EGP. Photos, repair estimate, and ownership proof are attached.', ['photos', 'repair_estimate', 'ownership_proof']],
    ['C009', 'On 2026-10-03 I had a routine diagnostic consultation and tests prescribed by my doctor. The total invoice is 2400 EGP. Medical report, invoice, and prescription are attached.', ['medical_report', 'invoice', 'prescription']],
    ['C011', 'On 2026-10-12 my car side mirror was broken by another vehicle while parked. Replacement costs 2800 EGP. Police report, photos, and repair estimate are attached.', ['police_report', 'photos', 'repair_estimate']],
    ['C012', 'On 2026-10-13 an internal washing-machine hose leaked and damaged my laundry room flooring. Repairs cost 7200 EGP. Photos, repair estimate, and ownership proof are attached.', ['photos', 'repair_estimate', 'ownership_proof']],
    ['C002', 'On 2026-10-10 my car was totally destroyed in a collision. The claimed amount is 480000 EGP.', ['police_report', 'photos', 'repair_estimate']],
    ['C003', 'On 2026-10-10 a stone cracked my windshield. Repair estimate 6000 EGP.', ['police_report', 'photos', 'repair_estimate']],
    ['C005', 'On 2026-10-10 someone hit my parked car door. Repair estimate 9000 EGP.', ['police_report', 'photos', 'repair_estimate']],
    ['C006', 'On 2026-08-20 my car was scratched in a parking lot. Repair estimate 3000 EGP.', ['police_report', 'photos', 'repair_estimate']],
    ['C007', 'On 2026-10-05 my car crashed during a track racing event. Repair estimate 25000 EGP.', ['police_report', 'photos', 'repair_estimate']],
    ['C008', 'On 2026-10-08 I visited a hospital after a fall. The bill is 7000 EGP.', ['medical_report', 'invoice', 'prescription']],
    ['C010', 'On 2026-10-08 my flight was cancelled and my luggage was delayed. I request 5000 EGP.', ['booking_confirmation', 'incident_proof', 'receipts']],
    ['C999', 'On 2026-10-10 my parked car was scratched. Repair estimate 3000 EGP.', ['police_report', 'photos', 'repair_estimate']]]

def autofill_customer(customer_id: str) -> tuple[str, list[str]]:
    """Fill the claim and document checkboxes when a customer is selected."""
    for example_customer, example_claim, example_docs in EXAMPLES:
        if example_customer == customer_id:
            return example_claim, [doc for doc in example_docs if doc in DOCUMENT_CHOICES]

    customer_info = _MOCK.get("customers", {}).get(customer_id, {})
    return customer_info.get("_test_claim", ""), []


# ---------------------------------------------------------------------------
# Turning results into readable text
# ---------------------------------------------------------------------------
def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- none"


def render_result(result) -> tuple[str, str]:
    """Return (headline, details) as markdown for any final result type."""
    if isinstance(result, ClaimSummary):
        headline = "### Summary ready for the human adjuster"
        details = (
            f"**Claim:** {result.claim_id}  \n"
            f"**Status:** {result.status} (a human adjuster makes the decision)\n\n"
            f"#### Summary\n{result.summary}\n\n"
            f"#### Coverage evidence\n{result.coverage_evidence}\n\n"
            f"#### Risk flags\n{_bullets(result.risk_flags)}\n\n"
            f"#### Missing documents\n{_bullets(result.missing_documents)}\n\n"
            f"#### Estimated deductible\n{result.estimated_deductible:,.0f} EGP\n\n"
            f"#### Evidence gaps\n{_bullets(result.gaps)}"
        )
    elif isinstance(result, HumanReviewNeeded):
        headline = "### Sent straight to human review"
        details = (
            f"**Claim:** {result.claim_id}\n\n"
            f"#### Why\n{result.reason}\n\n"
            f"#### Risk flags\n{_bullets(result.risk_flags)}"
        )
    elif isinstance(result, NeedsManual):
        headline = "### Needs manual review"
        details = (
            f"**Claim:** {result.claim_id}\n\n"
            f"#### Why\n{result.reason}\n\n"
            f"#### Last feedback\n{result.last_feedback or 'none'}"
        )
    else:
        headline = "### Unexpected result"
        details = f"`{type(result).__name__}`"
    return headline, details


def _format_log(ctx: ClaimContext) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(ctx.audit_log, start=1))


# ---------------------------------------------------------------------------
# The button handler
# ---------------------------------------------------------------------------
async def assess_claim(customer_id, claim_text, documents):
    customer_id = (customer_id or "").strip()
    claim_text = (claim_text or "").strip()

    if not customer_id or not claim_text:
        return "### Missing information", "Choose a customer and describe the claim.", "", ""
    if not os.getenv("GEMINI_API_KEY"):
        return (
            "### API key not found",
            "Add `GEMINI_API_KEY=your-key` to the `.env` file, then restart the app.",
            "",
            "",
        )

    ctx = ClaimContext(claim_id=f"CL-{uuid.uuid4().hex[:8].upper()}")
    try:
        with trace("claim_assessment"):
            evidence = await run_intake_and_evidence_async(
                claim_text, customer_id, list(documents or []), ctx
            )
            if isinstance(evidence, EvidenceBundle):
                result = await run_revision_loop(evidence, ctx)
            else:
                result = evidence  # HumanReviewNeeded: skip the writer, go to a human
            outcome = send_to_human(result, ctx)
    except Exception as exc:  # never crash the interface
        ctx.log(f"App error: {type(exc).__name__}: {exc}")
        return "### Something went wrong", f"`{type(exc).__name__}`: {exc}", "", _format_log(ctx)

    headline, details = render_result(outcome["result"])
    if outcome.get("error"):
        case_text = f"Case NOT saved: {outcome['error']}"
    else:
        case_text = f"Case #{outcome['case_id']} saved for the human adjuster"
    return headline, details, case_text, _format_log(ctx)


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Insurance Claims Assessment", theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown(
        "# Insurance claims assessment\n"
        "Describe a claim and the assistant prepares a case summary for a human adjuster. "
        "It never approves or rejects a claim and never promises a payment."
    )
    with gr.Row():
        with gr.Column(scale=1):
            customer = gr.Dropdown(
                choices=CUSTOMER_CHOICES,
                value=CUSTOMER_CHOICES[0][1],
                label="Customer",
                allow_custom_value=True,
            )
            claim_text = gr.Textbox(
                label="What happened?",
                lines=6,
                placeholder="Example: Yesterday my car was rear-ended at a traffic light. The repair estimate is 8000 EGP.",
            )
            documents = gr.CheckboxGroup(choices=DOCUMENT_CHOICES, label="Documents attached")
            customer.change(
                fn=autofill_customer,
                inputs=customer,
                outputs=[claim_text, documents],
            )
            submit = gr.Button("Assess claim", variant="primary")
        with gr.Column(scale=1):
            headline = gr.Markdown("Results appear here after you assess a claim.")
            details = gr.Markdown()
            case_box = gr.Textbox(label="Saved case", interactive=False)
            with gr.Accordion("Step-by-step audit log", open=False):
                audit = gr.Textbox(lines=10, show_label=False, interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[customer, claim_text, documents],
        cache_examples=False,
        label="Demo cases (click one, then Assess claim)",
    )

    demo.load(
        fn=autofill_customer,
        inputs=customer,
        outputs=[claim_text, documents],
    )

    submit.click(
        assess_claim,
        inputs=[customer, claim_text, documents],
        outputs=[headline, details, case_box, audit],
    )

if __name__ == "__main__":
    demo.launch()