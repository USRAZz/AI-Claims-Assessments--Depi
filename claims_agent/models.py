"""
models.py: the shared "contract" between Person A (pipeline, tools) and
Person B (writer, evaluator, guardrails out, UI, case tools).

Every agent returns one of these models. Both teammates import from here.
Change this file only through a small pull request that both review.
All data in this project is synthetic (no real customers).
"""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ClaimType = Literal["auto", "health", "property", "travel", "other"]

# Phrases a claim summary must never contain (the LLM may not promise outcomes).
FORBIDDEN_PHRASES = [
    "claim is approved",
    "claim has been approved",
    "will be approved",
    "will be paid",
    "you will be paid",
    "we will refund",
    "guaranteed",
]


# ---------------------------------------------------------------------------
# A builds these
# ---------------------------------------------------------------------------
class ClaimFacts(BaseModel):
    """Output of the Intake agent (pipeline/intake.py)."""

    claim_id: str
    customer_id: str
    claim_type: ClaimType
    incident_date: date
    claimed_amount: float = Field(gt=0)
    description: str = Field(min_length=10)
    missing_fields: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class PolicyEvidence(BaseModel):
    """Output of the Policy worker."""

    policy_number: str
    policy_status: Literal["active", "expired", "cancelled"]
    policy_start_date: date
    is_covered: bool
    coverage_limit: float = Field(ge=0)
    deductible: float = Field(ge=0)
    exclusions_matched: list[str] = Field(default_factory=list)
    estimated_insurer_share: Optional[float] = None
    estimated_customer_share: Optional[float] = None


class HistoryEvidence(BaseModel):
    """Output of the History worker."""

    previous_claims_count: int = Field(ge=0)
    claims_last_90_days: int = Field(ge=0)
    suspicious_pattern: bool = False
    notes: str = ""


class DocsEvidence(BaseModel):
    """Output of the Docs worker."""

    required_documents: list[str]
    received_documents: list[str] = Field(default_factory=list)
    missing_documents: list[str] = Field(default_factory=list)
    is_complete: bool

    @model_validator(mode="after")
    def complete_means_nothing_missing(self):
        if self.is_complete and self.missing_documents:
            raise ValueError("is_complete is True but missing_documents is not empty")
        return self


class EvidenceBundle(BaseModel):
    """Output of merge_evidence(). A failed worker leaves its field as None
    and adds a sentence to `gaps` (we never invent data)."""

    facts: ClaimFacts
    policy: Optional[PolicyEvidence] = None
    history: Optional[HistoryEvidence] = None
    docs: Optional[DocsEvidence] = None
    gaps: list[str] = Field(default_factory=list)


class HumanReviewNeeded(BaseModel):
    """Returned by risk_route() when a claim must stop and go to a human."""

    claim_id: str
    reason: str
    risk_flags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# B builds these
# ---------------------------------------------------------------------------
class ClaimSummary(BaseModel):
    """Output of the Claim writer. Prepared for a HUMAN adjuster.

    There is deliberately no 'approved' or 'rejected' value anywhere:
    the system never decides coverage.
    """

    claim_id: str
    summary: str = Field(min_length=20)
    coverage_evidence: str
    risk_flags: list[str] = Field(default_factory=list)
    missing_documents: list[str] = Field(default_factory=list)
    estimated_deductible: float = Field(ge=0)
    gaps: list[str] = Field(default_factory=list)
    status: Literal["PENDING-ADJUSTER", "AI-ASSESSED"] = "PENDING-ADJUSTER"

    @field_validator("summary", "coverage_evidence")
    @classmethod
    def no_promises(cls, value: str) -> str:
        lowered = value.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in lowered:
                raise ValueError(f"Must not promise an outcome: '{phrase}'")
        return value


class EvaluationResult(BaseModel):
    """Output of the Evaluator."""

    is_acceptable: bool
    has_unsupported_promise: bool = False
    has_missing_evidence: bool = False
    has_contradiction: bool = False
    feedback: str = ""  # what the writer must fix in the next revision

    @model_validator(mode="after")
    def acceptable_means_no_problems(self):
        problems = (
            self.has_unsupported_promise
            or self.has_missing_evidence
            or self.has_contradiction
        )
        if self.is_acceptable and problems:
            raise ValueError("is_acceptable cannot be True when a problem is flagged")
        if not self.is_acceptable and not self.feedback:
            raise ValueError("feedback is required when the draft is not acceptable")
        return self


class NeedsManual(BaseModel):
    """Returned by the revision loop when the draft is still rejected after
    MAX_REVISIONS attempts."""

    claim_id: str
    reason: str
    last_feedback: str = ""