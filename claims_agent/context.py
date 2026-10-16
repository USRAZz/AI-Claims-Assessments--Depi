"""
context.py: the object passed to agents through RunContextWrapper.

It carries information that every agent/tool in one claim run may need,
without putting it in the prompt text.
"""

from dataclasses import dataclass, field


@dataclass
class ClaimContext:
    claim_id: str
    customer_id: str = ""
    raw_claim_text: str = ""
    attached_documents: list[str] = field(default_factory=list)
    revision_count: int = 0                      # used by the revision loop
    audit_log: list[str] = field(default_factory=list)  # short step-by-step notes

    def log(self, message: str) -> None:
        self.audit_log.append(message)