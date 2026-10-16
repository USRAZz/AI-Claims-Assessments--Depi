"""
guardrails_in.py: INPUT guardrail (plain Python, no LLM).

check_input() runs BEFORE any agent sees the text. It returns:
  - a reason (str) -> the text is blocked
  - None           -> the text is allowed
"""

import re
from typing import Optional

MAX_LENGTH = 4000

_RULES = [
    ("prompt injection", [
        r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your)\s+(instructions?|rules?|prompts?)",
        r"disregard\s+.{0,30}(instructions?|rules?)",
        r"(reveal|show|print|repeat)\s+.{0,20}(system|hidden)\s+prompt",
        r"\bsystem\s+prompt\b",
        r"\byou\s+are\s+now\b",
        r"developer\s+mode",
        r"\bjailbreak",
        r"new\s+instructions?\s*:",
        r"override\s+(the\s+)?(rules?|instructions?|guardrails?)",
    ]),
    ("claim manipulation", [
        r"say\s+(it|that\s+it)\s+happened",
        r"change\s+the\s+(date|amount)",
        r"back\s*-?date",
        r"make\s+it\s+(look|seem)",
        r"\bpretend\b",
        r"\bforge[d]?\b",
        r"fake\s+(the\s+)?(report|document|receipt|date)",
        r"(approve|accept)\s+(my|this|the)\s+claim",
        r"mark\s+(it|this|my\s+claim)\s+as\s+(approved|paid)",
        r"guarantee\s+(me\s+)?(a\s+)?(payment|refund|payout)",
    ]),
    ("policy change request", [
        r"(raise|increase|extend|upgrade)\s+(my\s+)?(coverage|limit|policy)",
        r"change\s+my\s+(policy|coverage|deductible|plan)",
        r"(lower|reduce|remove)\s+(my\s+)?(deductible|exclusions?)",
        r"(add|remove)\s+(coverage|an?\s+exclusion)",
        r"cancel\s+my\s+policy",
    ]),
]


def check_input(text: str) -> Optional[str]:
    """Return a reason if the text must be blocked, otherwise None."""
    if not isinstance(text, str) or not text.strip():
        return "empty input: the claim text is missing"
    if len(text) > MAX_LENGTH:
        return f"input too long: more than {MAX_LENGTH} characters"

    normalized = " ".join(text.split())
    for category, patterns in _RULES:
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                return f"{category}: found '{match.group(0)}'"
    return None