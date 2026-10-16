"""
docs_tools.py: required-documents tool used by the Docs worker.

Plain Python (no LLM). Keys of the returned dict match the fields of DocsEvidence.
"""

from tools.common import ToolDataError, check_claim_type, load_data


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def check_required_documents(claim_type: str, submitted_documents: list[str]) -> dict:
    """Compare the submitted documents with the documents required for this claim type.

    Extra documents that are not on the required list are ignored.
    """
    check_claim_type(claim_type)
    if not isinstance(submitted_documents, list) or not all(
        isinstance(d, str) for d in submitted_documents
    ):
        raise ToolDataError("submitted_documents must be a list of strings")

    required = load_data().get("required_documents", {}).get(claim_type)
    if required is None:
        raise ToolDataError(f"No required-documents list for claim type '{claim_type}'")

    submitted = {_normalize(d) for d in submitted_documents if d.strip()}
    received = [d for d in required if d in submitted]
    missing = [d for d in required if d not in submitted]
    return {
        "required_documents": list(required),
        "received_documents": received,
        "missing_documents": missing,
        "is_complete": not missing,
    }
