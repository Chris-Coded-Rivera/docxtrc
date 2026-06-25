"""
verify.py — the reliability backbone.

Structured outputs guarantee the *shape* of the answer. They do not guarantee
the answer is *true*. So for every field, we asked the model for the verbatim
quote it pulled the value from (see schema.Evidence), and here we check —
deterministically, with no extra model call — that the quote actually appears
in the source document. A field whose quote isn't in the text is flagged as
ungrounded: a cheap, high-signal hallucination check.

This is non-destructive: we return a report rather than mutating the result,
so the caller decides policy (null it out, re-ask, route to a human, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class VerificationReport:
    grounded: dict[str, bool]            # field name -> was its quote found in the text
    coverage: float                      # fraction of filled fields that are grounded

    @property
    def ungrounded(self) -> list[str]:
        return [f for f, ok in self.grounded.items() if not ok]


def _norm(text: str) -> str:
    """Collapse whitespace so quotes match despite reflowed line breaks."""
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_evidence(result: BaseModel, document: str) -> VerificationReport:
    haystack = _norm(document)

    # Map field -> quote from the evidence list the model returned.
    quotes: dict[str, str] = {}
    for item in getattr(result, "evidence", []) or []:
        quotes[item.field] = item.quote

    grounded: dict[str, bool] = {}
    for name in result.__class__.model_fields:
        if name in ("summary", "evidence"):
            continue
        value = getattr(result, name, None)
        if value is None:
            continue  # nothing claimed, nothing to verify
        quote = quotes.get(name, "")
        grounded[name] = bool(quote) and _norm(quote) in haystack

    coverage = (sum(grounded.values()) / len(grounded)) if grounded else 1.0
    return VerificationReport(grounded=grounded, coverage=coverage)
