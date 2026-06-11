"""Experimental classification prompt variants: binary and fine-grained.

Candidate replacements for the production classifier, evaluated by
`ticker-news eval classify` against the hand-labeled ground-truth set
(spec: docs/superpowers/specs/2026-06-12-classify-eval-design.md).
Not wired into the pipeline; promoting a winner stays a human decision.
"""

from __future__ import annotations

from typing import Literal, Optional, get_args

from pydantic import BaseModel

BinaryLabel = Literal["real news", "none news"]
BINARY_LABELS: list[str] = list(get_args(BinaryLabel))


class BinaryClassification(BaseModel):
    """Structured verdict of the binary ACT / DON'T-ACT classifier."""

    label: BinaryLabel
    confidence: Optional[float] = None
    reason: str = ""


FinegrainedCategory = Literal[
    # --- a real, newly-occurring event is being reported (NEWS) ---
    "earnings-reporting",
    "dividend-reporting",
    "merger/investment/funding",
    "legal-event",
    "MACRO-investment",
    "news-event",
    "news-report",
    # --- not a new event (NOT-NEWS) ---
    "recap/review",
    "market speculation",
    "MACRO-political",
    "legal-call",
    "conference-PR",
    "marketing fluff",
    "book PR",
    "Other-filing-reporting",
    "other",
]
FINEGRAINED_CATEGORIES: list[str] = list(get_args(FinegrainedCategory))

# Fine-grained categories that count as ACT/YES for the binary ground truth.
NEWS_SUBTYPES: frozenset[str] = frozenset({
    "earnings-reporting",
    "dividend-reporting",
    "merger/investment/funding",
    "legal-event",
    "MACRO-investment",
    "news-event",
    "news-report",
})


class FinegrainedClassification(BaseModel):
    """Structured verdict of the fine-grained taxonomy classifier."""

    category: FinegrainedCategory
    reason: str = ""


def is_act_binary(label: str) -> bool:
    return label == "real news"


def is_act_finegrained(category: str) -> bool:
    return category in NEWS_SUBTYPES
