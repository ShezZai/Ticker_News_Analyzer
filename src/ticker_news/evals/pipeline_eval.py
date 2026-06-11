"""E2E pipeline eval: re-run articles through the real stage chain against the
shared DB, then score the verdict against the actual price move as a Langfuse
experiment.

Scoring is directional agreement: buy+up = 1, sell+down = 1, wrong direction
= 0; hold / no-verdict / no-price-data are excluded (value None) with an
explanatory comment. The raw entry->close move is recorded as a second score.

Design: docs/superpowers/specs/2026-06-11-e2e-pipeline-eval-design.md
"""

from __future__ import annotations


def score_directional(
    action: str | None, gain_pct: float | None, *, skip_reason: str | None = None
) -> tuple[float | None, str]:
    """Directional-agreement score for one verdict vs the realized move.

    Returns (value, comment) where value is 1.0 / 0.0, or None when the item
    cannot be scored (hold, no verdict, no price data) — None scores are
    excluded from Langfuse aggregates by the run evaluator.
    """
    if action is None:
        return None, f"no verdict ({skip_reason or 'sentiment skipped'})"
    if action == "hold":
        return None, "hold verdict - no direction to verify"
    if action not in ("buy", "sell"):
        return None, f"unknown action '{action}'"
    if gain_pct is None:
        return None, f"no price data ({skip_reason or 'unknown'})"
    correct = gain_pct > 0 if action == "buy" else gain_pct < 0
    verdict = "agree" if correct else "disagree"
    return (1.0 if correct else 0.0), f"{action} with {gain_pct:+.2f}% by close -> {verdict}"
