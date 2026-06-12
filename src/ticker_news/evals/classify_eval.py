"""Single-pass classification prompt experiments against Langfuse datasets.

Two experiments, one LLM call per dataset item, everything else deterministic:
- binary      -> dataset 140-articles-act-no-act   (expected {"label": YES|NO})
- finegrained -> dataset 140-articles-categories   (expected {"label": <category>})

Read-only: article text is prefetched from the DB in one query; production
pipeline tables are never written. Scores include accuracy, per-item latency
and cost, and run-level totals (time, cost, tokens).

Design: docs/superpowers/specs/2026-06-12-classify-single-pass-experiments-design.md
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import psycopg
from langfuse import Evaluation

from ticker_news.evals.pipeline_eval import connect_eval

# USD per 1M tokens (input, output); paid tier, verified 2026-06-13.
GEMINI_PRICES_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}


def item_cost_usd(output) -> float | None:
    """Cost of one item from its token usage, or None when unknowable.

    Longest price-table key wins so 'gemini-2.5-flash' never shadows
    'gemini-2.5-flash-lite' in a substring match.
    """
    if not output:
        return None
    model = output.get("model") or ""
    tokens_in, tokens_out = output.get("input_tokens"), output.get("output_tokens")
    if tokens_in is None or tokens_out is None:
        return None
    for known in sorted(GEMINI_PRICES_USD_PER_1M, key=len, reverse=True):
        if known in model:
            p_in, p_out = GEMINI_PRICES_USD_PER_1M[known]
            return (tokens_in * p_in + tokens_out * p_out) / 1_000_000
    return None


def label_accuracy_evaluator(*, output, expected_output, **kwargs) -> Evaluation:
    """Predicted dataset-space label vs the item's expected label (1.0 / 0.0)."""
    expected = (expected_output or {}).get("label")
    if expected is None:
        return Evaluation(name="label_accuracy_skip", value="no expected label")
    if not output:
        return Evaluation(name="label_accuracy", value=0.0,
                          comment=f"no output, expected={expected}")
    predicted, label = output.get("predicted"), output.get("label")
    return Evaluation(
        name="label_accuracy", value=1.0 if label == expected else 0.0,
        comment=f"predicted={predicted!r} -> {label}, expected={expected}",
    )


def predicted_label_evaluator(*, output, **kwargs) -> Evaluation:
    """Raw predicted label/category (categorical) — misclassifications are
    filterable in the UI."""
    predicted = (output or {}).get("predicted")
    return Evaluation(name="predicted_label", value=predicted or "<none>")


def latency_evaluator(*, output, **kwargs) -> Evaluation:
    lat = (output or {}).get("latency_s")
    if lat is None:
        return Evaluation(name="latency_s_skip", value="no latency recorded")
    return Evaluation(name="latency_s", value=lat)


def cost_evaluator(*, output, **kwargs) -> Evaluation:
    cost = item_cost_usd(output)
    if cost is None:
        model = (output or {}).get("model") or "<none>"
        return Evaluation(name="cost_usd_skip", value=f"unknown model/usage: {model}")
    return Evaluation(name="cost_usd", value=cost)


SHARED_EVALUATORS = (
    label_accuracy_evaluator,
    predicted_label_evaluator,
    latency_evaluator,
    cost_evaluator,
)

# Completed in Task 5 (run evaluators + dataclass); placeholder keeps imports working.
EXPERIMENTS: dict = {}
