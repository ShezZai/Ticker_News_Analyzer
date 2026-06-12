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


# ---------------------------------------------------------------------------
# Run-level evaluators
# ---------------------------------------------------------------------------

def _expected_label(item) -> str | None:
    expected = item.get("expected_output") if isinstance(item, dict) else item.expected_output
    return (expected or {}).get("label")


def make_totals_run_evaluator(started_monotonic: float):
    """Run-level wall-clock/cost/token totals. The closure pins the start time
    taken just before run_experiment; the evaluator runs after the last item."""

    def totals_run_evaluator(*, item_results, **kwargs) -> list[Evaluation]:
        outputs = [r.output for r in item_results]
        evals = [Evaluation(
            name="total_time_s",
            value=time.monotonic() - started_monotonic,
            comment=f"{len(item_results)} items",
        )]
        lats = [o["latency_s"] for o in outputs if o and o.get("latency_s") is not None]
        if lats:
            evals.append(Evaluation(name="avg_time_per_item_s",
                                    value=sum(lats) / len(lats),
                                    comment=f"{len(lats)} timed items"))
        costs = [item_cost_usd(o) for o in outputs]
        known = [c for c in costs if c is not None]
        if known:
            evals.append(Evaluation(name="total_cost_usd", value=sum(known),
                                    comment=f"{len(known)}/{len(costs)} items with usage"))
        tin = sum(o.get("input_tokens") or 0 for o in outputs if o)
        tout = sum(o.get("output_tokens") or 0 for o in outputs if o)
        if tin or tout:
            evals.append(Evaluation(name="total_tokens", value=tin + tout,
                                    comment=f"input={tin} output={tout}"))
        return evals

    return totals_run_evaluator


def label_accuracy_run_evaluator(*, item_results, **kwargs) -> list[Evaluation]:
    """Fraction of labeled items predicted exactly right; an errored task
    (output None) counts as wrong, not absent."""
    labeled = [r for r in item_results if _expected_label(r.item) is not None]
    if not labeled:
        return [Evaluation(name="label_accuracy_avg_skip", value="no labeled items")]
    correct = sum(
        1 for r in labeled
        if r.output and r.output.get("label") == _expected_label(r.item)
    )
    return [Evaluation(name="label_accuracy_avg", value=correct / len(labeled),
                       comment=f"{correct}/{len(labeled)} exact (errored = wrong)")]


def binary_confusion_run_evaluator(*, item_results, **kwargs) -> list[Evaluation]:
    """Precision/recall/F1 for the YES class (the GT is 42 YES / 98 NO —
    accuracy alone would flatter an always-NO classifier). An errored task is
    always wrong: FN on YES items, FP on NO items."""
    tp = fp = fn = tn = 0
    for r in item_results:
        expected = _expected_label(r.item)
        predicted = (r.output or {}).get("label")
        if expected == "YES":
            tp, fn = (tp + 1, fn) if predicted == "YES" else (tp, fn + 1)
        elif expected == "NO":
            wrong = predicted != "NO"
            fp, tn = (fp + 1, tn) if wrong else (fp, tn + 1)
    if tp + fp + fn + tn == 0:
        return [Evaluation(name="act_metrics_skip", value="no scorable items")]
    counts = f"TP={tp} FP={fp} FN={fn} TN={tn}"
    evals: list[Evaluation] = []
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None:
        evals.append(Evaluation(name="act_precision_skip",
                                value=f"no YES predictions ({counts})"))
    else:
        evals.append(Evaluation(name="act_precision", value=precision, comment=counts))
    if recall is None:
        evals.append(Evaluation(name="act_recall_skip",
                                value=f"no YES items ({counts})"))
    else:
        evals.append(Evaluation(name="act_recall", value=recall, comment=counts))
    if precision is not None and recall is not None:
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        evals.append(Evaluation(name="act_f1", value=f1, comment=counts))
    return evals


def derived_act_run_evaluator(*, item_results, **kwargs) -> list[Evaluation]:
    """Finegrained only: collapse expected and predicted categories through
    NEWS_SUBTYPES to YES/NO — do miscategorizations cross the ACT boundary?"""
    from ticker_news.classification.variants import is_act_finegrained

    labeled = [r for r in item_results if _expected_label(r.item) is not None]
    if not labeled:
        return [Evaluation(name="derived_act_accuracy_skip", value="no labeled items")]
    correct = 0
    for r in labeled:
        expected_act = is_act_finegrained(_expected_label(r.item))
        predicted = (r.output or {}).get("label")
        if predicted is not None and is_act_finegrained(predicted) is expected_act:
            correct += 1
    return [Evaluation(name="derived_act_accuracy", value=correct / len(labeled),
                       comment=f"{correct}/{len(labeled)} on the right ACT side")]


# Completed in Task 5 (run evaluators + dataclass); placeholder keeps imports working.
EXPERIMENTS: dict = {}
