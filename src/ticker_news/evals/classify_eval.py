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


def prefetch_articles(
    conn: psycopg.Connection, ids: list[int]
) -> dict[int, tuple[str, str]]:
    """One query for every article body up front — the experiment task then
    does zero DB work (the per-item fetch over the tunneled shared DB used to
    dwarf the LLM call). Loud failure on unusable articles."""
    rows = conn.execute(
        "SELECT id, title, coalesce(content, '') FROM public.articles WHERE id = ANY(%s)",
        (ids,),
    ).fetchall()
    found = {row[0]: (row[1] or "", row[2]) for row in rows}
    missing = sorted(set(ids) - set(found))
    if missing:
        raise ValueError(f"article ids not found: {missing}")
    empty = sorted(aid for aid, (_, content) in found.items() if not content.strip())
    if empty:
        raise ValueError(f"articles have no scraped content: {empty}")
    return found


def make_task(classifier, articles: dict[int, tuple[str, str]], trace_prefix: str):
    """Async experiment task: exactly one LLM call, everything else local."""

    async def classify_task(*, item, **kwargs) -> dict:
        from langchain_core.callbacks import UsageMetadataCallbackHandler
        from langfuse import propagate_attributes

        from ticker_news.shared import observability as obs

        data = item["input"] if isinstance(item, dict) else item.input
        article_id = data["article_id"]
        title, content = articles[article_id]
        # fresh handler per invocation — callbacks are not thread-safe to share
        usage = UsageMetadataCallbackHandler()
        cfg = obs.chain_config() or {}
        cfg = {**cfg, "callbacks": [*cfg.get("callbacks", []), usage]}
        t0 = time.monotonic()
        # The SDK names every experiment-item trace AND its root span
        # "experiment-item-run"; rename both.
        with propagate_attributes(trace_name=f"{trace_prefix}:article-{article_id}"):
            if (lf := obs.client()) is not None:
                lf.update_current_span(name=f"{trace_prefix}:article-{article_id}")
            verdict = await classifier.classify(title, content, config=cfg)
        latency = time.monotonic() - t0
        tin = tout = None
        if usage.usage_metadata:  # {model: {input_tokens, output_tokens, ...}}
            tin = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
            tout = sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())
        return {
            "predicted": classifier.label_of(verdict),
            "label": classifier.dataset_label_of(verdict),
            "reason": verdict.reason or None,
            "confidence": getattr(verdict, "confidence", None),
            "latency_s": round(latency, 3),
            "input_tokens": tin,
            "output_tokens": tout,
            "model": classifier.model,
        }

    return classify_task


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    experiment_name: str
    evaluators: tuple
    run_evaluators: tuple  # totals evaluator is added per run (needs a start time)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "binary": ExperimentSpec(
        dataset="140-articles-act-no-act",
        experiment_name="classify-binary",
        evaluators=SHARED_EVALUATORS,
        run_evaluators=(label_accuracy_run_evaluator, binary_confusion_run_evaluator),
    ),
    "finegrained": ExperimentSpec(
        dataset="140-articles-categories",
        experiment_name="classify-finegrained",
        evaluators=SHARED_EVALUATORS,
        run_evaluators=(label_accuracy_run_evaluator, derived_act_run_evaluator),
    ),
}

_DESCRIPTION = (
    "Single-pass classification prompt experiment: one LLM call per article, "
    "scored against the dataset's expected label."
)


def _warn_failed_items(result, requested_ids: list[int]) -> None:
    """Failed items vanish from the result (SDK logs only); make them loud.

    Nothing is left dirty in the DB — the task is read-only — but a missing
    item silently skews the run metrics."""
    done: set[int] = set()
    for r in result.item_results:
        item = r.item
        data = item["input"] if isinstance(item, dict) else item.input
        done.add(data["article_id"])
    failed = sorted(set(requested_ids) - done)
    if failed:
        print(
            f"WARNING: {len(failed)} item(s) errored and are missing from the "
            f"run: {failed}. Re-run with --ids {','.join(map(str, failed))} "
            f"and the same --run-name to fill them in."
        )


def run_eval(
    variants: tuple[str, ...],
    *,
    model: str = "lite",
    dataset_name: str | None = None,
    dsn: str | None = None,
    run_name: str | None = None,
    ids: list[int] | None = None,
    concurrency: int = 16,
) -> list[tuple[str, object]]:
    """Run one experiment per variant. Returns [(variant, ExperimentResult), ...]."""
    from ticker_news.classification.variants import MODEL_CHOICES, make_classifier
    from ticker_news.shared import observability as obs
    from ticker_news.shared import prompts
    from ticker_news.shared.config import get_settings

    client = obs.client()
    if client is None:
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are required - "
            "eval results live in Langfuse."
        )
    if not get_settings().google_api_key:
        raise SystemExit("missing required keys: GOOGLE_API_KEY")
    if model not in MODEL_CHOICES:
        raise SystemExit(f"unknown model {model!r} (expected one of {sorted(MODEL_CHOICES)})")
    if dataset_name and len(variants) > 1:
        raise SystemExit("--dataset override requires a single --variant")

    model_name = MODEL_CHOICES[model]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: list[tuple[str, object]] = []
    try:
        for variant in variants:
            spec = EXPERIMENTS[variant]
            ds_name = dataset_name or spec.dataset
            dataset = client.get_dataset(ds_name)
            data = list(dataset.items)
            if not data:
                raise SystemExit(f"dataset '{ds_name}' has no items")
            if ids:
                wanted = set(ids)
                data = [it for it in data if (it.input or {}).get("article_id") in wanted]
                if not data:
                    raise SystemExit("none of the requested --ids are in the dataset")
                found_ids = {(it.input or {}).get("article_id") for it in data}
                dropped = sorted(wanted - found_ids)
                if dropped:
                    print(f"WARNING: --ids not in dataset '{ds_name}' (skipped): {dropped}")
            article_ids = [(it.input or {}).get("article_id") for it in data]
            conn = connect_eval(dsn)
            try:
                articles = prefetch_articles(conn, article_ids)
            finally:
                conn.close()
            classifier = make_classifier(variant, model_name)  # fetches prompts -> versions_seen
            if run_name:
                rn = f"{run_name}-{variant}" if len(variants) > 1 else run_name
            else:
                rn = f"{variant}-{model}-{stamp}"
            t0 = time.monotonic()
            result = client.run_experiment(
                name=spec.experiment_name,
                run_name=rn,
                description=_DESCRIPTION,
                data=data,
                task=make_task(classifier, articles, spec.experiment_name),
                evaluators=list(spec.evaluators),
                run_evaluators=[make_totals_run_evaluator(t0), *spec.run_evaluators],
                # async task -> max_concurrency gates real parallelism; the
                # shared Gemini rate limiter caps requests per second anyway.
                max_concurrency=concurrency,
                metadata={
                    "variant": variant,
                    "model": model_name,
                    "prompt_versions": prompts.versions_seen(),
                    "entrypoint": "eval",
                },
            )
            _warn_failed_items(result, article_ids)
            results.append((variant, result))
    finally:
        obs.flush()
    return results
