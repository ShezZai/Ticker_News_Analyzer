"""Classification prompt-variant eval against the hand-labeled ground truth.

Read-only: loads article text from the DB, runs the binary and/or
fine-grained variant chains, and scores ACT/DON'T-ACT agreement with the
ground-truth labels as Langfuse experiments on a shared dataset. Never
writes to pipeline tables (the production `category` column is untouched).

Design: docs/superpowers/specs/2026-06-12-classify-eval-design.md
"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime
from pathlib import Path

import psycopg
from langfuse import Evaluation

from ticker_news.evals.pipeline_eval import connect_eval

DATASET_DEFAULT = "classify-ground-truth"

_REQUIRED_COLUMNS = {"article id", "Act_GT"}


def load_ground_truth(csv_path: str | Path) -> list[dict]:
    """Parse the GT csv into [{article_id, header, act}] with validation.

    utf-8-sig tolerates the Excel BOM; Act_GT is normalized to upper-case
    YES/NO; integer, unique article ids enforced. Raises ValueError with the
    offending line number on any violation.
    """
    rows: list[dict] = []
    seen: set[int] = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path}: missing required column(s): {', '.join(sorted(missing))}"
            )
        for lineno, row in enumerate(reader, start=2):
            raw_id = (row.get("article id") or "").strip()
            if not raw_id.isdigit():
                raise ValueError(f"{csv_path} line {lineno}: bad article id {raw_id!r}")
            article_id = int(raw_id)
            if article_id in seen:
                raise ValueError(
                    f"{csv_path} line {lineno}: duplicate article id {article_id}"
                )
            seen.add(article_id)
            act = (row.get("Act_GT") or "").strip().upper()
            if act not in ("YES", "NO"):
                raise ValueError(
                    f"{csv_path} line {lineno}: Act_GT must be YES or NO, got {act!r}"
                )
            rows.append({
                "article_id": article_id,
                "header": (row.get("header") or "").strip(),
                "act": act,
            })
    if not rows:
        raise ValueError(f"{csv_path}: ground-truth csv has no rows")
    return rows


def build_items(conn: psycopg.Connection, gt_rows: list[dict]) -> list[dict]:
    """GT rows -> Langfuse dataset items; loud failure on unusable articles.

    Bodies are NOT stored in the dataset — the DB row is the single source
    of truth (same convention as the pipeline eval); the task reads content
    by article id at run time.
    """
    ids = [r["article_id"] for r in gt_rows]
    db_rows = conn.execute(
        "SELECT id, title, status, coalesce(content, '') <> '' "
        "FROM public.articles WHERE id = ANY(%s)",
        (ids,),
    ).fetchall()
    found = {row[0]: row for row in db_rows}
    missing = sorted(set(ids) - set(found))
    if missing:
        raise ValueError(f"article ids not found: {missing}")
    bad = sorted(
        aid for aid, (_, _, status, has_content) in found.items()
        if status != "ok" or not has_content
    )
    if bad:
        raise ValueError(f"articles have no scraped content: {bad}")
    return [
        {
            "id": f"article-{r['article_id']}",
            "input": {
                "article_id": r["article_id"],
                "title": found[r["article_id"]][1] or "",
            },
            "expected_output": {"act": r["act"]},
            "metadata": {"gt_header": r["header"]},
        }
        for r in gt_rows
    ]


def act_accuracy_evaluator(*, output, expected_output, **kwargs) -> Evaluation:
    """Langfuse item evaluator: predicted ACT vs ground truth (1.0 / 0.0)."""
    expected = (expected_output or {}).get("act")
    if not output:
        return Evaluation(name="act_accuracy", value=0.0,
                          comment=f"no output, gt={expected}")
    predicted, act = output.get("predicted"), output.get("act")
    value = 1.0 if act == expected else 0.0
    return Evaluation(
        name="act_accuracy", value=value,
        comment=f"predicted={predicted!r} -> act={act}, gt={expected}",
    )


def predicted_label_evaluator(*, output, **kwargs) -> Evaluation:
    """Langfuse item evaluator: raw predicted label/category (categorical),
    so misclassifications are filterable in the UI."""
    predicted = (output or {}).get("predicted")
    return Evaluation(name="predicted_label", value=predicted or "<none>")


def _expected_act(item) -> str | None:
    expected = item.get("expected_output") if isinstance(item, dict) else item.expected_output
    return (expected or {}).get("act")


def act_metrics_run_evaluator(*, item_results, **kwargs) -> list[Evaluation]:
    """Run-level confusion metrics for the YES class.

    The GT is imbalanced (42 YES / 98 NO) — precision/recall/F1 keep an
    always-NO classifier from looking good. A task that errored (output None)
    always counts as wrong — FN on YES items, FP on NO items — rather than
    vanishing from the denominator (or being rewarded as a TN).
    """
    tp = fp = fn = tn = 0
    for r in item_results:
        expected = _expected_act(r.item)
        predicted = (r.output or {}).get("act")
        if expected == "YES":
            tp, fn = (tp + 1, fn) if predicted == "YES" else (tp, fn + 1)
        elif expected == "NO":
            # An errored task (r.output is None) is always wrong — treat as FP,
            # not TN. Only a genuine {"act": "NO"} output earns TN credit.
            wrong = predicted == "YES" or r.output is None
            fp, tn = (fp + 1, tn) if wrong else (fp, tn + 1)
    total = tp + fp + fn + tn
    if total == 0:
        return [Evaluation(name="act_metrics_skip", value="no scorable items")]
    counts = f"TP={tp} FP={fp} FN={fn} TN={tn}"
    evals = [Evaluation(
        name="act_accuracy_avg", value=(tp + tn) / total,
        comment=f"{counts}; {total} items",
    )]
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
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
        evals.append(Evaluation(name="act_f1", value=f1, comment=counts))
    return evals


EXPERIMENT_PREFIX = "classify"
_DESCRIPTION = (
    "Classification prompt-variant eval: predicted ACT/DON'T-ACT vs the "
    "hand-labeled ground truth (binary and fine-grained prompts)."
)


def make_task(runner, dsn: str | None):
    """Async experiment task: read the article, classify, return the verdict.

    Read-only — never writes to pipeline tables. A fresh connection per
    invocation (sync psycopg connections must not be shared across the
    runner's concurrent tasks; the blocking fetch runs in a thread).
    """

    async def classify_task(*, item, **kwargs) -> dict:
        from ticker_news.shared import observability as obs

        data = item["input"] if isinstance(item, dict) else item.input
        article_id = data["article_id"]

        def _fetch() -> tuple[str | None, str | None]:
            conn = connect_eval(dsn)
            try:
                row = conn.execute(
                    "SELECT title, content FROM public.articles WHERE id = %s",
                    (article_id,),
                ).fetchone()
            finally:
                conn.close()
            if row is None:
                raise ValueError(f"article {article_id} not found")
            return row

        title, content = await asyncio.to_thread(_fetch)
        verdict, confirmed = await runner.classify(
            title, content or "", config=obs.chain_config() or None
        )
        label = runner.label_of(verdict)
        return {
            "predicted": label,
            "act": "YES" if runner.is_act(label) else "NO",
            "confidence": getattr(verdict, "confidence", None),
            "reason": verdict.reason or None,
            "confirmed": confirmed,
        }

    return classify_task


def _warn_failed_items(result, requested_ids: list[int]) -> None:
    """Failed items vanish from the result (SDK logs only); make them loud.

    Unlike the pipeline eval, nothing is left dirty in the DB — the task is
    read-only — but a missing item silently skews the run metrics."""
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
            f"to fill them in."
        )


def run_eval(
    variants: tuple[str, ...],
    *,
    mode: str = "two-pass",
    dataset_name: str = DATASET_DEFAULT,
    gt_csv: str | None = None,
    dsn: str | None = None,
    run_name: str | None = None,
    ids: list[int] | None = None,
) -> list[tuple[str, object]]:
    """Run one experiment per variant over the GT dataset.

    With gt_csv, (re)seeds the dataset first (idempotent upsert keyed on
    article id). Returns [(variant, ExperimentResult), ...].
    """
    from ticker_news.classification.variants import make_runner, models_for_mode
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

    if gt_csv:
        gt_rows = load_ground_truth(gt_csv)
        conn = connect_eval(dsn)
        try:
            items = build_items(conn, gt_rows)
        finally:
            conn.close()
        try:
            client.create_dataset(name=dataset_name, description=_DESCRIPTION)
        except Exception:  # noqa: BLE001 - already exists is fine
            pass
        for it in items:
            client.create_dataset_item(
                dataset_name=dataset_name,
                id=it["id"],
                input=it["input"],
                expected_output=it["expected_output"],
                metadata=it["metadata"],
            )

    dataset = client.get_dataset(dataset_name)
    data = list(dataset.items)
    if not data:
        raise SystemExit(
            f"dataset '{dataset_name}' has no items (seed it with --gt-csv)"
        )
    if ids:
        wanted = set(ids)
        data = [it for it in data if (it.input or {}).get("article_id") in wanted]
        if not data:
            raise SystemExit("none of the requested --ids are in the dataset")
    requested_ids = [(it.input or {}).get("article_id") for it in data]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: list[tuple[str, object]] = []
    try:
        for variant in variants:
            runner = make_runner(variant, mode)  # fetches prompts -> versions_seen
            if run_name:
                rn = f"{run_name}-{variant}" if len(variants) > 1 else run_name
            else:
                rn = f"{variant}-{mode}-{stamp}"
            result = client.run_experiment(
                name=f"{EXPERIMENT_PREFIX}-{variant}",
                run_name=rn,
                description=_DESCRIPTION,
                data=data,
                task=make_task(runner, dsn),
                evaluators=[act_accuracy_evaluator, predicted_label_evaluator],
                run_evaluators=[act_metrics_run_evaluator],
                # async task -> max_concurrency gates real parallelism; the
                # shared Gemini rate limiter caps requests per second anyway.
                max_concurrency=8,
                metadata={
                    "variant": variant,
                    "mode": mode,
                    "models": models_for_mode(mode),
                    "prompt_versions": prompts.versions_seen(),
                    "entrypoint": "eval",
                },
            )
            _warn_failed_items(result, requested_ids)
            results.append((variant, result))
    finally:
        obs.flush()
    return results
