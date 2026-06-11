"""Classification prompt-variant eval against the hand-labeled ground truth.

Read-only: loads article text from the DB, runs the binary and/or
fine-grained variant chains, and scores ACT/DON'T-ACT agreement with the
ground-truth labels as Langfuse experiments on a shared dataset. Never
writes to pipeline tables (the production `category` column is untouched).

Design: docs/superpowers/specs/2026-06-12-classify-eval-design.md
"""

from __future__ import annotations

import csv
from pathlib import Path

import psycopg
from langfuse import Evaluation

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
