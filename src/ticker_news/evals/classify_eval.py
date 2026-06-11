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
