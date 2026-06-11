# Classification Eval + Fast Repeatable E2E — Design

**Date:** 2026-06-12
**Status:** Approved

## Goal

Two things, both aimed at fast, repeatable, Langfuse-visible evaluation:

1. **`ticker-news eval classify`** — a read-only Langfuse experiment that
   compares two classification prompt variants against a hand-labeled
   ground-truth set of 140 articles:
   - **binary**: `real news` / `none news` (ACT / DON'T ACT)
   - **fine-grained**: 16-category taxonomy whose 7 NEWS subtypes map to ACT
   Both variants are scored on the same binary ground truth, on the same
   Langfuse dataset, so runs compare side-by-side in the UI and prompts can
   be tuned iteratively against the labels.
2. **`--skip-stages` on `ticker-news eval pipeline`** — repeat E2E runs on a
   stable article set (e.g. the 100 "peaceful days" articles) without
   re-embedding - the article was already scrapped and embbedded we want to quickly test the following steps in the flow (classification tagging insight extraction and verdict generation)

## Context

- Ground truth: `C:\Agents\final-project\classify\ground-truth-class.csv` —
  columns `article id, header, Act_GT` (YES/NO), 140 rows, **42 YES / 98 NO**
  (imbalanced — accuracy alone would flatter an always-NO classifier).
- The 100-article set `C:\Agents\final-project\100_articles_from_peacfull_days.csv`
  (one article id per line) is **disjoint** from the GT ids (verified: 0
  overlap), so it cannot be accuracy-scored. Decision: it stays reserved for
  the E2E pipeline eval; the classification comparison uses GT-140 only.
- Both id sets fully resolve in the shared DB (`sharedproject`, cloudflared
  tunnel at `localhost:15432`): 140/140 and 100/100 present, `status='ok'`,
  non-empty content, titles match the GT csv headers exactly.
- Prompt variant sources (already drafted by the user, to be adapted into
  in-repo templates):
  - `C:\Agents\final-project\classify\classify_news_binary_prompt.txt`
  - `C:\Agents\final-project\classify\classify_finegrained_prompt.txt`
- Production classifier (`classification/chain.py`) is two-pass:
  flash-lite labels everything, flash re-runs only items lite called
  `real news`; a failed confirmation keeps the lite verdict.
- Langfuse SDK v4 experiment runner; sync tasks run serially in 4.7.1, so
  the classify task is **async** to get real concurrency (each task is only
  1–2 Gemini calls).

## Decisions (made with user)

1. **Scoring set:** GT-140 only. The unlabeled 100-set is not part of the
   classify eval.
2. **Scope:** both the new classify eval AND the `--skip-stages` speedup for
   `eval pipeline`.
3. **Prompt management:** Langfuse prompts with in-repo fallbacks, exactly
   like the existing registry — new entries `classify-binary` and
   `classify-finegrained`, pushed by `ticker-news prompts push`, fetched with
   `get_prompt()`; each experiment run records the prompt versions it used.
4. **Model config:** CLI-selectable `--mode lite|flash|two-pass`, default
   `two-pass` (production mirror). Run names record variant + mode.

## Components

### 1. `src/ticker_news/classification/variants.py` (new)

Experimental classification variants live next to the production classifier
(they are candidate replacements for it), consumed by the eval:

- `BINARY_PROMPT_TEMPLATE` / `FINEGRAINED_PROMPT_TEMPLATE` — in-repo
  fallback templates adapted from the user's txt drafts; placeholders
  `{title}` / `{body}` (same contract as the production prompt).
- `BinaryClassification` (pydantic): `label: Literal["real news", "none news"]`,
  optional `confidence: float`, optional `reason: str`.
- `FinegrainedClassification` (pydantic): `category: Literal[<16 categories>]`,
  optional `reason: str`.
- `NEWS_SUBTYPES` — the 7 fine-grained categories that map to ACT/YES:
  `earnings-reporting`, `dividend-reporting`, `merger/investment/funding`,
  `legal-event`, `MACRO-investment`, `news-event`, `news-report`.
- `build_binary_classifier(model_name)` / `build_finegrained_classifier(model_name)`
  — `gemini_chat` + `with_structured_output` + retry, prompt via
  `get_prompt("classify-binary"|"classify-finegrained", fallback)`, with the
  same invalid-template fallback guard as `build_classifier`.
- `classify_binary(title, body, *, mode, config)` /
  `classify_finegrained(title, body, *, mode, config)` — async; implement the
  three modes. `two-pass` mirrors production semantics: lite labels all,
  flash re-runs only when lite said ACT (binary: `real news`; fine-grained:
  category in `NEWS_SUBTYPES`); flash failure keeps the lite verdict.
  Chains are built once per eval run (not lru_cached module-globals — the
  eval must pick up Langfuse prompt edits without a process restart).

### 2. `shared/prompts.py` registry additions

`registry()` gains `"classify-binary"` and `"classify-finegrained"` entries
importing from `classification/variants.py`. `ticker-news prompts push`
then publishes them with the `production` label; tuning happens in the
Langfuse UI (or in-repo, re-pushed).

### 3. `src/ticker_news/evals/classify_eval.py` (new)

- `load_ground_truth(csv_path) -> list[GtRow]` — utf-8-sig csv reader;
  validates integer ids, unique ids, `Act_GT in {YES, NO}` (trim/upper);
  raises with the offending row on any violation.
- `build_items(conn, gt_rows)` — verifies every id exists with `status='ok'`
  and non-empty content (loud failure listing missing/bad ids). Items:
  `input={"article_id", "title"}`, `expected_output={"act": "YES"|"NO"}`,
  `metadata={"gt_header": <csv header>}`. Bodies are NOT stored in the
  dataset — the DB row is the single source of truth (same convention as
  the pipeline eval).
- Dataset: default name `classify-ground-truth`; items upserted
  idempotently with `id=f"article-{id}"` whenever `--gt-csv` is passed.
  Subsequent runs can omit `--gt-csv` and run over the existing dataset.
- Task (`make_task(variant, mode, dsn)`) — **async**, read-only: fresh
  psycopg connection per invocation, `SELECT title, content` by article id,
  truncate body to the production `MAX_ARTICLE_CHARS` (6000), call the
  variant chain, close the connection. Returns
  `{"predicted": <label-or-category>, "act": "YES"|"NO",
    "confidence": float|None, "reason": str|None, "confirmed": bool}`.
  Never writes to `articles` (the production `category` column is untouched).
- Item evaluators:
  - `act_accuracy`: 1.0 / 0.0 — `output["act"]` vs `expected_output["act"]`;
    comment shows `predicted=<label> gt=<YES|NO>`.
  - `predicted_label`: categorical score (the raw predicted label/category)
    so misclassifications are filterable in the UI.
- Run evaluators (computed over item results): `act_accuracy_avg`,
  `act_precision`, `act_recall`, `act_f1` — precision/recall/F1 for the YES
  class, each with a comment carrying the confusion counts (TP/FP/FN/TN).
- `run_eval(variants, *, mode, dataset_name, gt_csv, dsn, run_name, ids)` —
  per variant, one `dataset.run_experiment` call:
  - experiment name `classify-binary` / `classify-finegrained`;
  - auto run name `{variant}-{mode}-{YYYYMMDD-HHMMSS}` unless `--run-name`
    given (suffixed `-binary`/`-finegrained` when running both);
  - metadata: `{"variant", "mode", "models": {...}, "prompt_versions":
    prompts.versions_seen(), "entrypoint": "eval"}`;
  - `max_concurrency=8` (async task, 1–2 light Gemini calls per item);
  - `--ids` filters the dataset items locally (fast iteration on a
    misclassified subset; scores still land on the same dataset items).
  - reuses `_warn_failed_items`-style reporting for items that errored.
- Required keys: `LANGFUSE_*` and `GOOGLE_API_KEY` (hard error if missing).
  `MASSIVE_API_KEY` / `OPENAI_API_KEY` are NOT required — no prices, no
  embeddings.

### 4. CLI: `ticker-news eval classify`

```
ticker-news eval classify
    [--variant both|binary|finegrained]   # default both
    [--mode two-pass|lite|flash]          # default two-pass
    [--gt-csv PATH]                       # seed/refresh the dataset from the GT csv
    [--dataset classify-ground-truth]     # Langfuse dataset name
    [--ids 168,221,...]                   # subset for quick prompt iteration
    [--dsn postgresql://...]              # shared-DB DSN (default DATABASE_URL)
    [--run-name NAME]
```

First run seeds the dataset: `--gt-csv <path>`. After that the csv flag is
optional. Errors clearly if the dataset is empty and no csv was given.
Prints each experiment's `result.format()` (with the existing cp1252
fallback).

### 5. `eval pipeline --skip-stages` (changed)

- New option `--skip-stages embed,insights` — comma list, allowed values
  `embed`, `classify`, `tag`, `insights` (sentiment always re-runs; it is
  what the experiment scores). Invalid names → `BadParameter`.
- `reset_article(conn, article_id, keep=frozenset())` — outputs of kept
  stages are not cleared:
  - `embed` → keep `embedding`;
  - `classify` → keep `category`, `category_reason`;
  - `tag` → keep `primary_ticker`, `primary_segment`, `more_tickers`,
    `more_segments`;
  - `insights` → keep `article_insights` rows and `insights_extracted_at`.
  `article_sentiment` is always deleted. The idempotent stage adapters then
  no-op naturally on kept stages (cheap SELECT, no LLM/API calls); the task
  code does not change.
- Run metadata gains `"skipped_stages"` so runs that reused stage outputs
  are distinguishable in Langfuse.
- New option `--ids-file PATH` (one integer per line, e.g. the 100-article
  csv) as an alternative to `--ids`; mutually additive (union) with `--ids`.
- Default behavior (no flags) is unchanged: full reset, full re-run.

Repeatable fast E2E on the stable set then is:

```
ticker-news eval pipeline --ids-file C:\...\100_articles_from_peacfull_days.csv `
    --dataset pipeline-100-peaceful --skip-stages embed --dsn <shared>
```

(first run seeds the dataset; later runs can use `--dataset` alone).

## Langfuse UI layout

- Dataset `classify-ground-truth` (140 items, expected_output = GT label) —
  the Runs tab compares `classify-binary` vs `classify-finegrained` runs
  side by side on `act_accuracy` / `act_precision` / `act_recall` / `act_f1`.
- Per-item view shows predicted vs expected, the model's reason, and the
  full trace of the underlying Gemini call(s).
- Prompt versions: `classify-binary` / `classify-finegrained` prompts are
  versioned in Langfuse; each run's metadata records the exact versions
  used, so a prompt edit → re-run → score delta is fully traceable.
- E2E runs on the 100-set live on their own dataset (`pipeline-100-peaceful`)
  with the existing directional-agreement scores.

## Error handling

- GT csv malformed / unknown ids / empty content → loud failure before any
  experiment starts (nothing partial lands in Langfuse).
- Per-item LLM failure → item errors in Langfuse, others continue; failed
  items reported at the end (same pattern as the pipeline eval). No DB
  cleanup needed — the classify task never writes.
- Langfuse keys missing → hard SystemExit (an eval without Langfuse is
  pointless). Tunnel down → connect fails fast before any run starts.

## Testing

Offline unit tests (`tests/evals/test_classify_eval.py`, no markers):

- `load_ground_truth`: utf-8-sig BOM, YES/NO normalization, duplicate-id and
  bad-value rejection.
- ACT mapping: binary label → YES/NO; each of the 16 fine-grained categories
  → its NEWS_SUBTYPES side.
- Run evaluators: precision/recall/F1 on a synthetic confusion matrix,
  including the zero-division edges (no YES predictions, no YES items).
- Two-pass semantics with stub chains: confirm runs only on lite-ACT;
  confirm failure keeps the lite verdict.
- `reset_article` keep-semantics: for each `keep` subset, assert the SQL
  touches exactly the expected columns/tables (mocked conn, same style as
  the existing pipeline-eval tests).
- CLI: `--skip-stages` validation; `--ids-file` parsing.

## Out of scope (YAGNI)

- Fine-grained-vs-fine-grained ground truth (the GT csv is binary only; a
  per-category labeled set can be added to the same dataset later as a
  second expected_output field).
- Auto-promotion of a winning prompt to the production `classify-article`
  prompt — promotion stays a human decision.
- Scoring the unlabeled 100-set for classification (no labels; it stays the
  E2E set).
- Confidence-threshold sweeps / calibration curves on the binary
  classifier's confidence output (recorded per item; analysis can happen in
  the UI or a notebook later).
- Disk-caching Massive price bars across eval-pipeline runs (in-process
  lru_cache already dedupes within a run).
