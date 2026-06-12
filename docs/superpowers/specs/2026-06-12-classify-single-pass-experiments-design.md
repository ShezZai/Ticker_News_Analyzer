# Single-pass classifier experiments — design

Date: 2026-06-12
Status: approved

## Goal

Replace the two-pass classification eval with two **single-pass** classifier
experiments in Langfuse, scored by deterministic in-repo code evaluators:

| Variant | Prompt (Langfuse-managed) | Dataset | Expected output |
|---|---|---|---|
| `binary` | `classify-binary` | `140-articles-act-no-act` | `{label: "YES" \| "NO"}` (42 YES / 98 NO) |
| `finegrained` | `classify-finegrained` | `140-articles-categories` | `{label: <FinegrainedCategory>}` (16 categories) |

Both datasets are already in Langfuse (verified 2026-06-12): 140 items each,
same article ids, input `{article_id}`. All category labels match
`FINEGRAINED_CATEGORIES` exactly.

Each experiment item performs **exactly one LLM call**; everything else
(label mapping, scoring, time/cost accounting) is deterministic code, so a
full 140-item run should finish in well under a minute.

## Decisions (user-confirmed)

1. **Single-pass only.** Two-pass mode, `VariantRunner`'s confirm chain,
   `MODES`, and `models_for_mode` are deleted. The production classifier
   (`classification/chain.py`) is untouched.
2. **Model is a CLI flag**: `--model lite|flash` → `GEMINI_FLASH_LITE`
   (default) / `GEMINI_FLASH`.
3. **Evaluators are SDK functions in the repo** (passed to `run_experiment`),
   not Langfuse-UI code evaluators. Versioned in git, unit-tested.
4. **Cost from an in-repo price table** (USD per 1M input/output tokens per
   Gemini model), applied to token usage captured in-process. Prices verified
   against Google's published pricing at implementation time.
5. **`ticker-news eval classify` is replaced** — `--gt-csv` seeding and
   `--mode` are dropped. The old `classify-ground-truth` dataset stays in
   Langfuse but is no longer targeted.
6. Finegrained keeps the extra `derived_act_accuracy` run score; per-item
   `latency_s` / `cost_usd` scores are kept alongside the run totals.

## 1. Classifiers — `classification/variants.py`

**Deleted:** `MODES`, `VariantRunner`, `models_for_mode`, all two-pass logic.

**Kept:** `BinaryClassification`, `FinegrainedClassification`, label
constants, `NEWS_SUBTYPES`, `is_act_binary`, `is_act_finegrained`.

**New:**

```python
@dataclass
class Classifier:
    chain: Any            # prompt | structured-output Gemini
    variant: str          # "binary" | "finegrained"
    label_of: Callable    # verdict -> raw predicted label/category
    dataset_label: Callable  # verdict -> dataset label space

    async def classify(self, title, body, config=None) -> verdict
```

- `classify()` makes one `ainvoke` with `run_name=f"classify-{self.variant}"`.
  Inputs truncated as today (title 300 chars, body `MAX_ARTICLE_CHARS`).
- `dataset_label`: binary maps `"real news"` → `"YES"`, anything else →
  `"NO"` (via `is_act_binary`; the prompt's output vocabulary is unchanged).
  Finegrained returns the category as-is.
- `make_classifier(variant, model_name) -> Classifier` factory. Chains are
  built **fresh per run** (no `lru_cache`) so Langfuse prompt edits apply
  without a process restart.
- `_build()` keeps the template-validation guard ({title, body} only,
  fallback on bad remote template) and structured output + retry
  (`with_retry`, 4 attempts) as today.

## 2. Prompt management & linking — `shared/prompts.py`

Prompts stay Langfuse-managed with in-repo fallbacks as source of truth
(`classify-binary`, `classify-finegrained` are already in the
`prompts push` registry; unchanged).

**New:** `get_prompt_entry(name, fallback) -> tuple[str, object | None]`
returning the prompt **text** and the Langfuse **prompt client object** (or
`None` when disabled/unavailable). `get_prompt()` remains and delegates to it.

`_build()` attaches the object to the LangChain template:

```python
ChatPromptTemplate.from_template(text, metadata={"langfuse_prompt": prompt_obj})
```

so generations link to the prompt version in Langfuse (prompt metrics UI).
We keep using the raw `.prompt` text — NOT `get_langchain_prompt()`, which
would mangle the escaped `{{...}}` JSON braces in our f-string-convention
templates. Verify the exact linking mechanism against current Langfuse docs
during implementation; if metadata-on-template doesn't link, fall back to the
documented config-metadata form.

When Langfuse is down/disabled: fallback text, no link, run still works
(eval itself requires Langfuse keys anyway — results live there).

## 3. Eval module — `evals/classify_eval.py` (refactor in place)

**Deleted:** `load_ground_truth`, `build_items`, `--gt-csv` seeding path,
`act_accuracy_evaluator` / `act_metrics_run_evaluator` in their
`expected_output.act` form, mode plumbing.

**Kept/reused:** `connect_eval` (from `pipeline_eval`), `_warn_failed_items`,
trace renaming via `propagate_attributes`, `obs.flush()` in a `finally`.

### Prefetch (speed)

Before `run_experiment`: one query
`SELECT id, title, content FROM public.articles WHERE id = ANY(%s)` for every
article id in the dataset → in-memory `{id: (title, content)}`.
Loud `ValueError` listing ids that are missing or have empty content.
The task then performs **zero DB work** — the old per-item fetch over the
tunneled shared DB often dwarfed the LLM call.

### Task

```python
async def task(*, item, **kw):
    t0 = time.monotonic()
    # UsageMetadataCallbackHandler appended to obs.chain_config() callbacks
    verdict = await classifier.classify(title, body, config=cfg)
    return {
        "predicted": <raw label>, "label": <dataset-space label>,
        "reason": ..., "confidence": ...,          # binary only
        "latency_s": time.monotonic() - t0,
        "input_tokens": ..., "output_tokens": ..., "model": <model name>,
    }
```

Trace + root span renamed to `classify-{variant}:article-{id}` (same
`propagate_attributes` + `update_current_span` pattern as today).
Token usage comes from LangChain's `UsageMetadataCallbackHandler` (structured
output hides the raw `AIMessage`, the callback is the clean capture point).
A fresh handler per invocation (not thread-safe to share — same rule as the
Langfuse `CallbackHandler`).

### Experiment spec table

```python
@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    experiment_name: str        # "classify-binary" / "classify-finegrained"
    evaluators: list
    run_evaluators: list

EXPERIMENTS: dict[str, ExperimentSpec] = {"binary": ..., "finegrained": ...}
```

One generic `run_eval(variant, model, ...)` drives any spec:
`run_experiment(name=spec.experiment_name, run_name=..., data=...,
task=..., evaluators=..., run_evaluators=..., max_concurrency=N,
metadata={"variant", "model", "prompt_versions", "entrypoint": "eval"})`.
Default run name: `{variant}-{model}-{YYYYmmdd-HHMMSS}`.

## 4. Evaluators (all deterministic)

### Item-level (both variants)

| Score | Type | Value |
|---|---|---|
| `label_accuracy` | numeric | 1.0 if `output.label == expected.label` else 0.0; 0.0 with comment when output is missing |
| `predicted_label` | categorical | raw predicted label (filterable in UI) |
| `latency_s` | numeric | task wall-clock |
| `cost_usd` | numeric | tokens × price table |

### Run-level (both variants)

| Score | Value |
|---|---|
| `label_accuracy_avg` | correct / total (errored items count as wrong) |
| `total_time_s` | wall clock around the whole `run_experiment` call (closure over a start timestamp) |
| `avg_time_per_item_s` | mean of item `latency_s` |
| `total_cost_usd` | sum of item costs |
| `total_tokens` | sum input+output (comment breaks down in/out) |

### Binary-only (run-level)

Confusion metrics on the YES class, adapted from the existing
`act_metrics_run_evaluator` (the GT is 42/98 imbalanced): `act_precision`,
`act_recall`, `act_f1`, with the same `_skip` categorical scores for empty
denominators. An errored item (output `None`) always counts as wrong —
FN on YES items, FP on NO items.

### Finegrained-only (run-level)

`derived_act_accuracy`: map expected and predicted categories through
`NEWS_SUBTYPES` → YES/NO; fraction agreeing. Shows whether
miscategorizations cross the ACT boundary.

### Price table

`GEMINI_PRICES_USD_PER_1M = {model: (input, output)}` for
`gemini-2.5-flash-lite` and `gemini-2.5-flash`, values checked against
Google's current published pricing during implementation. Unknown model in
usage data ⇒ cost scores emitted as `cost_usd_skip` categorical rather than
a silent 0.

## 5. CLI — `ticker-news eval classify`

```
--variant binary|finegrained|both   default: both (runs two experiments)
--model lite|flash                  default: lite
--ids 1,2,3                         subset of dataset items
--run-name NAME                     default: {variant}-{model}-{stamp}
--dsn DSN                           DB override (shared/tunneled DB)
--concurrency N                     default: 16 (max_concurrency)
--dataset NAME                      dataset override (testing/escape hatch);
                                    only valid with a single --variant, errors
                                    when combined with both
```

Parallelism: async task + `max_concurrency=16` default; the process-global
Gemini rate limiter (8 rps) remains the quota guard → ~18 s dispatch floor
for 140 items. Prints `result.format()` plus a one-line summary per run.
CLAUDE.md command-table row updated.

## 6. Error handling

- Langfuse keys absent → `SystemExit` (results live in Langfuse). Missing
  `GOOGLE_API_KEY` → `SystemExit`.
- Dataset empty / `--ids` matching nothing → `SystemExit`.
- Prefetch: missing articles or empty content → loud `ValueError` before any
  LLM call.
- Items that error mid-run vanish from SDK results → `_warn_failed_items`
  prints the ids and a ready-to-paste `--ids` re-run line (read-only task, so
  nothing is left dirty).
- LLM transient failures: `with_retry` (4 attempts) inside the chain.

## 7. Testing (offline, no network)

- `tests/classification/test_variants.py` (rework): factory validation
  (unknown variant/model), single-pass `classify()` with a stub chain
  (asserts exactly one invoke, run_name, truncation), `dataset_label`
  mapping both variants.
- `tests/evals/test_classify_eval.py` (rework): prefetch validation errors;
  every evaluator against synthetic outputs/`item_results` — accuracy,
  confusion metrics incl. errored items, derived-act, time/cost aggregation,
  price-table math, unknown-model skip; spec table sanity (datasets,
  experiment names); CLI wiring smoke (Typer runner with `run_eval` mocked).
- Existing two-pass tests are deleted with the code they cover.

## Out of scope

- Production pipeline classifier (`chain.py`, `classify` stage) — untouched.
- Promoting a winning prompt to production — stays a human decision.
- Macro/per-class F1 for finegrained (add later if needed).
- The old `classify-ground-truth` dataset and its seeding path.
