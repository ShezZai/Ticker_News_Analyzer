# Single-Pass Classifier Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-pass classification eval with two single-pass classifier experiments (binary → `140-articles-act-no-act`, finegrained → `140-articles-categories`) scored by deterministic in-repo evaluators including time and cost metrics.

**Architecture:** `classification/variants.py` slims to a `Classifier` dataclass (one chain, one LLM call) built by `make_classifier(variant, model)`; prompts stay Langfuse-managed and are now *linked* to generations via `metadata={"langfuse_prompt": ...}`. `evals/classify_eval.py` is reworked in place: a one-query article prefetch, a pure-LLM async task, a declarative `EXPERIMENTS` spec table, and SDK evaluator functions (item: accuracy/label/latency/cost; run: totals, binary confusion, finegrained derived-ACT). Spec: `docs/superpowers/specs/2026-06-12-classify-single-pass-experiments-design.md`.

**Tech Stack:** Python 3.11+, LangChain (`langchain_core` 1.4.6), `langfuse` 4.7.1 (`run_experiment`), `langchain-google-genai` (Gemini 2.5 flash / flash-lite), psycopg, Typer, pytest.

**Verified facts (do not re-derive):**
- Datasets exist in Langfuse: `140-articles-act-no-act` (140 items, expected `{"label": "YES"|"NO"}`, 42/98) and `140-articles-categories` (140 items, expected `{"label": "<category>"}`, all 16 labels match `FINEGRAINED_CATEGORIES`). Item input is `{"article_id": <int>}`. Item ids look like `140-articles-act-no-act--10190`.
- `from langchain_core.callbacks import UsageMetadataCallbackHandler` works; after a chain run, `handler.usage_metadata` is `{model_name: {"input_tokens": int, "output_tokens": int, "total_tokens": int}}`.
- Langfuse prompt↔generation linking for LangChain (per current docs): set `prompt_template.metadata = {"langfuse_prompt": <prompt client object>}` on the `ChatPromptTemplate`. Keep using the raw `.prompt` text — `get_langchain_prompt()` would mangle our escaped `{{...}}` JSON braces.
- Gemini paid-tier prices (USD per 1M tokens, verified 2026-06-13 at ai.google.dev/gemini-api/docs/pricing): flash-lite 0.10 in / 0.40 out; flash 0.30 in / 2.50 out.

**Conventions:**
- Run tests with the venv python: `.venv/Scripts/python.exe -m pytest <path> -v` (Windows). Offline suite: `-m "not db and not integration"`.
- Commit messages: clean, no Co-Authored-By, no Claude/Anthropic attribution (user's global rule).
- `tests/conftest.py` already neutralizes `.env` and `LANGFUSE_*` env vars — tests never need network.

---

### Task 0: Land the pre-existing uncommitted work

The working tree contains coherent, unrelated-to-this-plan observability improvements from a prior session (invoke-time `run_name`s, `structured-output` span naming, Gemini runnable naming, insights/pipeline_eval tweaks). They touch the same files this plan refactors, so they must land first as their own commit.

**Files (already modified, just commit):**
- `src/ticker_news/classification/chain.py`, `src/ticker_news/classification/variants.py`, `src/ticker_news/enrichment/insights.py`, `src/ticker_news/evals/classify_eval.py`, `src/ticker_news/evals/pipeline_eval.py`, `src/ticker_news/shared/llm.py`, `tests/classification/test_chain.py`, `tests/evals/test_classify_eval.py`

- [ ] **Step 0.1: Run the offline suite to confirm the tree is green**

Run: `.venv/Scripts/python.exe -m pytest -m "not db and not integration" -q`
Expected: all pass. If anything fails, STOP and report — do not commit a red tree.

- [ ] **Step 0.2: Commit exactly those eight files (nothing from `??` untracked)**

```bash
git add src/ticker_news/classification/chain.py src/ticker_news/classification/variants.py src/ticker_news/enrichment/insights.py src/ticker_news/evals/classify_eval.py src/ticker_news/evals/pipeline_eval.py src/ticker_news/shared/llm.py tests/classification/test_chain.py tests/evals/test_classify_eval.py
git commit -m "feat: named observation spans for classifier passes and structured output"
```

---

### Task 1: `get_prompt_entry` in shared/prompts.py

Returns `(text, prompt_object | None)` so chains can link generations to the Langfuse prompt version. `get_prompt` becomes a thin wrapper.

**Files:**
- Modify: `src/ticker_news/shared/prompts.py`
- Test: `tests/shared/test_prompts.py`

- [ ] **Step 1.1: Write the failing tests** — append to `tests/shared/test_prompts.py`:

```python
def test_get_prompt_entry_returns_text_and_object(monkeypatch):
    class FakePrompt:
        prompt = "REMOTE {title} {body}"
        version = 3

    fake = FakePrompt()

    class FakeClient:
        def get_prompt(self, name, label=None):
            assert label == prompts.PROMPT_LABEL
            return fake

    monkeypatch.setattr(prompts, "client", lambda: FakeClient())
    text, obj = prompts.get_prompt_entry("classify-binary", "fb")
    assert text == "REMOTE {title} {body}"
    assert obj is fake
    assert prompts.versions_seen()["classify-binary"] == 3


def test_get_prompt_entry_fallback_when_disabled(monkeypatch):
    _disable(monkeypatch)
    text, obj = prompts.get_prompt_entry("classify-binary", "fb {x}")
    assert text == "fb {x}"
    assert obj is None


def test_get_prompt_entry_fallback_on_fetch_failure(monkeypatch):
    class FakeClient:
        def get_prompt(self, name, label=None):
            raise RuntimeError("network down")

    monkeypatch.setattr(prompts, "client", lambda: FakeClient())
    text, obj = prompts.get_prompt_entry("classify-binary", "fb")
    assert text == "fb"
    assert obj is None
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/shared/test_prompts.py -v`
Expected: 3 new tests FAIL with `AttributeError: ... no attribute 'get_prompt_entry'`; existing tests PASS.

- [ ] **Step 1.3: Implement** — in `src/ticker_news/shared/prompts.py`, replace the body of `get_prompt` with a delegation and add `get_prompt_entry` above it:

```python
def get_prompt_entry(name: str, fallback: str) -> tuple[str, object | None]:
    """Return (prompt text, Langfuse prompt object) for label=production.

    The object is None when Langfuse is disabled or the fetch fails — callers
    use it to link generations to the prompt version (metadata
    {"langfuse_prompt": obj} on a LangChain prompt template); text falls back
    to the in-repo template so the service stays fully operational.
    """
    c = client()
    if c is None:
        return fallback, None
    try:
        p = c.get_prompt(name, label=PROMPT_LABEL)
        _seen_versions[name] = p.version
        return p.prompt, p
    except Exception as exc:
        logger.warning("langfuse prompt %r unavailable (%r); using fallback", name, exc)
        return fallback, None


def get_prompt(name: str, fallback: str) -> str:
    """Return Langfuse prompt text (label=production) or the in-repo fallback."""
    return get_prompt_entry(name, fallback)[0]
```

(Keep the original `get_prompt` docstring content if preferred, but the two functions must share the single fetch path above — no duplicated try/except.)

- [ ] **Step 1.4: Run the file's tests**

Run: `.venv/Scripts/python.exe -m pytest tests/shared/test_prompts.py -v`
Expected: ALL pass (old `get_prompt` tests still green via delegation).

- [ ] **Step 1.5: Commit**

```bash
git add src/ticker_news/shared/prompts.py tests/shared/test_prompts.py
git commit -m "feat: get_prompt_entry returns the Langfuse prompt object for generation linking"
```

---

### Task 2: Single-pass `Classifier` in classification/variants.py

Strip two-pass machinery; add `Classifier` + `make_classifier`; link prompts in `_build`.

**Files:**
- Modify: `src/ticker_news/classification/variants.py`
- Test: `tests/classification/test_variants.py` (rework — delete `TestVariantRunner`, `TestMakeRunner`, the old `StubChain` stays)

- [ ] **Step 2.1: Rework the test file** — in `tests/classification/test_variants.py`, keep `TestSchemas`, `TestActMapping`, `StubChain`, and `_run` unchanged; DELETE `TestVariantRunner` and `TestMakeRunner`; add:

```python
class TestClassifier:
    def _binary(self, *verdicts):
        from ticker_news.classification.variants import Classifier, is_act_binary

        chain = StubChain(*verdicts)
        clf = Classifier(
            chain=chain, variant="binary", model="gemini-2.5-flash-lite",
            label_of=lambda v: v.label,
            dataset_label_of=lambda v: "YES" if is_act_binary(v.label) else "NO",
        )
        return clf, chain

    def test_single_llm_call(self):
        clf, chain = self._binary(BinaryClassification(label="real news"))
        verdict = _run(clf.classify("T", "B"))
        assert verdict.label == "real news"
        assert len(chain.calls) == 1

    def test_inputs_truncated_like_production(self):
        from ticker_news.classification.chain import MAX_ARTICLE_CHARS

        clf, chain = self._binary(BinaryClassification(label="none news"))
        _run(clf.classify("  t  " * 200, "x" * 10_000))
        sent = chain.calls[0]
        assert len(sent["title"]) <= 300
        assert len(sent["body"]) == MAX_ARTICLE_CHARS

    def test_run_name_carries_variant(self):
        from ticker_news.classification.variants import Classifier

        seen = {}

        class CfgChain(StubChain):
            async def ainvoke(self, inputs, config=None):
                seen.update(config or {})
                return await super().ainvoke(inputs, config)

        chain = CfgChain(FinegrainedClassification(category="other"))
        clf = Classifier(chain=chain, variant="finegrained", model="m",
                         label_of=lambda v: v.category,
                         dataset_label_of=lambda v: v.category)
        _run(clf.classify("T", "B"))
        assert seen["run_name"] == "classify-finegrained"


class TestDatasetLabelMapping:
    def test_binary_maps_to_yes_no(self):
        from ticker_news.classification.variants import make_classifier

        clf = _make_offline(make_classifier, "binary")
        assert clf.dataset_label_of(BinaryClassification(label="real news")) == "YES"
        assert clf.dataset_label_of(BinaryClassification(label="none news")) == "NO"

    def test_finegrained_passes_category_through(self):
        from ticker_news.classification.variants import make_classifier

        clf = _make_offline(make_classifier, "finegrained")
        v = FinegrainedClassification(category="legal-call")
        assert clf.dataset_label_of(v) == "legal-call"
        assert clf.label_of(v) == "legal-call"


def _make_offline(make_classifier, variant):
    """make_classifier without a Gemini client: stub the chain builder."""
    from unittest.mock import patch

    target = ("ticker_news.classification.variants."
              f"build_{'binary' if variant == 'binary' else 'finegrained'}_classifier")
    with patch(target, return_value=StubChain()):
        return make_classifier(variant, "gemini-2.5-flash-lite")


class TestMakeClassifier:
    def test_rejects_unknown_variant(self):
        from ticker_news.classification.variants import make_classifier

        with pytest.raises(ValueError, match="variant"):
            make_classifier("ternary", "gemini-2.5-flash-lite")

    def test_records_variant_and_model(self):
        from ticker_news.classification.variants import make_classifier

        clf = _make_offline(make_classifier, "binary")
        assert clf.variant == "binary"
        assert clf.model == "gemini-2.5-flash-lite"

    def test_model_choices_map_cli_names(self):
        from ticker_news.classification.variants import MODEL_CHOICES
        from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE

        assert MODEL_CHOICES == {"lite": GEMINI_FLASH_LITE, "flash": GEMINI_FLASH}


class _FakeLLM:
    """Stub for gemini_chat: each builder step returns self until with_retry,
    which must return a real Runnable so `prompt | structured` composes."""

    def with_structured_output(self, schema):
        return self

    def with_config(self, **kw):
        return self

    def with_retry(self, **kw):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda x: x)


class TestBuildLinksPrompt:
    def _patch(self, monkeypatch, prompt_text, prompt_obj):
        from ticker_news.shared import prompts

        monkeypatch.setattr(
            prompts, "get_prompt_entry", lambda name, fb: (prompt_text, prompt_obj)
        )
        monkeypatch.setattr(
            "ticker_news.shared.llm.gemini_chat", lambda m, timeout_s: _FakeLLM()
        )

    def test_build_attaches_langfuse_prompt_metadata(self, monkeypatch):
        from ticker_news.classification import variants

        class FakePromptObj:
            prompt = variants.BINARY_PROMPT_TEMPLATE
            version = 9

        fake = FakePromptObj()
        self._patch(monkeypatch, fake.prompt, fake)
        chain = variants.build_binary_classifier("gemini-2.5-flash-lite")
        # chain is prompt | llm; the first runnable is the ChatPromptTemplate
        assert chain.first.metadata == {"langfuse_prompt": fake}

    def test_bad_remote_template_falls_back_without_link(self, monkeypatch):
        from ticker_news.classification import variants

        class FakePromptObj:
            prompt = "broken {only_title}"
            version = 9

        fake = FakePromptObj()
        self._patch(monkeypatch, fake.prompt, fake)
        chain = variants.build_binary_classifier("gemini-2.5-flash-lite")
        assert set(chain.first.input_variables) == {"title", "body"}
        assert not (chain.first.metadata or {}).get("langfuse_prompt")
```

NOTE for the implementer: `_build` imports `get_prompt_entry` via `from ticker_news.shared.prompts import get_prompt_entry` *inside the function* (existing pattern). For the monkeypatch above to take effect, change `_build` to import the **module** and call `prompts.get_prompt_entry(...)` (see Step 2.3) — patching the module attribute then works.

- [ ] **Step 2.2: Run to verify new tests fail**

Run: `.venv/Scripts/python.exe -m pytest tests/classification/test_variants.py -v`
Expected: `TestSchemas`/`TestActMapping` PASS; everything new FAILS with ImportError/AttributeError (`Classifier`, `make_classifier`, `MODEL_CHOICES` missing).

- [ ] **Step 2.3: Implement in `src/ticker_news/classification/variants.py`**

Delete: `MODES`, `VariantRunner`, `make_runner`, `models_for_mode`. Keep: schemas, constants, `is_act_*`, `build_binary_classifier`, `build_finegrained_classifier`, `GEMINI_TIMEOUT_S`, `RETRIES`. Update the module docstring (single-pass eval classifiers; spec path). Replace `_build` and add below the builders:

```python
def _build(model_name: str, prompt_name: str, fallback: str, schema: type):
    """prompt | structured-output Gemini. The Langfuse prompt object (when
    available) is attached as template metadata so generations link to the
    prompt version in Langfuse. NOT cached — the eval rebuilds per run so
    Langfuse prompt edits apply without a process restart."""
    from ticker_news.shared import llm as shared_llm
    from ticker_news.shared import prompts as shared_prompts

    llm = shared_llm.gemini_chat(model_name, timeout_s=GEMINI_TIMEOUT_S)
    structured = (
        llm.with_structured_output(schema)
        .with_config(run_name="structured-output")
        .with_retry(stop_after_attempt=RETRIES, wait_exponential_jitter=True)
    )
    template, prompt_obj = shared_prompts.get_prompt_entry(prompt_name, fallback)
    try:
        prompt = ChatPromptTemplate.from_template(template)
        if set(prompt.input_variables) != {"title", "body"}:
            raise ValueError(f"unexpected variables: {prompt.input_variables}")
    except Exception as exc:
        logger.warning("%s prompt invalid (%r); using in-repo fallback", prompt_name, exc)
        prompt = ChatPromptTemplate.from_template(fallback)
        prompt_obj = None
    if prompt_obj is not None:
        prompt.metadata = {"langfuse_prompt": prompt_obj}
    return prompt | structured


# CLI model names -> Gemini model ids (single source for cli.py and the eval).
def _model_choices() -> dict[str, str]:
    from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE

    return {"lite": GEMINI_FLASH_LITE, "flash": GEMINI_FLASH}


MODEL_CHOICES = _model_choices()


@dataclass
class Classifier:
    """Single-pass classifier: one chain, exactly one LLM call per article.

    label_of: verdict -> the raw predicted label/category.
    dataset_label_of: verdict -> the dataset's expected-output label space
    (binary collapses to YES/NO; finegrained is the category itself).
    """

    chain: Any
    variant: str
    model: str
    label_of: Callable[[Any], str]
    dataset_label_of: Callable[[Any], str]

    async def classify(self, title: Optional[str], body: str, config=None) -> Any:
        from ticker_news.classification.chain import MAX_ARTICLE_CHARS

        inputs = {
            "title": (title or "").strip()[:300],
            "body": (body or "")[:MAX_ARTICLE_CHARS],
        }
        cfg = {**(config or {}), "run_name": f"classify-{self.variant}"}
        return await self.chain.ainvoke(inputs, config=cfg)


def make_classifier(variant: str, model_name: str) -> Classifier:
    """Build the single-pass classifier for a variant (fresh — no lru_cache)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of {VARIANTS})")
    if variant == "binary":
        chain = build_binary_classifier(model_name)
        label_of = lambda v: v.label  # noqa: E731
        dataset_label_of = lambda v: "YES" if is_act_binary(v.label) else "NO"  # noqa: E731
    else:
        chain = build_finegrained_classifier(model_name)
        label_of = lambda v: v.category  # noqa: E731
        dataset_label_of = label_of
    return Classifier(chain=chain, variant=variant, model=model_name,
                      label_of=label_of, dataset_label_of=dataset_label_of)
```

Also remove the now-unused `Tuple` import if nothing else uses it (check with the linter/tests). NOTE: `MODEL_CHOICES` calls `_model_choices()` at import time — `shared.llm` only reads settings lazily inside `gemini_chat`, so this is safe offline.

- [ ] **Step 2.4: Run the classification tests**

Run: `.venv/Scripts/python.exe -m pytest tests/classification/ -v`
Expected: ALL pass.

- [ ] **Step 2.5: Confirm nothing else imports the deleted names**

Run: `grep -rn "make_runner\|VariantRunner\|models_for_mode\|MODES" src/ tests/`
Expected: hits ONLY in `src/ticker_news/cli.py` (fixed in Task 6) and `src/ticker_news/evals/classify_eval.py` + `tests/evals/test_classify_eval.py` (reworked in Tasks 3–5). If anything else appears, fix it now.

- [ ] **Step 2.6: Commit**

```bash
git add src/ticker_news/classification/variants.py tests/classification/test_variants.py
git commit -m "refactor: single-pass Classifier with Langfuse prompt linking replaces two-pass VariantRunner"
```

(`tests/evals/test_classify_eval.py` and the CLI are temporarily broken — they import deleted names; they are reworked in the very next tasks on this branch.)

---

### Task 3: Price table + item-level evaluators in evals/classify_eval.py

Pure functions first. The whole file is rewritten across Tasks 3–5; in this task replace the module docstring, imports, and the old `load_ground_truth` / `build_items` / `act_accuracy_evaluator` / `predicted_label_evaluator` / `act_metrics_run_evaluator` (delete them) with the pieces below, and rewrite `tests/evals/test_classify_eval.py` from scratch.

**Files:**
- Modify: `src/ticker_news/evals/classify_eval.py`
- Test: `tests/evals/test_classify_eval.py` (full rewrite)

- [ ] **Step 3.1: Replace `tests/evals/test_classify_eval.py` entirely with:**

```python
"""Offline unit tests for the single-pass classification experiments. No DB, no network."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ticker_news.classification.variants import (
    BinaryClassification,
    Classifier,
    is_act_binary,
)
from ticker_news.evals import classify_eval
from ticker_news.evals.classify_eval import (
    EXPERIMENTS,
    GEMINI_PRICES_USD_PER_1M,
    cost_evaluator,
    item_cost_usd,
    label_accuracy_evaluator,
    latency_evaluator,
    predicted_label_evaluator,
)


class TestPriceTable:
    def test_known_models_priced(self):
        assert GEMINI_PRICES_USD_PER_1M["gemini-2.5-flash-lite"] == (0.10, 0.40)
        assert GEMINI_PRICES_USD_PER_1M["gemini-2.5-flash"] == (0.30, 2.50)

    def test_item_cost_flash_lite(self):
        out = {"model": "gemini-2.5-flash-lite", "input_tokens": 1_000_000,
               "output_tokens": 500_000}
        assert item_cost_usd(out) == pytest.approx(0.10 + 0.20)

    def test_item_cost_matches_prefixed_model_name(self):
        out = {"model": "models/gemini-2.5-flash-lite", "input_tokens": 1_000_000,
               "output_tokens": 0}
        assert item_cost_usd(out) == pytest.approx(0.10)

    def test_flash_lite_does_not_match_flash_price(self):
        # "gemini-2.5-flash" is a substring of "gemini-2.5-flash-lite";
        # the longest key must win.
        out = {"model": "gemini-2.5-flash-lite", "input_tokens": 1_000_000,
               "output_tokens": 1_000_000}
        assert item_cost_usd(out) == pytest.approx(0.50)  # not 2.80

    def test_unknown_model_returns_none(self):
        out = {"model": "gpt-x", "input_tokens": 10, "output_tokens": 10}
        assert item_cost_usd(out) is None

    def test_missing_usage_returns_none(self):
        assert item_cost_usd({"model": "gemini-2.5-flash"}) is None
        assert item_cost_usd(None) is None


class TestItemEvaluators:
    def test_correct_label_scores_one(self):
        ev = label_accuracy_evaluator(
            output={"predicted": "real news", "label": "YES"},
            expected_output={"label": "YES"},
        )
        assert ev.name == "label_accuracy"
        assert ev.value == 1.0
        assert "real news" in ev.comment

    def test_wrong_label_scores_zero(self):
        ev = label_accuracy_evaluator(
            output={"predicted": "conference-PR", "label": "conference-PR"},
            expected_output={"label": "legal-call"},
        )
        assert ev.value == 0.0
        assert "legal-call" in ev.comment

    def test_missing_output_scores_zero(self):
        ev = label_accuracy_evaluator(output=None, expected_output={"label": "NO"})
        assert ev.value == 0.0
        assert "no output" in ev.comment

    def test_no_expected_label_skips(self):
        ev = label_accuracy_evaluator(
            output={"predicted": "x", "label": "YES"}, expected_output=None
        )
        assert ev.name == "label_accuracy_skip"

    def test_predicted_label_is_categorical(self):
        ev = predicted_label_evaluator(output={"predicted": "legal-call"})
        assert ev.name == "predicted_label"
        assert ev.value == "legal-call"

    def test_predicted_label_handles_missing_output(self):
        ev = predicted_label_evaluator(output=None)
        assert ev.value == "<none>"

    def test_latency_numeric(self):
        ev = latency_evaluator(output={"latency_s": 1.25})
        assert ev.name == "latency_s"
        assert ev.value == 1.25

    def test_latency_skips_when_absent(self):
        ev = latency_evaluator(output=None)
        assert ev.name == "latency_s_skip"

    def test_cost_numeric(self):
        ev = cost_evaluator(output={"model": "gemini-2.5-flash-lite",
                                    "input_tokens": 1000, "output_tokens": 100})
        assert ev.name == "cost_usd"
        assert ev.value == pytest.approx((1000 * 0.10 + 100 * 0.40) / 1e6)

    def test_cost_skips_unknown_model(self):
        ev = cost_evaluator(output={"model": "mystery", "input_tokens": 1,
                                    "output_tokens": 1})
        assert ev.name == "cost_usd_skip"
        assert "mystery" in ev.value
```

- [ ] **Step 3.2: Run to verify failures**

Run: `.venv/Scripts/python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: ImportError (`EXPERIMENTS`, `item_cost_usd`, ... not defined).

- [ ] **Step 3.3: Implement in `src/ticker_news/evals/classify_eval.py`**

Rewrite the module header and add the pieces (the old GT-csv functions and old evaluators are deleted; `EXPERIMENTS` gets a placeholder completed in Task 5 — define it now with evaluator tuples so the import works):

```python
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
```

Everything else from the old module (`make_task`, `_warn_failed_items`, `run_eval`, `DATASET_DEFAULT`, `EXPERIMENT_PREFIX`, `_DESCRIPTION`, `upsert_dataset_items` import) is deleted in this step; Tasks 4–5 rebuild what's needed. (The CLI is broken until Task 6 — acceptable mid-branch.)

- [ ] **Step 3.4: Run the eval tests**

Run: `.venv/Scripts/python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: ALL pass.

- [ ] **Step 3.5: Commit**

```bash
git add src/ticker_news/evals/classify_eval.py tests/evals/test_classify_eval.py
git commit -m "feat: price table and item-level evaluators for single-pass classify experiments"
```

---

### Task 4: Run-level evaluators

Totals (time/cost/tokens), generic accuracy, binary confusion (YES class), finegrained derived-ACT.

**Files:**
- Modify: `src/ticker_news/evals/classify_eval.py`
- Test: `tests/evals/test_classify_eval.py`

- [ ] **Step 4.1: Append the failing tests:**

```python
def _ir(expected_label, output):
    """Minimal stand-in for an SDK item result: .item and .output."""
    item = {"expected_output": {"label": expected_label} if expected_label else None}
    return SimpleNamespace(item=item, output=output)


def _out(label, *, latency=1.0, tin=1000, tout=100, model="gemini-2.5-flash-lite"):
    return {"predicted": label, "label": label, "latency_s": latency,
            "input_tokens": tin, "output_tokens": tout, "model": model}


def _by_name(evaluations):
    return {e.name: e for e in evaluations}


class TestTotalsRunEvaluator:
    def test_time_cost_token_totals(self, monkeypatch):
        from ticker_news.evals.classify_eval import make_totals_run_evaluator

        monkeypatch.setattr(classify_eval.time, "monotonic", lambda: 110.0)
        totals = make_totals_run_evaluator(started_monotonic=100.0)
        results = [
            _ir("YES", _out("YES", latency=2.0)),
            _ir("NO", _out("NO", latency=4.0)),
        ]
        evals = _by_name(totals(item_results=results))
        assert evals["total_time_s"].value == pytest.approx(10.0)
        assert evals["avg_time_per_item_s"].value == pytest.approx(3.0)
        per_item = (1000 * 0.10 + 100 * 0.40) / 1e6
        assert evals["total_cost_usd"].value == pytest.approx(2 * per_item)
        assert evals["total_tokens"].value == 2200
        assert "input=2000" in evals["total_tokens"].comment

    def test_errored_items_excluded_from_averages_but_not_total_time(self, monkeypatch):
        from ticker_news.evals.classify_eval import make_totals_run_evaluator

        monkeypatch.setattr(classify_eval.time, "monotonic", lambda: 105.0)
        totals = make_totals_run_evaluator(started_monotonic=100.0)
        evals = _by_name(totals(item_results=[_ir("YES", None), _ir("NO", _out("NO", latency=2.0))]))
        assert evals["total_time_s"].value == pytest.approx(5.0)
        assert evals["avg_time_per_item_s"].value == pytest.approx(2.0)
        assert "1/2" in evals["total_cost_usd"].comment

    def test_empty_run_only_reports_total_time(self, monkeypatch):
        from ticker_news.evals.classify_eval import make_totals_run_evaluator

        monkeypatch.setattr(classify_eval.time, "monotonic", lambda: 101.0)
        evals = _by_name(make_totals_run_evaluator(100.0)(item_results=[]))
        assert set(evals) == {"total_time_s"}


class TestLabelAccuracyRunEvaluator:
    def test_average_counts_errored_as_wrong(self):
        from ticker_news.evals.classify_eval import label_accuracy_run_evaluator

        results = [
            _ir("YES", _out("YES")), _ir("NO", _out("YES")), _ir("YES", None),
        ]
        evals = _by_name(label_accuracy_run_evaluator(item_results=results))
        assert evals["label_accuracy_avg"].value == pytest.approx(1 / 3)

    def test_unlabeled_items_excluded(self):
        from ticker_news.evals.classify_eval import label_accuracy_run_evaluator

        results = [_ir(None, _out("YES")), _ir("YES", _out("YES"))]
        evals = _by_name(label_accuracy_run_evaluator(item_results=results))
        assert evals["label_accuracy_avg"].value == pytest.approx(1.0)

    def test_all_unlabeled_skips(self):
        from ticker_news.evals.classify_eval import label_accuracy_run_evaluator

        evals = _by_name(label_accuracy_run_evaluator(item_results=[_ir(None, _out("x"))]))
        assert set(evals) == {"label_accuracy_avg_skip"}


class TestBinaryConfusionRunEvaluator:
    def test_confusion_metrics(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        results = [
            _ir("YES", _out("YES")), _ir("YES", _out("YES")),  # TP x2
            _ir("NO", _out("YES")),                            # FP
            _ir("YES", _out("NO")),                            # FN
            _ir("NO", _out("NO")), _ir("NO", _out("NO")),      # TN x2
        ]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert evals["act_precision"].value == pytest.approx(2 / 3)
        assert evals["act_recall"].value == pytest.approx(2 / 3)
        assert evals["act_f1"].value == pytest.approx(2 / 3)
        assert "TP=2 FP=1 FN=1 TN=2" in evals["act_precision"].comment

    def test_errored_yes_item_is_false_negative(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        results = [_ir("YES", None), _ir("YES", _out("YES"))]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert evals["act_recall"].value == pytest.approx(0.5)

    def test_errored_no_item_is_false_positive(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        # an errored task on a NO item must hurt (FP), not count as TN:
        # TP=0 FP=1 -> precision exists and is 0.0; no YES items -> recall skips
        results = [_ir("NO", None), _ir("NO", _out("NO"))]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert evals["act_precision"].value == 0.0
        assert "TP=0 FP=1 FN=0 TN=1" in evals["act_precision"].comment
        assert "no YES items" in evals["act_recall_skip"].value

    def test_no_yes_predictions_skips_precision(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        results = [_ir("YES", _out("NO")), _ir("NO", _out("NO"))]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert "act_precision" not in evals
        assert "no YES predictions" in evals["act_precision_skip"].value
        assert evals["act_recall"].value == 0.0
        assert "act_f1" not in evals

    def test_empty_skips_everything(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        evals = _by_name(binary_confusion_run_evaluator(item_results=[]))
        assert set(evals) == {"act_metrics_skip"}


class TestDerivedActRunEvaluator:
    def test_miscategorization_within_same_side_still_counts(self):
        from ticker_news.evals.classify_eval import derived_act_run_evaluator

        results = [
            # expected NEWS subtype, predicted different NEWS subtype -> act agrees
            _ir("earnings-reporting", _out("news-event")),
            # expected NOT-NEWS, predicted NOT-NEWS -> act agrees
            _ir("legal-call", _out("marketing fluff")),
            # crosses the boundary -> act disagrees
            _ir("legal-event", _out("legal-call")),
        ]
        evals = _by_name(derived_act_run_evaluator(item_results=results))
        assert evals["derived_act_accuracy"].value == pytest.approx(2 / 3)

    def test_errored_item_counts_as_wrong(self):
        from ticker_news.evals.classify_eval import derived_act_run_evaluator

        results = [_ir("recap/review", None), _ir("recap/review", _out("other"))]
        evals = _by_name(derived_act_run_evaluator(item_results=results))
        assert evals["derived_act_accuracy"].value == pytest.approx(0.5)

    def test_no_labeled_items_skips(self):
        from ticker_news.evals.classify_eval import derived_act_run_evaluator

        evals = _by_name(derived_act_run_evaluator(item_results=[_ir(None, _out("x"))]))
        assert set(evals) == {"derived_act_accuracy_skip"}
```

- [ ] **Step 4.2: Run to verify failures**

Run: `.venv/Scripts/python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: new tests FAIL with ImportError on the four run-evaluator names; Task 3 tests still PASS.

- [ ] **Step 4.3: Implement** — append to `src/ticker_news/evals/classify_eval.py`:

```python
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
            wrong = predicted == "YES" or r.output is None
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
    if precision is not None and recall is not None and (precision + recall) > 0:
        evals.append(Evaluation(name="act_f1",
                                value=2 * precision * recall / (precision + recall),
                                comment=counts))
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
```

- [ ] **Step 4.4: Run the eval tests**

Run: `.venv/Scripts/python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: ALL pass.

- [ ] **Step 4.5: Commit**

```bash
git add src/ticker_news/evals/classify_eval.py tests/evals/test_classify_eval.py
git commit -m "feat: run-level evaluators - totals, binary confusion, derived ACT accuracy"
```

---

### Task 5: Prefetch, task, spec table, and run_eval

**Files:**
- Modify: `src/ticker_news/evals/classify_eval.py`
- Test: `tests/evals/test_classify_eval.py`

- [ ] **Step 5.1: Append the failing tests:**

```python
class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows=None):
        self.executed = []
        self._rows = rows or []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return FakeCursor(self._rows)

    def close(self):
        pass


class TestPrefetchArticles:
    def test_returns_id_to_title_content(self):
        from ticker_news.evals.classify_eval import prefetch_articles

        conn = FakeConn(rows=[(595, "Title 595", "Body 595"), (14682, None, "Body")])
        articles = prefetch_articles(conn, [595, 14682])
        assert articles == {595: ("Title 595", "Body 595"), 14682: ("", "Body")}
        # one parametrized query for all ids
        assert len(conn.executed) == 1
        assert conn.executed[0][1] == ([595, 14682],)

    def test_missing_ids_raise(self):
        from ticker_news.evals.classify_eval import prefetch_articles

        conn = FakeConn(rows=[(595, "T", "B")])
        with pytest.raises(ValueError, match="not found.*14682"):
            prefetch_articles(conn, [595, 14682])

    def test_empty_content_raises(self):
        from ticker_news.evals.classify_eval import prefetch_articles

        conn = FakeConn(rows=[(595, "T", "   ")])
        with pytest.raises(ValueError, match="no scraped content.*595"):
            prefetch_articles(conn, [595])


class StubChain:
    def __init__(self, *verdicts):
        self._verdicts = list(verdicts)
        self.calls = []
        self.configs = []

    async def ainvoke(self, inputs, config=None):
        self.calls.append(inputs)
        self.configs.append(config or {})
        return self._verdicts.pop(0)


def _binary_classifier(*verdicts):
    chain = StubChain(*verdicts)
    return Classifier(
        chain=chain, variant="binary", model="gemini-2.5-flash-lite",
        label_of=lambda v: v.label,
        dataset_label_of=lambda v: "YES" if is_act_binary(v.label) else "NO",
    ), chain


class TestMakeTask:
    def test_task_returns_output_dict(self):
        from ticker_news.evals.classify_eval import make_task

        clf, chain = _binary_classifier(
            BinaryClassification(label="real news", confidence=0.7, reason="earnings")
        )
        task = make_task(clf, {595: ("Title", "Body text")}, "classify-binary")
        out = asyncio.run(task(item={"input": {"article_id": 595}}))
        assert out["predicted"] == "real news"
        assert out["label"] == "YES"
        assert out["confidence"] == 0.7
        assert out["reason"] == "earnings"
        assert out["model"] == "gemini-2.5-flash-lite"
        assert out["latency_s"] >= 0
        # no usage captured from the stub chain -> tokens None, cost skips
        assert out["input_tokens"] is None
        assert out["output_tokens"] is None
        # prefetched text was used; no DB call possible (no conn anywhere)
        assert chain.calls[0]["title"] == "Title"

    def test_task_appends_usage_handler_to_config(self):
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        from ticker_news.evals.classify_eval import make_task

        clf, chain = _binary_classifier(BinaryClassification(label="none news"))
        task = make_task(clf, {1: ("T", "B")}, "classify-binary")
        asyncio.run(task(item={"input": {"article_id": 1}}))
        callbacks = chain.configs[0].get("callbacks", [])
        assert any(isinstance(cb, UsageMetadataCallbackHandler) for cb in callbacks)

    def test_task_names_the_trace(self, monkeypatch):
        import langfuse

        from ticker_news.evals.classify_eval import make_task

        seen = {}

        @contextmanager
        def fake_propagate(**kwargs):
            seen.update(kwargs)
            yield

        monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
        clf, _ = _binary_classifier(BinaryClassification(label="none news"))
        task = make_task(clf, {595: ("T", "B")}, "classify-binary")
        asyncio.run(task(item={"input": {"article_id": 595}}))
        assert seen["trace_name"] == "classify-binary:article-595"


class TestExperimentSpecs:
    def test_spec_table(self):
        assert set(EXPERIMENTS) == {"binary", "finegrained"}
        b, f = EXPERIMENTS["binary"], EXPERIMENTS["finegrained"]
        assert b.dataset == "140-articles-act-no-act"
        assert b.experiment_name == "classify-binary"
        assert f.dataset == "140-articles-categories"
        assert f.experiment_name == "classify-finegrained"

    def test_binary_has_confusion_finegrained_has_derived_act(self):
        from ticker_news.evals.classify_eval import (
            binary_confusion_run_evaluator,
            derived_act_run_evaluator,
            label_accuracy_run_evaluator,
        )

        assert binary_confusion_run_evaluator in EXPERIMENTS["binary"].run_evaluators
        assert derived_act_run_evaluator in EXPERIMENTS["finegrained"].run_evaluators
        for spec in EXPERIMENTS.values():
            assert label_accuracy_run_evaluator in spec.run_evaluators
            assert label_accuracy_evaluator in spec.evaluators
            assert cost_evaluator in spec.evaluators
```

- [ ] **Step 5.2: Run to verify failures**

Run: `.venv/Scripts/python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: new tests FAIL (`prefetch_articles`, `make_task` missing; `EXPERIMENTS` empty).

- [ ] **Step 5.3: Implement** — append to `src/ticker_news/evals/classify_eval.py` (and replace the `EXPERIMENTS: dict = {}` placeholder):

```python
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
            f"to fill them in."
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
```

Also confirm the module imports at top include everything used (`dataclass`, `datetime`, `time`, `psycopg`, `Evaluation`, `connect_eval`) and nothing unused.

- [ ] **Step 5.4: Run the eval tests, then the offline suite**

Run: `.venv/Scripts/python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: ALL pass.
Run: `.venv/Scripts/python.exe -m pytest -m "not db and not integration" -q`
Expected: only `tests/test_cli.py` eval-classify tests fail (old flags) — fixed in Task 6. Anything else failing must be fixed now.

- [ ] **Step 5.5: Commit**

```bash
git add src/ticker_news/evals/classify_eval.py tests/evals/test_classify_eval.py
git commit -m "feat: prefetch, experiment spec table, and run_eval for single-pass experiments"
```

---

### Task 6: CLI rework

**Files:**
- Modify: `src/ticker_news/cli.py` (the `eval classify` command, currently around lines 606–639)
- Test: `tests/test_cli.py` (the five `test_eval_classify_*` tests)

- [ ] **Step 6.1: Replace the five `test_eval_classify_*` tests in `tests/test_cli.py` with:**

```python
def test_eval_classify_passes_args(monkeypatch):
    captured = {}

    def fake_run_eval(variants, **kwargs):
        captured["variants"] = variants
        captured.update(kwargs)
        return []

    from ticker_news.evals import classify_eval
    monkeypatch.setattr(classify_eval, "run_eval", fake_run_eval)
    result = runner.invoke(cli.app, [
        "eval", "classify", "--variant", "binary", "--model", "flash",
        "--ids", "595,14682", "--dsn", "postgresql://x",
        "--run-name", "tuning-1", "--concurrency", "4",
        "--dataset", "my-dataset",
    ])
    assert result.exit_code == 0, result.output
    assert captured["variants"] == ("binary",)
    assert captured["model"] == "flash"
    assert captured["ids"] == [595, 14682]
    assert captured["dsn"] == "postgresql://x"
    assert captured["run_name"] == "tuning-1"
    assert captured["concurrency"] == 4
    assert captured["dataset_name"] == "my-dataset"


def test_eval_classify_defaults(monkeypatch):
    captured = {}

    def fake_run_eval(variants, **kwargs):
        captured["variants"] = variants
        captured.update(kwargs)
        return []

    from ticker_news.evals import classify_eval
    monkeypatch.setattr(classify_eval, "run_eval", fake_run_eval)
    result = runner.invoke(cli.app, ["eval", "classify"])
    assert result.exit_code == 0, result.output
    assert captured["variants"] == ("binary", "finegrained")
    assert captured["model"] == "lite"
    assert captured["dataset_name"] is None
    assert captured["ids"] is None
    assert captured["concurrency"] == 16


def test_eval_classify_rejects_bad_variant():
    result = runner.invoke(cli.app, ["eval", "classify", "--variant", "ternary"])
    assert result.exit_code != 0


def test_eval_classify_rejects_bad_model():
    result = runner.invoke(cli.app, ["eval", "classify", "--model", "ultra"])
    assert result.exit_code != 0


def test_eval_classify_rejects_bad_ids():
    result = runner.invoke(cli.app, ["eval", "classify", "--ids", "1,x"])
    assert result.exit_code != 0


def test_eval_classify_rejects_dataset_with_both_variants():
    result = runner.invoke(cli.app, ["eval", "classify", "--dataset", "d"])
    assert result.exit_code != 0
```

- [ ] **Step 6.2: Run to verify failures**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -k eval_classify -v`
Expected: FAIL (old command still has `--mode`/`--gt-csv`, lacks `--model`/`--concurrency`).

- [ ] **Step 6.3: Replace the `eval_classify` command in `src/ticker_news/cli.py`:**

```python
@eval_app.command("classify")
def eval_classify(
    variant: str = typer.Option("both", "--variant", help="both | binary | finegrained."),
    model: str = typer.Option("lite", "--model", help="lite (gemini-2.5-flash-lite) | flash (gemini-2.5-flash)."),
    dataset: str | None = typer.Option(None, "--dataset", help="Dataset override (single --variant only; default: the variant's own dataset)."),
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated article ids: run only this dataset subset (fast prompt iteration)."),
    dsn: str | None = typer.Option(None, "--dsn", help="Target DB DSN (default: DATABASE_URL)."),
    run_name: str | None = typer.Option(None, "--run-name", help="Experiment run name (default: <variant>-<model>-<timestamp>)."),
    concurrency: int = typer.Option(16, "--concurrency", help="Max concurrent dataset items."),
) -> None:
    """Single-pass classifier experiments: binary vs ACT labels, finegrained vs categories."""
    from ticker_news.classification.variants import MODEL_CHOICES, VARIANTS
    from ticker_news.evals import classify_eval

    if variant not in ("both", *VARIANTS):
        raise typer.BadParameter(f"--variant must be one of: both, {', '.join(VARIANTS)}")
    if model not in MODEL_CHOICES:
        raise typer.BadParameter(f"--model must be one of: {', '.join(MODEL_CHOICES)}")
    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else None
    except ValueError as exc:
        raise typer.BadParameter(f"--ids must be comma-separated integers: {exc}")
    if dataset and variant == "both":
        raise typer.BadParameter("--dataset requires a single --variant")
    variants = VARIANTS if variant == "both" else (variant,)
    try:
        results = classify_eval.run_eval(
            variants, model=model, dataset_name=dataset,
            dsn=dsn, run_name=run_name, ids=id_list, concurrency=concurrency,
        )
    except SystemExit as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    for variant_name, result in results:
        typer.echo(f"=== {variant_name} ===")
        _echo_summary(result.format())
```

- [ ] **Step 6.4: Run the CLI tests, then the offline suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: ALL pass.
Run: `.venv/Scripts/python.exe -m pytest -m "not db and not integration" -q`
Expected: ALL pass (the tree is fully green again).

- [ ] **Step 6.5: Commit**

```bash
git add src/ticker_news/cli.py tests/test_cli.py
git commit -m "feat: eval classify CLI for single-pass experiments (--model, --concurrency)"
```

---

### Task 7: Docs

**Files:**
- Modify: `CLAUDE.md` (the `eval classify` row in the Commands table)

- [ ] **Step 7.1: Update the command-table row** — in `CLAUDE.md`, replace:

```
| `ticker-news eval classify --gt-csv F` | Classification prompt eval: binary vs fine-grained variants scored against ground-truth ACT labels on a Langfuse dataset (`--variant`, `--mode lite\|flash\|two-pass`, `--ids` subset) |
```

with:

```
| `ticker-news eval classify` | Single-pass classifier experiments: binary vs `140-articles-act-no-act`, finegrained vs `140-articles-categories`; scores accuracy + time/cost per run (`--variant`, `--model lite\|flash`, `--ids`, `--concurrency`) |
```

(If the exact old row text differs, find the `eval classify` row and rewrite it with the new text.)

- [ ] **Step 7.2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: eval classify command reflects single-pass experiments"
```

---

### Task 8: Live verification (needs user keys + shared DB)

The offline suite cannot prove the Langfuse wiring. Verify against the real services — coordinate with the user (keys come from `.env`; the articles live in the shared tunneled DB, same DSN used when the datasets were seeded — ask the user to confirm the `--dsn` value, likely `postgresql://...@localhost:15432/...`).

Known env gotcha (from memory): a stale `OPENAI_API_KEY` may sit in the OS env and override `.env` — irrelevant here (no embeddings), but a stale `GOOGLE_API_KEY` would matter; if auth fails, check `$env:GOOGLE_API_KEY`.

- [ ] **Step 8.1: Smoke run — 3 items, binary**

Run: `.venv/Scripts/ticker-news.exe eval classify --variant binary --ids <3 ids from the dataset, e.g. 10190,15036,5802> --dsn <shared DSN> --run-name smoke-binary`
Expected: exits 0; prints the experiment summary with `label_accuracy_avg`, `act_precision`/`_skip`, `total_time_s`, `avg_time_per_item_s`, `total_cost_usd`, `total_tokens`.

- [ ] **Step 8.2: Verify in Langfuse**

Check (UI or `npx langfuse-cli api`): the run `smoke-binary` exists on experiment `classify-binary`; traces are named `classify-binary:article-<id>`; the generation inside shows model `gemini-2.5-flash-lite` and is **linked to the `classify-binary` prompt version**; run-level scores include the time/cost metrics.

- [ ] **Step 8.3: Full runs, both variants**

Run: `.venv/Scripts/ticker-news.exe eval classify --dsn <shared DSN>`
Expected: two experiments complete (140 items each) in roughly a minute apiece or less; `_warn_failed_items` prints nothing (or lists ids to re-run).

- [ ] **Step 8.4: Report results to the user**

Summarize: accuracy/precision/recall/F1 (binary), exact-match + derived ACT accuracy (finegrained), total time, avg per item, total cost for each run. No commit — this task produces no code changes.

---

## Self-review notes

- **Spec coverage:** classifiers single-pass (Task 2), prompt linking (Tasks 1–2), prefetch + one-LLM-call task (Task 5), spec table + two experiments (Task 5), item/run evaluators incl. time+cost (Tasks 3–4), CLI (Task 6), docs (Task 7), live verification (Task 8). Error handling: prefetch loud failure (5), `_warn_failed_items` (5), SystemExits (5–6), bad-template fallback without link (2).
- **Type consistency:** `Classifier(chain, variant, model, label_of, dataset_label_of)` used identically in Tasks 2 and 5; `make_task(classifier, articles, trace_prefix)` consistent; output dict keys (`predicted`, `label`, `reason`, `confidence`, `latency_s`, `input_tokens`, `output_tokens`, `model`) consistent across task, item evaluators, and run evaluators; `run_eval(variants, model=, dataset_name=, dsn=, run_name=, ids=, concurrency=)` matches the CLI call.
- **Known mid-branch breakage:** after Task 2 the eval/CLI temporarily reference deleted names; restored by Tasks 3–6. Each task's own tests are green at its commit.
