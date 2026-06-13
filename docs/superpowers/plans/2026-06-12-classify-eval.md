# Classification Eval + Fast Repeatable E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only `ticker-news eval classify` Langfuse experiment comparing binary vs fine-grained classification prompts against the 140-article ground truth, plus `--skip-stages`/`--ids-file` on `eval pipeline` for fast repeatable E2E runs.

**Architecture:** New `classification/variants.py` holds the two experimental prompt variants (schemas, templates, chain builders, two-pass runner); new `evals/classify_eval.py` turns the GT csv into a Langfuse dataset and runs one experiment per variant with accuracy/precision/recall/F1 scoring; `evals/pipeline_eval.py` gains keep-semantics in `reset_article` so idempotent stage adapters skip preserved stages naturally.

**Tech Stack:** Python 3.11+, Typer CLI, LangChain + langchain-google-genai (Gemini structured output), Langfuse SDK v4 experiments, psycopg/Postgres (shared DB via tunnel), pytest.

**Spec:** `docs/superpowers/specs/2026-06-12-classify-eval-design.md`

**Verified SDK facts (don't re-derive):**
- Langfuse v4 `_run_task` awaits coroutine tasks; `max_concurrency` (default 50) gates concurrent items — an **async** task gets real parallelism (sync tasks run serially).
- `client.run_experiment(data=<list of DatasetItem>)` links every item that has `.id` and `.dataset_id` to a dataset run (`dataset_run_items.create`) — so passing a **filtered subset** of `dataset.items` keeps full dataset-run linkage in the UI.
- `Evaluation(name=..., value=<str>)` is the categorical-score convention already used by `pipeline_eval` for exclusions (Langfuse rejects `value=None`).
- Run evaluators receive `item_results`; each result has `.item` (dict or DatasetItem with `.expected_output`) and `.output`.

**Conventions to follow:**
- Offline tests only (no `db`/`integration` markers) in `tests/evals/`, using the `FakeConn`/`FakeCursor` pattern from `tests/evals/test_pipeline_eval.py`.
- CLI tests use `typer.testing.CliRunner` + monkeypatch, pattern from `tests/test_cli.py`.
- Commits: clean messages, **no Co-Authored-By / no attribution lines** (user rule).
- Run tests with: `.venv\Scripts\python.exe -m pytest <path> -v` from the repo root.

---

### Task 1: Variant schemas, categories, and ACT mapping (`classification/variants.py`)

**Files:**
- Create: `src/ticker_news/classification/variants.py`
- Create: `tests/classification/test_variants.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/classification/test_variants.py`:

```python
"""Offline tests for the experimental classification variants."""

import pytest
from pydantic import ValidationError

from ticker_news.classification.variants import (
    BINARY_LABELS,
    FINEGRAINED_CATEGORIES,
    NEWS_SUBTYPES,
    BinaryClassification,
    FinegrainedClassification,
    is_act_binary,
    is_act_finegrained,
)


class TestSchemas:
    def test_binary_labels(self):
        assert BINARY_LABELS == ["real news", "none news"]

    def test_binary_schema_accepts_valid_label(self):
        v = BinaryClassification(label="real news", confidence=0.9, reason="earnings")
        assert v.label == "real news"

    def test_binary_schema_rejects_unknown_label(self):
        with pytest.raises(ValidationError):
            BinaryClassification(label="maybe news")

    def test_binary_confidence_and_reason_optional(self):
        v = BinaryClassification(label="none news")
        assert v.confidence is None
        assert v.reason == ""

    def test_finegrained_has_16_categories(self):
        assert len(FINEGRAINED_CATEGORIES) == 16
        assert len(set(FINEGRAINED_CATEGORIES)) == 16

    def test_finegrained_schema_rejects_unknown_category(self):
        with pytest.raises(ValidationError):
            FinegrainedClassification(category="not-a-category")

    def test_finegrained_reason_optional(self):
        v = FinegrainedClassification(category="earnings-reporting")
        assert v.reason == ""


class TestActMapping:
    def test_binary_real_news_is_act(self):
        assert is_act_binary("real news") is True

    def test_binary_none_news_is_not_act(self):
        assert is_act_binary("none news") is False

    def test_news_subtypes_are_the_seven_from_the_spec(self):
        assert NEWS_SUBTYPES == frozenset({
            "earnings-reporting", "dividend-reporting",
            "merger/investment/funding", "legal-event", "MACRO-investment",
            "news-event", "news-report",
        })

    def test_every_finegrained_category_maps(self):
        acts = {c for c in FINEGRAINED_CATEGORIES if is_act_finegrained(c)}
        assert acts == NEWS_SUBTYPES
        non_acts = set(FINEGRAINED_CATEGORIES) - acts
        assert non_acts == {
            "recap/review", "market speculation", "MACRO-political",
            "legal-call", "conference-PR", "marketing fluff", "book PR",
            "Other-filing-reporting", "other",
        }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/classification/test_variants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ticker_news.classification.variants'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/ticker_news/classification/variants.py`:

```python
"""Experimental classification prompt variants: binary and fine-grained.

Candidate replacements for the production classifier, evaluated by
`ticker-news eval classify` against the hand-labeled ground-truth set
(spec: docs/superpowers/specs/2026-06-12-classify-eval-design.md).
Not wired into the pipeline; promoting a winner stays a human decision.
"""

from __future__ import annotations

from typing import Literal, Optional, get_args

from pydantic import BaseModel

BinaryLabel = Literal["real news", "none news"]
BINARY_LABELS: list[str] = list(get_args(BinaryLabel))


class BinaryClassification(BaseModel):
    """Structured verdict of the binary ACT / DON'T-ACT classifier."""

    label: BinaryLabel
    confidence: Optional[float] = None
    reason: str = ""


FinegrainedCategory = Literal[
    # --- a real, newly-occurring event is being reported (NEWS) ---
    "earnings-reporting",
    "dividend-reporting",
    "merger/investment/funding",
    "legal-event",
    "MACRO-investment",
    "news-event",
    "news-report",
    # --- not a new event (NOT-NEWS) ---
    "recap/review",
    "market speculation",
    "MACRO-political",
    "legal-call",
    "conference-PR",
    "marketing fluff",
    "book PR",
    "Other-filing-reporting",
    "other",
]
FINEGRAINED_CATEGORIES: list[str] = list(get_args(FinegrainedCategory))

# Fine-grained categories that count as ACT/YES for the binary ground truth.
NEWS_SUBTYPES: frozenset[str] = frozenset({
    "earnings-reporting",
    "dividend-reporting",
    "merger/investment/funding",
    "legal-event",
    "MACRO-investment",
    "news-event",
    "news-report",
})


class FinegrainedClassification(BaseModel):
    """Structured verdict of the fine-grained taxonomy classifier."""

    category: FinegrainedCategory
    reason: str = ""


def is_act_binary(label: str) -> bool:
    return label == "real news"


def is_act_finegrained(category: str) -> bool:
    return category in NEWS_SUBTYPES
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/classification/test_variants.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ticker_news/classification/variants.py tests/classification/test_variants.py
git commit -m "feat: binary + fine-grained classification variant schemas"
```

---

### Task 2: Prompt templates + registry entries

**Files:**
- Modify: `src/ticker_news/classification/variants.py` (append templates)
- Modify: `src/ticker_news/shared/prompts.py:63-76` (`registry()`)
- Modify: `tests/shared/test_prompts.py:49-57` (registry coverage test)

- [ ] **Step 1: Update the failing registry test**

In `tests/shared/test_prompts.py`, change `test_registry_covers_all_llm_prompts` to expect the two new names:

```python
def test_registry_covers_all_llm_prompts(monkeypatch):
    _disable(monkeypatch)
    reg = prompts.registry()
    assert set(reg) == {
        "classify-article", "extract-insights",
        "classify-binary", "classify-finegrained",
        "analyst-fundamentals", "analyst-market_context",
        "analyst-historical_precedent", "synthesize-verdict",
    }
    assert all(isinstance(v, str) and v for v in reg.values())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/shared/test_prompts.py::test_registry_covers_all_llm_prompts -v`
Expected: FAIL — registry is missing `classify-binary` / `classify-finegrained`

- [ ] **Step 3: Append the two templates to `variants.py`**

Append to `src/ticker_news/classification/variants.py` (these are the user's drafts adapted verbatim — `{title}`/`{body}` placeholders, `{{ }}` escapes for the JSON example, same contract as the production prompt in `classification/chain.py`):

```python
BINARY_PROMPT_TEMPLATE = """You are an equity-research editor deciding whether an article is genuine market NEWS or not.

Label "real news" ONLY if the article is FIRST-HAND reporting of a CONCRETE, NEWLY-OCCURRING, market-relevant EVENT -- something that just happened that a trader could act on. Qualifying events include:
- earnings / results / guidance changes; a dividend declared or changed;
- M&A, a strategic equity investment / stake, or a funding round;
- a product / service launch, a partnership / contract / agreement, a leadership or
  board change, an expansion or new entity, a listing;
- a supply-chain disruption, a major customer win or loss, a regulatory action, or
  another operational development that materially changes the business;
- a lawsuit / class action actually FILED, an investigation opened, a ruling / settlement;
- a GOVERNMENT equity / investment action (taking a stake, a national investment program);
- a bankruptcy, restructuring, or insolvency event; a major contract win or loss; or a
  significant technological breakthrough.
The article must BE the primary report of the event.

Label "none news" for everything else, including:
- opinion / recap -- "why X moved", "should you buy X?", predictions, listicles, market-wraps;
- PR / marketing / advertorials / product copy / market-research-report promos / book PR;
- conference / trade-show / awards announcements;
- law-firm SOLICITATIONS -- "investor alert", "deadline alert", "opportunity to lead",
  "contact the firm", lead-plaintiff reminders;
- general politics / macro / policy commentary -- tariffs, Fed / rates, elections,
  inflation or jobs data, broad market direction;
- routine filings reported as notices -- share buybacks, insider / managers' transactions,
  annual reports / proxy notices;
- rehashed summaries of already-known events, pure speculation / commentary with no new
  facts, generic market chatter, or content whose primary purpose is engagement not info;
- lifestyle / entertainment / human-interest / obituaries / charity / university events /
  boilerplate / anything else.

Two quick tests for "real news":
(1) Did a CONCRETE event just happen (not a prediction, not a recap of an old move)?
(2) Is THIS article the PRIMARY source breaking it (not a law firm's pitch, not an
    opinion column, not a marketing release)?
Both YES -> "real news". Otherwise -> "none news".
Borderline guidance: a law-firm "lawsuit filed" report and a government taking a company
stake ARE real news; a law-firm deadline/solicitation and a tariff/Fed commentary are NOT.

Judge by MATERIALITY, not sentiment or tone. Positive, negative, and neutral developments
can ALL be "real news" if the event could plausibly affect a company's revenue,
profitability, growth prospects, competitive position, market share, regulatory /
operational risk, strategic direction, or investor perception. Conversely, a highly
emotional or dramatic article that carries NO concrete new information is "none news".
Ignore sentiment, and ignore WHICH ticker or company is involved -- judge only whether the
article carries genuinely NEW, investment-relevant factual information. New facts outweigh
opinion; materiality outweighs popularity or drama. When genuinely uncertain, prefer
"none news".

Return ONLY a JSON object:
{{"label": "<real news | none news>", "confidence": 0.0-1.0, "reason": "<one short clause>"}}

ARTICLE TITLE: {title}

ARTICLE BODY:
\"\"\"
{body}
\"\"\""""


FINEGRAINED_PROMPT_TEMPLATE = """You are an equity-research editor sorting financial news articles into a fine-grained taxonomy.

First decide: is the article FIRST-HAND reporting of a CONCRETE, NEWLY-OCCURRING, market-relevant EVENT (a thing that just happened -- results, a deal, a filing, a launch, a government action)? Or is it promotion / opinion / solicitation / a recap of an already-happened move?

Then assign EXACTLY ONE label.

=== A real new event is being reported (NEWS) ===
- "earnings-reporting": first-hand reporting of a company's earnings / financial
  results, revenue, a guidance change, or an earnings-call transcript.
  (e.g. "X reports Q4 results", "X beats on EPS", "<co> earnings call transcript".)
- "dividend-reporting": a company DECLARING or CHANGING a cash dividend, or a fund
  announcing distributions. (e.g. "declares quarterly dividend", "raises dividend 10%".)
  NOT share buybacks -> "Other-filing-reporting".
- "merger/investment/funding": an M&A acquisition/merger, a strategic equity
  investment or stake, or a funding round. (e.g. "X acquires Y", "secures investment
  from Z".) A purely commercial supply/partnership deal with NO equity -> "news-event".
- "legal-event": an actual legal action that OCCURRED -- a lawsuit / class action
  FILED, an investigation opened, a ruling or settlement. The hook is the filing itself.
- "MACRO-investment": a GOVERNMENT acting on equity / investment -- the government
  taking or weighing a stake in a company, or a government-driven investment
  program/initiative. (e.g. "White House eyes Intel stake", "Trump announces $500B AI
  investment".)
- "news-event": any OTHER first-hand corporate event announced by the parties --
  product/service launch, partnership / contract / agreement, leadership or board
  appointment, expansion / new entity, listing, capacity / pricing change. The article
  IS the announcement (typically a company press release).
- "news-report": SECONDARY journalism about a development -- sourced from "a report" /
  Bloomberg / Fortune, rumors, layoffs / restructuring plans, multi-item deal roundups,
  or reporting on a third-party study. Reportage ABOUT events, not a primary release.

=== Not a new event (NOT-NEWS) ===
- "recap/review": opinion / educational / retrospective -- "why X rose/fell today",
  "should you buy X?", "3 stocks to buy now", "what [event] means for investors",
  performance explainers, market-wraps, listicles. Explaining an already-happened move.
- "market speculation": forward-looking conjecture -- predictions, "will X hit $Y",
  price-target / scenario guesswork.
- "MACRO-political": general political / policy / geopolitical / macro news NOT tied to
  a government equity action -- tariffs, Fed / rate decisions, elections, legislation,
  war / sanctions, inflation / jobs data, broad market commentary.
- "legal-call": a law-firm SOLICITATION -- "investor / deadline / shareholder alert",
  "opportunity to lead", "secure counsel before deadline", "contact the firm",
  lead-plaintiff reminders. A call to action, not a new filing.
- "conference-PR": a company sponsoring / speaking / exhibiting / being recognized at a
  conference, trade show, summit, or AWARDS event; event promotion / registration.
- "marketing fluff": promotional / advertorial / vendor self-promotion, product
  marketing copy, market-research-report promos, newsletter / subscription promos,
  "about the company" filler -- no decision-useful substance.
- "book PR": a press release announcing / promoting a book (novel, poetry, children's,
  memoir, self-help) or an author / book award.
- "Other-filing-reporting": a routine corporate / exchange filing reported as a NOTICE
  -- share buybacks / repurchases, managers' / insider transactions, annual reports /
  Form 20-F, proxy / meeting notices. (A dividend goes to "dividend-reporting".)
- "other": genuinely none of the above -- obituaries, charity, university
  commencements, human-interest, retraction notices, boilerplate.

If an article fits more than one NEWS subtype, prefer in this order:
earnings-reporting > dividend-reporting > merger/investment/funding > MACRO-investment >
legal-event > (news-event if it is a primary press release, else news-report).

Pick the single best fit. Return ONLY a JSON object:
{{"category": "<one category>", "reason": "<one short clause justifying the label>"}}

ARTICLE TITLE: {title}

ARTICLE BODY:
\"\"\"
{body}
\"\"\""""
```

- [ ] **Step 4: Add the registry entries**

In `src/ticker_news/shared/prompts.py`, inside `registry()`, change the imports and dict:

```python
def registry() -> dict[str, str]:
    """name -> in-repo fallback text, for `ticker-news prompts push`."""
    from ticker_news.classification.chain import PROMPT_TEMPLATE as classify_prompt
    from ticker_news.classification.variants import (
        BINARY_PROMPT_TEMPLATE,
        FINEGRAINED_PROMPT_TEMPLATE,
    )
    from ticker_news.enrichment.insights import PROMPT_TEMPLATE as insights_prompt
    from ticker_news.sentiment.analysts import ANALYST_PROMPTS, SYNTHESIS_PROMPT

    reg = {
        "classify-article": classify_prompt,
        "classify-binary": BINARY_PROMPT_TEMPLATE,
        "classify-finegrained": FINEGRAINED_PROMPT_TEMPLATE,
        "extract-insights": insights_prompt,
        "synthesize-verdict": SYNTHESIS_PROMPT,
    }
    for role, prompt in ANALYST_PROMPTS.items():
        reg[f"analyst-{role}"] = prompt
    return reg
```

- [ ] **Step 5: Run the prompts tests**

Run: `.venv\Scripts\python.exe -m pytest tests/shared/test_prompts.py tests/classification/test_variants.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/ticker_news/classification/variants.py src/ticker_news/shared/prompts.py tests/shared/test_prompts.py
git commit -m "feat: variant prompt templates + Langfuse registry entries"
```

---

### Task 3: Chain builders + two-pass VariantRunner

**Files:**
- Modify: `src/ticker_news/classification/variants.py` (append builders/runner)
- Modify: `tests/classification/test_variants.py` (append runner tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/classification/test_variants.py`:

```python
import asyncio


class StubChain:
    """Async chain double recording invocations; returns canned verdicts in order."""

    def __init__(self, *verdicts):
        self._verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def ainvoke(self, inputs, config=None):
        self.calls.append(inputs)
        result = self._verdicts.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _run(coro):
    return asyncio.run(coro)


class TestVariantRunner:
    def test_two_pass_confirms_act_verdicts(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="real news"))
        confirm = StubChain(BinaryClassification(label="none news"))
        runner = VariantRunner(lite=lite, confirm=confirm,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "none news"   # confirm overturned
        assert confirmed is True
        assert len(lite.calls) == 1 and len(confirm.calls) == 1

    def test_two_pass_skips_confirm_for_non_act(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="none news"))
        confirm = StubChain()
        runner = VariantRunner(lite=lite, confirm=confirm,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "none news"
        assert confirmed is False
        assert confirm.calls == []

    def test_two_pass_keeps_lite_verdict_when_confirm_fails(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="real news", confidence=0.8))
        confirm = StubChain(RuntimeError("quota"))
        runner = VariantRunner(lite=lite, confirm=confirm,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "real news"
        assert confirmed is True

    def test_single_pass_lite_never_confirms(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="real news"))
        runner = VariantRunner(lite=lite, confirm=None,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "real news"
        assert confirmed is False

    def test_single_pass_flash_only(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        flash = StubChain(BinaryClassification(label="real news"))
        runner = VariantRunner(lite=None, confirm=flash,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "real news"
        assert confirmed is False
        assert len(flash.calls) == 1

    def test_inputs_truncated_like_production(self):
        from ticker_news.classification.chain import MAX_ARTICLE_CHARS
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="none news"))
        runner = VariantRunner(lite=lite, confirm=None,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        _run(runner.classify("  t  " * 200, "x" * 10_000))
        sent = lite.calls[0]
        assert len(sent["title"]) <= 300
        assert len(sent["body"]) == MAX_ARTICLE_CHARS

    def test_fine_grained_two_pass_uses_news_subtypes(self):
        from ticker_news.classification.variants import (
            VariantRunner, is_act_finegrained,
        )

        lite = StubChain(FinegrainedClassification(category="conference-PR"))
        confirm = StubChain()
        runner = VariantRunner(lite=lite, confirm=confirm,
                               is_act=is_act_finegrained,
                               label_of=lambda v: v.category)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.category == "conference-PR"
        assert confirmed is False
        assert confirm.calls == []


class TestMakeRunner:
    def test_rejects_unknown_variant_and_mode(self):
        from ticker_news.classification.variants import make_runner

        with pytest.raises(ValueError, match="variant"):
            make_runner("ternary", "lite")
        with pytest.raises(ValueError, match="mode"):
            make_runner("binary", "warp")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/classification/test_variants.py -v`
Expected: FAIL — `ImportError: cannot import name 'VariantRunner'`

- [ ] **Step 3: Write the implementation**

Append to `src/ticker_news/classification/variants.py` (and extend the module's imports):

```python
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

GEMINI_TIMEOUT_S = 60.0
RETRIES = 4

MODES = ("lite", "flash", "two-pass")
VARIANTS = ("binary", "finegrained")


def _build(model_name: str, prompt_name: str, fallback: str, schema: type):
    """prompt | structured-output Gemini, with the same Langfuse-template
    guard as the production classifier. NOT cached — the eval rebuilds per
    run so Langfuse prompt edits apply without a process restart."""
    from ticker_news.shared.llm import gemini_chat
    from ticker_news.shared.prompts import get_prompt

    llm = gemini_chat(model_name, timeout_s=GEMINI_TIMEOUT_S)
    structured = llm.with_structured_output(schema).with_retry(
        stop_after_attempt=RETRIES, wait_exponential_jitter=True
    )
    template = get_prompt(prompt_name, fallback)
    try:
        prompt = ChatPromptTemplate.from_template(template)
        if set(prompt.input_variables) != {"title", "body"}:
            raise ValueError(f"unexpected variables: {prompt.input_variables}")
    except Exception as exc:
        logger.warning("%s prompt invalid (%r); using in-repo fallback", prompt_name, exc)
        prompt = ChatPromptTemplate.from_template(fallback)
    return prompt | structured


def build_binary_classifier(model_name: str):
    return _build(model_name, "classify-binary",
                  BINARY_PROMPT_TEMPLATE, BinaryClassification)


def build_finegrained_classifier(model_name: str):
    return _build(model_name, "classify-finegrained",
                  FINEGRAINED_PROMPT_TEMPLATE, FinegrainedClassification)


@dataclass
class VariantRunner:
    """One variant/mode pairing. Two-pass mirrors production semantics:
    `lite` labels everything, `confirm` re-runs only verdicts `is_act` calls
    ACT; a failed confirmation keeps the lite verdict. Single-pass modes set
    exactly one chain."""

    lite: Optional[Any]
    confirm: Optional[Any]
    is_act: Callable[[str], bool]
    label_of: Callable[[Any], str]

    async def classify(self, title: Optional[str], body: str,
                       config=None) -> Tuple[Any, bool]:
        from ticker_news.classification.chain import MAX_ARTICLE_CHARS

        inputs = {
            "title": (title or "").strip()[:300],
            "body": (body or "")[:MAX_ARTICLE_CHARS],
        }
        first_chain = self.lite if self.lite is not None else self.confirm
        first = await first_chain.ainvoke(inputs, config=config)
        two_pass = self.lite is not None and self.confirm is not None
        if not two_pass or not self.is_act(self.label_of(first)):
            return first, False
        try:
            return await self.confirm.ainvoke(inputs, config=config), True
        except Exception as exc:
            logger.warning("confirmation pass failed (%r); keeping lite verdict", exc)
            return first, True


def make_runner(variant: str, mode: str) -> VariantRunner:
    """Build the chains for a variant/mode pair (fresh — no lru_cache)."""
    from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of {VARIANTS})")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")
    if variant == "binary":
        build, is_act, label_of = (
            build_binary_classifier, is_act_binary, lambda v: v.label)
    else:
        build, is_act, label_of = (
            build_finegrained_classifier, is_act_finegrained, lambda v: v.category)
    lite = build(GEMINI_FLASH_LITE) if mode in ("lite", "two-pass") else None
    confirm = build(GEMINI_FLASH) if mode in ("flash", "two-pass") else None
    return VariantRunner(lite=lite, confirm=confirm, is_act=is_act, label_of=label_of)


def models_for_mode(mode: str) -> dict[str, str]:
    """Model names per mode, for experiment-run metadata."""
    from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE

    if mode == "lite":
        return {"lite": GEMINI_FLASH_LITE}
    if mode == "flash":
        return {"flash": GEMINI_FLASH}
    return {"lite": GEMINI_FLASH_LITE, "confirm": GEMINI_FLASH}
```

Note: `make_runner` validation happens before any chain construction, so the
`TestMakeRunner` tests need no API key (`gemini_chat` is never reached).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/classification/test_variants.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ticker_news/classification/variants.py tests/classification/test_variants.py
git commit -m "feat: variant chain builders + two-pass runner"
```

---

### Task 4: Ground-truth loader (`evals/classify_eval.py`)

**Files:**
- Create: `src/ticker_news/evals/classify_eval.py`
- Create: `tests/evals/test_classify_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/evals/test_classify_eval.py`:

```python
"""Offline unit tests for the classification eval. No DB, no network."""

import pytest

from ticker_news.evals.classify_eval import load_ground_truth


def _write_csv(tmp_path, text, *, bom=True, name="gt.csv"):
    path = tmp_path / name
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text(text, encoding=encoding)
    return path


GOOD_CSV = (
    "article id,header,Act_GT\n"
    "595,Some headline,NO\n"
    "14682,KraneShares Cross-Lists KOID,YES\n"
)


class TestLoadGroundTruth:
    def test_loads_rows_with_bom(self, tmp_path):
        rows = load_ground_truth(_write_csv(tmp_path, GOOD_CSV))
        assert rows == [
            {"article_id": 595, "header": "Some headline", "act": "NO"},
            {"article_id": 14682, "header": "KraneShares Cross-Lists KOID", "act": "YES"},
        ]

    def test_normalizes_case_and_whitespace(self, tmp_path):
        csv_text = "article id,header,Act_GT\n 595 ,H, yes \n"
        rows = load_ground_truth(_write_csv(tmp_path, csv_text, bom=False))
        assert rows == [{"article_id": 595, "header": "H", "act": "YES"}]

    def test_duplicate_id_raises_with_line_number(self, tmp_path):
        csv_text = "article id,header,Act_GT\n595,A,NO\n595,B,YES\n"
        with pytest.raises(ValueError, match="line 3.*duplicate"):
            load_ground_truth(_write_csv(tmp_path, csv_text))

    def test_bad_act_value_raises(self, tmp_path):
        csv_text = "article id,header,Act_GT\n595,A,MAYBE\n"
        with pytest.raises(ValueError, match="line 2.*MAYBE"):
            load_ground_truth(_write_csv(tmp_path, csv_text))

    def test_non_integer_id_raises(self, tmp_path):
        csv_text = "article id,header,Act_GT\nabc,A,NO\n"
        with pytest.raises(ValueError, match="line 2.*abc"):
            load_ground_truth(_write_csv(tmp_path, csv_text))

    def test_missing_column_raises(self, tmp_path):
        csv_text = "id,header,label\n595,A,NO\n"
        with pytest.raises(ValueError, match="Act_GT"):
            load_ground_truth(_write_csv(tmp_path, csv_text))

    def test_empty_csv_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no rows"):
            load_ground_truth(_write_csv(tmp_path, "article id,header,Act_GT\n"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ticker_news.evals.classify_eval'`

- [ ] **Step 3: Write the implementation**

Create `src/ticker_news/evals/classify_eval.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ticker_news/evals/classify_eval.py tests/evals/test_classify_eval.py
git commit -m "feat: ground-truth csv loader for classify eval"
```

---

### Task 5: Dataset items + item evaluators

**Files:**
- Modify: `src/ticker_news/evals/classify_eval.py`
- Modify: `tests/evals/test_classify_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evals/test_classify_eval.py` (reuse the FakeConn pattern from `test_pipeline_eval.py`):

```python
from ticker_news.evals.classify_eval import (
    act_accuracy_evaluator,
    build_items,
    predicted_label_evaluator,
)


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    def __init__(self, rows=None):
        self.executed = []
        self._rows = rows or []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return FakeCursor(self._rows)


GT_ROWS = [
    {"article_id": 595, "header": "H595", "act": "NO"},
    {"article_id": 14682, "header": "H14682", "act": "YES"},
]


class TestBuildItems:
    def test_builds_dataset_items(self):
        conn = FakeConn(rows=[
            (595, "Title 595", "ok", True),
            (14682, "Title 14682", "ok", True),
        ])
        items = build_items(conn, GT_ROWS)
        assert items == [
            {
                "id": "article-595",
                "input": {"article_id": 595, "title": "Title 595"},
                "expected_output": {"act": "NO"},
                "metadata": {"gt_header": "H595"},
            },
            {
                "id": "article-14682",
                "input": {"article_id": 14682, "title": "Title 14682"},
                "expected_output": {"act": "YES"},
                "metadata": {"gt_header": "H14682"},
            },
        ]

    def test_missing_ids_raise(self):
        conn = FakeConn(rows=[(595, "T", "ok", True)])
        with pytest.raises(ValueError, match="not found.*14682"):
            build_items(conn, GT_ROWS)

    def test_unscraped_articles_raise(self):
        conn = FakeConn(rows=[
            (595, "T", "error", False),
            (14682, "T", "ok", True),
        ])
        with pytest.raises(ValueError, match="no scraped content.*595"):
            build_items(conn, GT_ROWS)


class TestItemEvaluators:
    def test_correct_act_scores_one(self):
        ev = act_accuracy_evaluator(
            output={"predicted": "real news", "act": "YES"},
            expected_output={"act": "YES"},
        )
        assert ev.name == "act_accuracy"
        assert ev.value == 1.0
        assert "real news" in ev.comment

    def test_wrong_act_scores_zero(self):
        ev = act_accuracy_evaluator(
            output={"predicted": "conference-PR", "act": "NO"},
            expected_output={"act": "YES"},
        )
        assert ev.value == 0.0
        assert "gt=YES" in ev.comment

    def test_missing_output_scores_zero_with_comment(self):
        ev = act_accuracy_evaluator(output=None, expected_output={"act": "NO"})
        assert ev.value == 0.0
        assert "no output" in ev.comment

    def test_predicted_label_is_categorical(self):
        ev = predicted_label_evaluator(output={"predicted": "legal-call", "act": "NO"})
        assert ev.name == "predicted_label"
        assert ev.value == "legal-call"

    def test_predicted_label_handles_missing_output(self):
        ev = predicted_label_evaluator(output=None)
        assert ev.value == "<none>"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_items'`

- [ ] **Step 3: Write the implementation**

Append to `src/ticker_news/evals/classify_eval.py`:

```python
import psycopg
from langfuse import Evaluation


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ticker_news/evals/classify_eval.py tests/evals/test_classify_eval.py
git commit -m "feat: classify-eval dataset items + item evaluators"
```

---

### Task 6: Run-level confusion metrics

**Files:**
- Modify: `src/ticker_news/evals/classify_eval.py`
- Modify: `tests/evals/test_classify_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evals/test_classify_eval.py`:

```python
from types import SimpleNamespace

from ticker_news.evals.classify_eval import act_metrics_run_evaluator


def _item_result(expected_act, predicted_act):
    return SimpleNamespace(
        item={"expected_output": {"act": expected_act}},
        output={"predicted": "x", "act": predicted_act} if predicted_act else None,
    )


def _by_name(evaluations):
    return {e.name: e for e in evaluations}


class TestRunEvaluator:
    def test_confusion_metrics(self):
        # TP=2 FP=1 FN=1 TN=2 -> acc 4/6, precision 2/3, recall 2/3
        results = [
            _item_result("YES", "YES"), _item_result("YES", "YES"),  # TP
            _item_result("NO", "YES"),                               # FP
            _item_result("YES", "NO"),                               # FN
            _item_result("NO", "NO"), _item_result("NO", "NO"),      # TN
        ]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert evals["act_accuracy_avg"].value == pytest.approx(4 / 6)
        assert evals["act_precision"].value == pytest.approx(2 / 3)
        assert evals["act_recall"].value == pytest.approx(2 / 3)
        assert evals["act_f1"].value == pytest.approx(2 / 3)
        assert "TP=2 FP=1 FN=1 TN=2" in evals["act_accuracy_avg"].comment

    def test_no_yes_predictions_skips_precision(self):
        results = [_item_result("YES", "NO"), _item_result("NO", "NO")]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert "act_precision" not in evals
        assert "no YES predictions" in evals["act_precision_skip"].value
        assert evals["act_recall"].value == 0.0
        assert "act_f1" not in evals

    def test_no_yes_items_skips_recall(self):
        results = [_item_result("NO", "YES"), _item_result("NO", "NO")]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert "act_recall" not in evals
        assert "no YES items" in evals["act_recall_skip"].value
        assert evals["act_precision"].value == 0.0

    def test_failed_items_are_counted_as_wrong(self):
        # an errored task (output=None) on a YES item counts as FN
        results = [_item_result("YES", None), _item_result("YES", "YES")]
        evals = _by_name(act_metrics_run_evaluator(item_results=results))
        assert evals["act_recall"].value == pytest.approx(0.5)

    def test_empty_results_skip_everything(self):
        evals = _by_name(act_metrics_run_evaluator(item_results=[]))
        assert set(evals) == {"act_metrics_skip"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_classify_eval.py::TestRunEvaluator -v`
Expected: FAIL — `ImportError: cannot import name 'act_metrics_run_evaluator'`

- [ ] **Step 3: Write the implementation**

Append to `src/ticker_news/evals/classify_eval.py`:

```python
def _expected_act(item) -> str | None:
    expected = item.get("expected_output") if isinstance(item, dict) else item.expected_output
    return (expected or {}).get("act")


def act_metrics_run_evaluator(*, item_results, **kwargs) -> list[Evaluation]:
    """Run-level confusion metrics for the YES class.

    The GT is imbalanced (42 YES / 98 NO) — precision/recall/F1 keep an
    always-NO classifier from looking good. A task that errored (output
    None) counts as a miss on its side of the matrix rather than vanishing
    from the denominator.
    """
    tp = fp = fn = tn = 0
    for r in item_results:
        expected = _expected_act(r.item)
        predicted = (r.output or {}).get("act")
        if expected == "YES":
            tp, fn = (tp + 1, fn) if predicted == "YES" else (tp, fn + 1)
        elif expected == "NO":
            fp, tn = (fp + 1, tn) if predicted == "YES" else (fp, tn + 1)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/ticker_news/evals/classify_eval.py tests/evals/test_classify_eval.py
git commit -m "feat: run-level precision/recall/F1 for classify eval"
```

---

### Task 7: Async task + run_eval orchestrator

**Files:**
- Modify: `src/ticker_news/evals/classify_eval.py`
- Modify: `tests/evals/test_classify_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evals/test_classify_eval.py`:

```python
import asyncio

from ticker_news.classification.variants import (
    BinaryClassification,
    VariantRunner,
    is_act_binary,
)
from ticker_news.evals import classify_eval
from ticker_news.evals.classify_eval import make_task


class StubChain:
    def __init__(self, *verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    async def ainvoke(self, inputs, config=None):
        self.calls.append(inputs)
        return self._verdicts.pop(0)


class TestMakeTask:
    def test_task_reads_db_and_returns_verdict_dict(self, monkeypatch):
        conn = FakeConn(rows=[("Title", "Body text")])
        monkeypatch.setattr(classify_eval, "connect_eval", lambda dsn: conn)
        conn.close = lambda: None
        runner = VariantRunner(
            lite=StubChain(BinaryClassification(label="real news", confidence=0.7,
                                                reason="earnings")),
            confirm=None, is_act=is_act_binary, label_of=lambda v: v.label,
        )
        task = make_task(runner, dsn=None)
        out = asyncio.run(task(item={"input": {"article_id": 595, "title": "T"}}))
        assert out == {
            "predicted": "real news", "act": "YES", "confidence": 0.7,
            "reason": "earnings", "confirmed": False,
        }
        # the SELECT was parametrized on the article id
        assert any(p == (595,) for _, p in conn.executed)

    def test_task_raises_for_missing_article(self, monkeypatch):
        conn = FakeConn(rows=[])
        conn.close = lambda: None
        monkeypatch.setattr(classify_eval, "connect_eval", lambda dsn: conn)
        runner = VariantRunner(
            lite=StubChain(), confirm=None,
            is_act=is_act_binary, label_of=lambda v: v.label,
        )
        task = make_task(runner, dsn=None)
        with pytest.raises(ValueError, match="595"):
            asyncio.run(task(item={"input": {"article_id": 595, "title": "T"}}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_classify_eval.py::TestMakeTask -v`
Expected: FAIL — `ImportError: cannot import name 'make_task'`

- [ ] **Step 3: Write the implementation**

Append to `src/ticker_news/evals/classify_eval.py`:

```python
import asyncio
from datetime import datetime

from ticker_news.evals.pipeline_eval import connect_eval

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_classify_eval.py -v`
Expected: all PASS

- [ ] **Step 5: Run the whole offline suite to catch regressions**

Run: `.venv\Scripts\python.exe -m pytest -m "not db and not integration" -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/ticker_news/evals/classify_eval.py tests/evals/test_classify_eval.py
git commit -m "feat: classify eval task + experiment orchestrator"
```

---

### Task 8: CLI `ticker-news eval classify`

**Files:**
- Modify: `src/ticker_news/cli.py:557-589` (eval sub-app)
- Modify: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

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
        "eval", "classify", "--variant", "binary", "--mode", "lite",
        "--gt-csv", "gt.csv", "--ids", "595,14682", "--dsn", "postgresql://x",
        "--run-name", "tuning-1",
    ])
    assert result.exit_code == 0, result.output
    assert captured["variants"] == ("binary",)
    assert captured["mode"] == "lite"
    assert captured["gt_csv"] == "gt.csv"
    assert captured["ids"] == [595, 14682]
    assert captured["dsn"] == "postgresql://x"
    assert captured["run_name"] == "tuning-1"


def test_eval_classify_defaults_to_both_two_pass(monkeypatch):
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
    assert captured["mode"] == "two-pass"
    assert captured["dataset_name"] == "classify-ground-truth"
    assert captured["ids"] is None


def test_eval_classify_rejects_bad_variant():
    result = runner.invoke(cli.app, ["eval", "classify", "--variant", "ternary"])
    assert result.exit_code != 0


def test_eval_classify_rejects_bad_mode():
    result = runner.invoke(cli.app, ["eval", "classify", "--mode", "warp"])
    assert result.exit_code != 0


def test_eval_classify_rejects_bad_ids():
    result = runner.invoke(cli.app, ["eval", "classify", "--ids", "1,x"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py -k eval_classify -v`
Expected: FAIL — `eval classify` command does not exist (exit code 2 on the passing-args tests)

- [ ] **Step 3: Write the implementation**

In `src/ticker_news/cli.py`, first extract the duplicated cp1252-safe echo (used by `eval_pipeline` at lines 584-588) into a helper above the eval sub-app, and use it in `eval_pipeline`:

```python
def _echo_summary(summary: str) -> None:
    try:
        typer.echo(summary)
    except UnicodeEncodeError:  # Windows cp1252 console vs emoji in format()
        typer.echo(summary.encode("ascii", "backslashreplace").decode("ascii"))
```

Then add the command after `eval_pipeline`:

```python
@eval_app.command("classify")
def eval_classify(
    variant: str = typer.Option("both", "--variant", help="both | binary | finegrained."),
    mode: str = typer.Option("two-pass", "--mode", help="two-pass | lite | flash."),
    gt_csv: str | None = typer.Option(None, "--gt-csv", help="Seed/refresh the Langfuse dataset from this ground-truth csv (article id, header, Act_GT)."),
    dataset: str = typer.Option("classify-ground-truth", "--dataset", help="Langfuse dataset name."),
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated article ids: run only this dataset subset (fast prompt iteration)."),
    dsn: str | None = typer.Option(None, "--dsn", help="Target DB DSN (default: DATABASE_URL)."),
    run_name: str | None = typer.Option(None, "--run-name", help="Experiment run name (default: <variant>-<mode>-<timestamp>)."),
) -> None:
    """Compare classification prompt variants against the ground-truth labels."""
    from ticker_news.classification.variants import MODES, VARIANTS
    from ticker_news.evals import classify_eval

    if variant not in ("both", *VARIANTS):
        raise typer.BadParameter(f"--variant must be one of: both, {', '.join(VARIANTS)}")
    if mode not in MODES:
        raise typer.BadParameter(f"--mode must be one of: {', '.join(MODES)}")
    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else None
    except ValueError as exc:
        raise typer.BadParameter(f"--ids must be comma-separated integers: {exc}")
    variants = VARIANTS if variant == "both" else (variant,)
    try:
        results = classify_eval.run_eval(
            variants, mode=mode, dataset_name=dataset, gt_csv=gt_csv,
            dsn=dsn, run_name=run_name, ids=id_list,
        )
    except SystemExit as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    for variant_name, result in results:
        typer.echo(f"=== {variant_name} ===")
        _echo_summary(result.format())
```

And replace the tail of `eval_pipeline` (the existing try/except UnicodeEncodeError block) with:

```python
    _echo_summary(result.format())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py -v`
Expected: all PASS (including the pre-existing CLI tests)

- [ ] **Step 5: Commit**

```bash
git add src/ticker_news/cli.py tests/test_cli.py
git commit -m "feat: ticker-news eval classify command"
```

---

### Task 9: `reset_article` keep-semantics + parse helpers

**Files:**
- Modify: `src/ticker_news/evals/pipeline_eval.py:117-136` (`reset_article`)
- Modify: `tests/evals/test_pipeline_eval.py` (append; existing reset test stays valid)

- [ ] **Step 1: Write the failing tests**

Append to `tests/evals/test_pipeline_eval.py`:

```python
from ticker_news.evals.pipeline_eval import (
    SKIPPABLE_STAGES,
    parse_ids_file,
    parse_skip_stages,
)


class TestResetArticleKeep:
    def _update_sql(self, conn):
        updates = [s for s, _ in conn.executed if s.startswith("UPDATE")]
        return updates[0] if updates else ""

    def test_keep_embed_preserves_embedding(self):
        conn = FakeConn()
        reset_article(conn, 20512, keep=frozenset({"embed"}))
        sql = self._update_sql(conn)
        assert "embedding = NULL" not in sql
        assert "category = NULL" in sql
        assert any("article_insights" in s for s, _ in conn.executed)

    def test_keep_insights_preserves_rows_and_stamp(self):
        conn = FakeConn()
        reset_article(conn, 20512, keep=frozenset({"insights"}))
        assert not any("article_insights" in s for s, _ in conn.executed)
        sql = self._update_sql(conn)
        assert "insights_extracted_at = NULL" not in sql
        assert "embedding = NULL" in sql

    def test_keep_all_still_deletes_sentiment(self):
        conn = FakeConn()
        reset_article(conn, 20512, keep=frozenset(SKIPPABLE_STAGES))
        sqls = [s for s, _ in conn.executed if s != "COMMIT"]
        assert len(sqls) == 1
        assert "article_sentiment" in sqls[0]

    def test_default_resets_everything_as_before(self):
        conn = FakeConn()
        reset_article(conn, 20512)
        sql = self._update_sql(conn)
        for col in ("embedding", "category", "category_reason", "primary_ticker",
                    "primary_segment", "more_tickers", "more_segments",
                    "insights_extracted_at"):
            assert f"{col} = NULL" in sql


class TestParseSkipStages:
    def test_parses_comma_list(self):
        assert parse_skip_stages("embed, insights") == frozenset({"embed", "insights"})

    def test_none_and_empty_mean_no_skips(self):
        assert parse_skip_stages(None) == frozenset()
        assert parse_skip_stages("") == frozenset()

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="sentiment"):
            parse_skip_stages("embed,sentiment")


class TestParseIdsFile:
    def test_parses_one_id_per_line(self, tmp_path):
        f = tmp_path / "ids.csv"
        f.write_text("671\n685\n\n694\n", encoding="utf-8")
        assert parse_ids_file(f) == [671, 685, 694]

    def test_tolerates_bom_and_trailing_commas(self, tmp_path):
        f = tmp_path / "ids.csv"
        f.write_text("671,\n685\n", encoding="utf-8-sig")
        assert parse_ids_file(f) == [671, 685]

    def test_non_integer_line_raises_with_line_number(self, tmp_path):
        f = tmp_path / "ids.csv"
        f.write_text("671\nabc\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 2.*abc"):
            parse_ids_file(f)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_pipeline_eval.py -v`
Expected: FAIL — `ImportError: cannot import name 'SKIPPABLE_STAGES'`

- [ ] **Step 3: Write the implementation**

In `src/ticker_news/evals/pipeline_eval.py`, add near the top (after the imports):

```python
from pathlib import Path

# Stages whose outputs may be preserved across eval runs. Sentiment is never
# skippable - the verdict is what the experiment scores.
SKIPPABLE_STAGES = ("embed", "classify", "tag", "insights")

# stage -> articles columns nulled when the stage is NOT kept.
_STAGE_COLUMNS = {
    "embed": ("embedding",),
    "classify": ("category", "category_reason"),
    "tag": ("primary_ticker", "primary_segment", "more_tickers", "more_segments"),
    "insights": ("insights_extracted_at",),
}


def parse_skip_stages(raw: str | None) -> frozenset[str]:
    """Validate a comma-separated --skip-stages value. ValueError on unknowns."""
    if not raw:
        return frozenset()
    stages = {s.strip() for s in raw.split(",") if s.strip()}
    unknown = stages - set(SKIPPABLE_STAGES)
    if unknown:
        raise ValueError(
            f"unknown stage(s): {', '.join(sorted(unknown))} "
            f"(skippable: {', '.join(SKIPPABLE_STAGES)})"
        )
    return frozenset(stages)


def parse_ids_file(path: str | Path) -> list[int]:
    """One article id per line (utf-8, BOM/trailing-comma tolerant)."""
    ids: list[int] = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for lineno, line in enumerate(text.splitlines(), start=1):
        token = line.strip().rstrip(",").strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"{path} line {lineno}: not an article id: {token!r}")
        ids.append(int(token))
    return ids
```

Replace `reset_article` with:

```python
def reset_article(
    conn: psycopg.Connection, article_id: int, keep: frozenset[str] = frozenset()
) -> None:
    """Clear derived fields so the idempotent stage adapters re-run.

    Stages named in `keep` retain their outputs, so the corresponding
    adapters no-op naturally (no LLM/API cost). Scraped content is never
    touched; the sentiment verdict is always cleared (it is what the eval
    scores). One transaction: an eval article is never left half-reset.
    """
    conn.execute(
        "DELETE FROM public.article_sentiment WHERE article_id = %s", (article_id,)
    )
    if "insights" not in keep:
        conn.execute(
            "DELETE FROM public.article_insights WHERE article_id = %s", (article_id,)
        )
    columns = [
        col for stage in SKIPPABLE_STAGES if stage not in keep
        for col in _STAGE_COLUMNS[stage]
    ]
    if columns:
        assignments = ", ".join(f"{col} = NULL" for col in columns)
        conn.execute(
            f"UPDATE public.articles SET {assignments} WHERE id = %s",
            (article_id,),
        )
    conn.commit()
```

Note: with the default `keep=frozenset()` the generated UPDATE lists the same
eight columns as before, so the existing `TestResetArticle` test keeps passing
(verify the column order matches `SKIPPABLE_STAGES` × `_STAGE_COLUMNS`:
embedding, category, category_reason, primary_ticker, primary_segment,
more_tickers, more_segments, insights_extracted_at — same as the old literal).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_pipeline_eval.py -v`
Expected: all PASS (old `TestResetArticle` included)

- [ ] **Step 5: Commit**

```bash
git add src/ticker_news/evals/pipeline_eval.py tests/evals/test_pipeline_eval.py
git commit -m "feat: keep-semantics reset + skip-stages/ids-file parsers"
```

---

### Task 10: Wire `--skip-stages` / `--ids-file` through run_eval and the CLI

**Files:**
- Modify: `src/ticker_news/evals/pipeline_eval.py` (`make_task`, `run_eval`)
- Modify: `src/ticker_news/cli.py:561-589` (`eval_pipeline`)
- Modify: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_eval_pipeline_passes_skip_stages_and_ids_file(monkeypatch, tmp_path):
    ids_file = tmp_path / "ids.csv"
    ids_file.write_text("671\n685\n", encoding="utf-8")
    captured = {}

    class FakeResult:
        def format(self):
            return "ok"

    def fake_run_eval(ids, **kwargs):
        captured["ids"] = ids
        captured.update(kwargs)
        return FakeResult()

    from ticker_news.evals import pipeline_eval
    monkeypatch.setattr(pipeline_eval, "run_eval", fake_run_eval)
    result = runner.invoke(cli.app, [
        "eval", "pipeline", "--ids", "20512", "--ids-file", str(ids_file),
        "--skip-stages", "embed",
    ])
    assert result.exit_code == 0, result.output
    assert captured["ids"] == [671, 685, 20512]   # union, sorted
    assert captured["skip_stages"] == frozenset({"embed"})


def test_eval_pipeline_rejects_unknown_skip_stage(monkeypatch):
    result = runner.invoke(cli.app, [
        "eval", "pipeline", "--ids", "1", "--skip-stages", "sentiment",
    ])
    assert result.exit_code != 0


def test_eval_pipeline_rejects_bad_ids_file(monkeypatch, tmp_path):
    ids_file = tmp_path / "ids.csv"
    ids_file.write_text("671\nabc\n", encoding="utf-8")
    result = runner.invoke(cli.app, [
        "eval", "pipeline", "--ids-file", str(ids_file),
    ])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py -k eval_pipeline -v`
Expected: FAIL — `--ids-file` / `--skip-stages` are unknown options (exit code 2)

- [ ] **Step 3: Wire through `pipeline_eval`**

In `src/ticker_news/evals/pipeline_eval.py`:

`make_task` gains a parameter and passes it to the reset (only the signature
and the `reset_article` call change):

```python
def make_task(dsn: str | None, skip_stages: frozenset[str] = frozenset()):
    ...
    def run_pipeline_task(*, item, **kwargs) -> dict:
        ...
        conn = connect_eval(dsn)
        try:
            reset_article(conn, article_id, keep=skip_stages)
            ...
```

(The stage calls themselves are unchanged — kept stages' adapters see their
output already present and no-op.)

`run_eval` gains `skip_stages` and records it in metadata:

```python
def run_eval(
    ids: list[int],
    *,
    dataset_name: str | None = None,
    dsn: str | None = None,
    run_name: str | None = None,
    skip_stages: frozenset[str] = frozenset(),
):
```

and inside, change the `common` dict:

```python
    metadata = {"entrypoint": "eval"}
    if skip_stages:
        metadata["skipped_stages"] = sorted(skip_stages)
    common = dict(
        name=EXPERIMENT_NAME,
        run_name=run_name,
        description=_DESCRIPTION,
        task=make_task(dsn, skip_stages),
        evaluators=[directional_agreement_evaluator, price_move_evaluator],
        run_evaluators=[avg_directional_agreement],
        # Sync tasks run serially in langfuse 4.7.1 (no to_thread); this only
        # caps the async evaluators. Kept low deliberately - each item already
        # fans out ~7 LLM calls inside the stages.
        max_concurrency=2,
        metadata=metadata,
    )
```

- [ ] **Step 4: Wire through the CLI**

In `src/ticker_news/cli.py`, extend `eval_pipeline`:

```python
@eval_app.command("pipeline")
def eval_pipeline(
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated article ids to force through the full eval pipeline."),
    ids_file: str | None = typer.Option(None, "--ids-file", help="File with one article id per line (unioned with --ids)."),
    dataset: str | None = typer.Option(None, "--dataset", help="Langfuse dataset name: upsert --ids as items, then run over the whole dataset."),
    dsn: str | None = typer.Option(None, "--dsn", help="Target DB DSN (default: DATABASE_URL)."),
    run_name: str | None = typer.Option(None, "--run-name", help="Experiment run name (default: auto-generated)."),
    skip_stages: str | None = typer.Option(None, "--skip-stages", help="Comma-separated stages whose stored outputs are reused instead of re-run: embed, classify, tag, insights."),
) -> None:
    """Re-run articles E2E through the pipeline; score verdicts against actual price moves."""
    from ticker_news.evals import pipeline_eval

    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else []
    except ValueError as exc:
        raise typer.BadParameter(f"--ids must be comma-separated integers: {exc}")
    if ids_file:
        try:
            id_list = sorted(set(id_list) | set(pipeline_eval.parse_ids_file(ids_file)))
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"--ids-file: {exc}")
    try:
        skip = pipeline_eval.parse_skip_stages(skip_stages)
    except ValueError as exc:
        raise typer.BadParameter(f"--skip-stages: {exc}")
    if not id_list and not dataset:
        raise typer.BadParameter("provide --ids/--ids-file, or --dataset with existing items")
    try:
        result = pipeline_eval.run_eval(
            id_list, dataset_name=dataset, dsn=dsn, run_name=run_name,
            skip_stages=skip,
        )
    except SystemExit as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    _echo_summary(result.format())
```

- [ ] **Step 5: Run the tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py tests/evals/ -v`
Expected: all PASS

- [ ] **Step 6: Run the full offline suite**

Run: `.venv\Scripts\python.exe -m pytest -m "not db and not integration" -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/ticker_news/evals/pipeline_eval.py src/ticker_news/cli.py tests/test_cli.py
git commit -m "feat: --skip-stages and --ids-file for eval pipeline"
```

---

### Task 11: Docs + live smoke verification

**Files:**
- Modify: `CLAUDE.md` (commands table — `eval classify` row, `eval pipeline` flags)
- No code changes; live verification against the shared DB + Langfuse.

- [ ] **Step 1: Update CLAUDE.md**

In the commands table, change the `eval pipeline` row and add `eval classify`:

```markdown
| `ticker-news eval pipeline --ids N[,..]` | E2E eval: re-run articles through every stage, score verdict vs realized price move as a Langfuse experiment (`--dataset` for run-over-run comparison, `--dsn` for the shared DB, `--ids-file` for an id-per-line file, `--skip-stages embed[,..]` to reuse stable stage outputs) |
| `ticker-news eval classify --gt-csv F` | Classification prompt eval: binary vs fine-grained variants scored against ground-truth ACT labels on a Langfuse dataset (`--variant`, `--mode lite\|flash\|two-pass`, `--ids` subset) |
```

- [ ] **Step 2: Push the new prompts to Langfuse**

Note: clear the stale `OPENAI_API_KEY` OS env var first if set (it overrides
`.env`): `Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue`

Run: `.venv\Scripts\python.exe -m ticker_news.cli prompts push`
(or `ticker-news prompts push` if the entry point is on PATH)
Expected: count includes the two new prompts (8 total). Verify
`classify-binary` and `classify-finegrained` appear in the Langfuse UI
under Prompts with the `production` label.

- [ ] **Step 3: Smoke-run the classify eval on a tiny subset**

Run (shared-DB DSN from `_inspect_shared_db.py`):

```powershell
.venv\Scripts\python.exe -m ticker_news.cli eval classify `
  --gt-csv "C:\Agents\final-project\classify\ground-truth-class.csv" `
  --ids 595,14682 --mode lite `
  --dsn "postgresql://robert:JAeNyQFZL6jahShNRKmc1U9RZfgU@localhost:15432/sharedproject"
```

Expected: two experiment summaries print (binary + finegrained), each over 2
items; in the Langfuse UI the `classify-ground-truth` dataset exists with 140
items, and the two runs show `act_accuracy` per item plus run-level metrics.

- [ ] **Step 4: Full GT-140 run (both variants, two-pass)**

```powershell
.venv\Scripts\python.exe -m ticker_news.cli eval classify `
  --dsn "postgresql://robert:JAeNyQFZL6jahShNRKmc1U9RZfgU@localhost:15432/sharedproject"
```

Expected: completes in minutes (async, concurrency 8, rate-limited at 8 rps);
both runs visible side-by-side in the dataset's Runs tab with
accuracy/precision/recall/F1.

- [ ] **Step 5: Smoke the fast E2E path on a few of the 100 ids**

```powershell
.venv\Scripts\python.exe -m ticker_news.cli eval pipeline `
  --ids 671,685 --skip-stages embed `
  --dataset pipeline-100-peaceful `
  --dsn "postgresql://robert:JAeNyQFZL6jahShNRKmc1U9RZfgU@localhost:15432/sharedproject"
```

Expected: run completes without any OpenAI embedding call (embedding
preserved); run metadata in Langfuse shows `skipped_stages: ["embed"]`.
(Full 100-article seeding via `--ids-file` is the user's call to run — it
re-runs classify/tag/insights/sentiment for 100 articles.)

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: eval classify command + eval pipeline skip flags"
```

---

## Self-Review Notes

- **Spec coverage:** variants module (Tasks 1–3), registry (Task 2), GT loader/build_items/evaluators/run metrics (Tasks 4–6), async task + orchestrator + subset `--ids` + failed-item warning (Task 7), CLI classify (Task 8), reset keep-semantics + parsers (Task 9), CLI/metadata wiring for skip-stages + ids-file (Task 10), docs + live verification incl. prompts push (Task 11). The spec's "Langfuse UI layout" section is realized by dataset+experiment naming in Task 7; "Error handling" maps to loud ValueError/SystemExit paths in Tasks 4, 5, 7, 9, 10.
- **Type consistency:** `VariantRunner(lite, confirm, is_act, label_of)` used identically in Tasks 3 and 7 tests; `parse_skip_stages`/`parse_ids_file` return `frozenset[str]`/`list[int]` and the CLI passes `skip_stages: frozenset` into `run_eval(..., skip_stages=...)`; `make_task(runner, dsn)` (classify) vs `make_task(dsn, skip_stages)` (pipeline) live in different modules.
- **Known judgment call:** errored items count as misses in the confusion matrix (Task 6) rather than being excluded — deliberate, so a crashing variant can't score better by failing.
