# Pipeline v2 — Plan 5: Langfuse Cloud Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-article Langfuse traces across the whole service — one root trace per article with a span per stage and every LLM generation nested under it (restoring the cost visibility dropped in Plan 2) — plus prompt management with in-repo fallbacks. **Langfuse Cloud** (user decision): keys in `.env`, no local containers.

**Architecture:** Phase 5 of `docs/superpowers/specs/2026-06-10-pipeline-v2-design.md` (Langfuse section), amended by the cloud decision. All wiring goes through one new module, `shared/observability.py`, and one hard rule: **everything no-ops when keys are absent** — the existing 121-test offline suite must pass unchanged with no Langfuse env configured, and the worker's behavior must be byte-identical in no-op mode. SDK v4 APIs (`langfuse` ≥4,<5): `Langfuse` client, `langfuse.langchain.CallbackHandler` (new import path), `create_trace_id(seed=url)` for deterministic re-run correlation, `propagate_attributes` (metadata values must be strings ≤200 chars — known SDK quirk), `flush()` on shutdown. Implementer rule throughout: the research notes may lag the installed SDK — verify import paths/signatures against the installed package; the tests are the contract.

**Stable observation names (eval contract, from the spec):** `process-article` (root), `scrape`, `embed`, `classify`, `tag`, `insights`, `sentiment` (stage spans), `analyst:<role>`, `synthesize`.

**Tech Stack:** langfuse ≥4,<5 (new dep). No docker changes.

**Branch:** `refactor/pipeline-v2`. Baseline: **121 passed, 1 skipped** offline; **12 passed** db.

---

### Task 1: dep, settings, `shared/observability.py`

**Files:**
- Modify: `pyproject.toml` (add `"langfuse>=4,<5",`)
- Modify: `src/ticker_news/shared/config.py` (3 fields)
- Create: `src/ticker_news/shared/observability.py`
- Create: `tests/shared/test_observability.py`
- Modify: `tests/shared/test_config.py` (new fields test)

- [ ] **Step 1: dep + settings**

Add the dep; `pip install -e ".[dev]"`; report installed langfuse version. Add to `AppSettings` (after the API keys):

```python
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )
```

Add to `tests/shared/test_config.py`:

```python
def test_langfuse_disabled_by_default(monkeypatch):
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = AppSettings(_env_file=None)
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host == "https://cloud.langfuse.com"
```

- [ ] **Step 2: observability module (TDD)**

`tests/shared/test_observability.py` — the no-op contract is the load-bearing part:

```python
from ticker_news.shared import observability as obs
from ticker_news.shared.config import get_settings


def _disable(monkeypatch):
    get_settings.cache_clear()
    obs.client.cache_clear()
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_disabled_without_keys(monkeypatch):
    _disable(monkeypatch)
    assert obs.enabled() is False
    assert obs.client() is None


def test_chain_config_empty_when_disabled(monkeypatch):
    _disable(monkeypatch)
    assert obs.chain_config() == {}
    assert obs.chain_config(run_name="x") == {"run_name": "x"}


def test_article_trace_noops_when_disabled(monkeypatch):
    _disable(monkeypatch)
    with obs.article_trace("https://example.com/a", ticker="NVDA") as t:
        assert t is None


def test_stage_span_noops_when_disabled(monkeypatch):
    _disable(monkeypatch)
    with obs.stage_span("classify") as s:
        assert s is None


def test_flush_noops_when_disabled(monkeypatch):
    _disable(monkeypatch)
    obs.flush()  # must not raise


def test_enabled_with_keys(monkeypatch):
    get_settings.cache_clear()
    obs.client.cache_clear()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert obs.enabled() is True
    obs.client.cache_clear()
    get_settings.cache_clear()
```

(`test_enabled_with_keys` checks only the gate — constructing the real client with fake keys is allowed but do NOT call the network in tests; if `Langfuse(...)` eagerly validates against the host on construction in the installed SDK, drop the client construction from `enabled()`'s path and report.)

`src/ticker_news/shared/observability.py`:

```python
"""Langfuse Cloud wiring. Every helper degrades to a no-op when keys are absent.

One trace per article; stage spans; LLM generations nest via the LangChain
CallbackHandler. Stable observation names are an eval contract:
process-article, scrape, embed, classify, tag, insights, sentiment,
analyst:<role>, synthesize.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache

from ticker_news.shared.config import get_settings

logger = logging.getLogger(__name__)


def enabled() -> bool:
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


@lru_cache(maxsize=1)
def client():
    """Singleton Langfuse client, or None when disabled."""
    if not enabled():
        return None
    from langfuse import Langfuse

    s = get_settings()
    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )


def chain_config(run_name: str | None = None) -> dict:
    """Per-invoke config for chains/graphs: Langfuse callbacks + optional run_name.

    Returns {} (or just the run_name) when disabled, so call sites can pass it
    unconditionally: chain.invoke(x, config=chain_config() or None).
    """
    cfg: dict = {}
    if enabled():
        from langfuse.langchain import CallbackHandler

        cfg["callbacks"] = [CallbackHandler()]
    if run_name:
        cfg["run_name"] = run_name
    return cfg


@contextmanager
def article_trace(url: str, *, ticker: str | None = None):
    """Root trace for one article moving through the pipeline.

    Deterministic trace id seeded from the URL so re-runs correlate.
    Yields the root span (or None when disabled).
    """
    c = client()
    if c is None:
        yield None
        return
    from langfuse import propagate_attributes

    trace_id = c.create_trace_id(seed=url)
    with c.start_as_current_observation(
        as_type="span", name="process-article",
        trace_context={"trace_id": trace_id},
    ) as root:
        root.update_trace(input={"url": url})
        metadata = {"url": url[:200]}
        if ticker:
            metadata["ticker"] = ticker
        with propagate_attributes(tags=["pipeline-v2"], metadata=metadata):
            yield root


@contextmanager
def stage_span(name: str):
    c = client()
    if c is None:
        yield None
        return
    with c.start_as_current_observation(as_type="span", name=name) as span:
        yield span


def flush() -> None:
    c = client()
    if c is not None:
        c.flush()
```

Implementer note: verify `create_trace_id`, `start_as_current_observation(trace_context=...)`, `update_trace`, and `propagate_attributes` against the INSTALLED v4 SDK (`pip show langfuse`, read its source). If a name differs (e.g. `update_trace` lives on the client, or `trace_context` has another spelling), adapt the module — the no-op tests are the contract; the enabled paths must merely be consistent with the real SDK. Report every adaptation.

- [ ] **Step 3: run + commit**

Offline → 128 passed, 1 skipped (121 + 6 obs + 1 config). Report real counts.

```bash
git add pyproject.toml src/ticker_news/shared tests/shared
git commit -m "feat: Langfuse cloud observability module with strict no-op fallback"
```
(Standing rule: no Co-Authored-By, no AI signatures, every commit.)

---

### Task 2: config passthrough into every chain and the graph

**Files:**
- Modify: `src/ticker_news/classification/chain.py` (classify_article gains `config=None`)
- Modify: `src/ticker_news/enrichment/insights.py` (generate_boxes gains `config=None`)
- Modify: `src/ticker_news/sentiment/graph.py` (judge_article gains `config=None`; analyst node run_name)
- Modify: `src/ticker_news/service/stages.py` (stage adapters build and pass `chain_config()`)
- Modify: `src/ticker_news/sentiment/batch.py` + `src/ticker_news/classification/pipeline.py` + `src/ticker_news/enrichment/insights.py` extract_all (batch paths pass config too)
- Modify: tests — `tests/sentiment/test_graph.py` (+1 config test), `tests/classification/test_chain.py` (+1), `tests/enrichment/` insights chain test if one exists (+1 where it fits)

- [ ] **Step 1: signatures**

- `classify_article(title, content, *, lite=None, confirm=None, config=None)` — both `.invoke(inputs)` calls become `.invoke(inputs, config=config)`.
- `generate_boxes(article_text, *, chain=None, config=None)` — `chain.invoke(prompt, config=config)`.
- `judge_article(article, *, graph=None, config=None)` — `graph.invoke(state, config=config)`.
- In the sentiment graph's `analyst` node: `analyst_llm.invoke(prompt, config={"run_name": f"analyst:{payload['role']}"})` and in `synthesize`: `judge.invoke(prompt, config={"run_name": "synthesize"})` (LangGraph propagates the parent invoke's callbacks into node context automatically; the inner config adds only the run_name — verify merging works on the installed version, tests below are the contract).

New tests (exact):

`tests/sentiment/test_graph.py` append:

```python
def test_config_reaches_graph_and_run_names_set():
    analyst, judge, prompts, _ = _fakes()
    graph = build_graph(analyst_llm=analyst, judge=judge)
    verdict, _ = judge_article(ARTICLE, graph=graph, config={"metadata": {"k": "v"}})
    assert verdict.action == "buy"
```

(`RunnableLambda` accepts a config kwarg transparently; the real assertion is no-crash passthrough. For run_name, add to `_fakes()`'s analyst lambda a capture of config if feasible on the installed langchain — if RunnableLambda fakes can't see the inner config cleanly, keep the no-crash test and verify run_name manually via SDK source; report.)

`tests/classification/test_chain.py` append:

```python
def test_config_passes_through_to_chains():
    captured = {}

    class CfgChain:
        def invoke(self, inputs, config=None):
            captured["config"] = config
            return Classification(category="other")

    classify_article("T", "body", lite=CfgChain(), config={"callbacks": []})
    assert captured["config"] == {"callbacks": []}
```

- [ ] **Step 2: call sites**

- `stages.py`: `classify_stage` → `classify_article(title, content or "", config=obs.chain_config() or None)`; `insights_stage` → `generate_boxes(content, config=obs.chain_config() or None)`; `sentiment_stage` → `judge_article(article, config=obs.chain_config() or None)` (import `from ticker_news.shared import observability as obs`).
- Batch paths (`classification/pipeline.py classify_all`, `insights.py extract_all`, `sentiment/batch.py run_batch`): same pattern at their per-article call sites.

- [ ] **Step 3: run + commit**

Offline → 130 passed, 1 skipped (128 + 2). Db → 12. Report real.

```bash
git add -A -- src tests
git commit -m "feat: thread Langfuse callback config through chains and the sentiment graph"
```

---

### Task 3: worker tracing + flush

**Files:**
- Modify: `src/ticker_news/service/worker.py`
- Modify: `tests/service/test_worker.py` (no-op guarantee: existing tests must pass UNCHANGED; +1 new test)

- [ ] **Step 1: wrap process_article**

In `worker.py` (import `from ticker_news.shared import observability as obs`):

```python
async def process_article(job, runners, queue=jobs, *, conn) -> bool:
    stage = job.stage
    ticker = job.tickers[0] if job.tickers else None
    with obs.article_trace(job.article_url, ticker=ticker):
        try:
            while stage != DONE:
                runner = runners[stage]
                with obs.stage_span(stage):
                    result = await _run_stage(runner, job)
                ...  # existing advance/short-circuit logic UNCHANGED inside the loop
        except Exception as exc:
            ...  # existing fail paths UNCHANGED
```

Constraint: the context managers are sync and cheap; entering them on the event loop is fine (span start is local; export is batched/background). The existing four worker tests and the e2e db test must pass WITHOUT modification — that is the no-op proof. Add one test:

```python
async def test_process_article_unchanged_under_disabled_observability(monkeypatch):
    # belt-and-braces: explicitly disabled, full chain still runs in order
    from ticker_news.shared.config import get_settings
    get_settings.cache_clear()
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    ran = []
    runners = {s: (lambda job, _s=s: ran.append(_s)) for s in
               ["embed", "classify", "tag", "insights", "sentiment"]}
    q = FakeQueue()
    await worker.process_article(_job(stage="embed"), runners, q, conn=None)
    assert ran == ["embed", "classify", "tag", "insights", "sentiment"]
```

- [ ] **Step 2: flush on shutdown**

In `serve()`'s outer `finally`: add `obs.flush()` (after fetcher.aclose()). In `sentiment/batch.py run_batch`'s `finally`: `obs.flush()`. Same in `classification/pipeline.py classify_all` and `insights.py extract_all` finallys (cheap no-op when disabled).

- [ ] **Step 3: run + commit**

Offline → 131 passed, 1 skipped. Db → 12 (e2e unchanged). Report real.

```bash
git add -A -- src tests
git commit -m "feat: per-article Langfuse traces with stage spans and shutdown flush"
```

---

### Task 4: prompt management with in-repo fallbacks

**Files:**
- Create: `src/ticker_news/shared/prompts.py`
- Modify: `src/ticker_news/classification/chain.py`, `src/ticker_news/enrichment/insights.py`, `src/ticker_news/sentiment/analysts.py` (fetch-with-fallback at build/render time)
- Modify: `src/ticker_news/cli.py` (`prompts push` sub-typer)
- Create: `tests/shared/test_prompts.py`
- Modify: `tests/test_root_cli.py` (+1)

- [ ] **Step 1: shared/prompts.py (TDD)**

`tests/shared/test_prompts.py`:

```python
from ticker_news.shared import prompts
from ticker_news.shared.config import get_settings


def _disable(monkeypatch):
    get_settings.cache_clear()
    from ticker_news.shared import observability as obs
    obs.client.cache_clear()
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_fallback_when_disabled(monkeypatch):
    _disable(monkeypatch)
    assert prompts.get_prompt("classify-article", "FALLBACK {x}") == "FALLBACK {x}"


def test_registry_covers_all_llm_prompts(monkeypatch):
    _disable(monkeypatch)
    reg = prompts.registry()
    assert set(reg) == {
        "classify-article", "extract-insights",
        "analyst-fundamentals", "analyst-market_context",
        "analyst-historical_precedent", "synthesize-verdict",
    }
    assert all(isinstance(v, str) and v for v in reg.values())
```

`src/ticker_news/shared/prompts.py`:

```python
"""Prompt management: Langfuse-versioned with committed in-repo fallbacks.

The fallback IS the source of truth in the repo; Langfuse holds versioned,
labeled copies for A/B and eval runs. The service boots fine with Langfuse
down or disabled.
"""

from __future__ import annotations

import logging

from ticker_news.shared.observability import client

logger = logging.getLogger(__name__)

PROMPT_LABEL = "production"


def get_prompt(name: str, fallback: str) -> str:
    """Langfuse prompt text (label=production) or the in-repo fallback."""
    c = client()
    if c is None:
        return fallback
    try:
        return c.get_prompt(name, label=PROMPT_LABEL).prompt
    except Exception as exc:
        logger.warning("langfuse prompt %r unavailable (%r); using fallback", name, exc)
        return fallback


def registry() -> dict[str, str]:
    """name -> in-repo fallback text, for `ticker-news prompts push`."""
    from ticker_news.classification.chain import PROMPT_TEMPLATE as classify_prompt
    from ticker_news.enrichment.insights import PROMPT_TEMPLATE as insights_prompt
    from ticker_news.sentiment.analysts import ANALYST_PROMPTS, SYNTHESIS_PROMPT

    reg = {
        "classify-article": classify_prompt,
        "extract-insights": insights_prompt,
        "synthesize-verdict": SYNTHESIS_PROMPT,
    }
    for role, prompt in ANALYST_PROMPTS.items():
        reg[f"analyst-{role}"] = prompt
    return reg


def push_all() -> int:
    """Upsert every registry prompt to Langfuse with the production label."""
    c = client()
    if c is None:
        raise SystemExit("Langfuse keys not configured (LANGFUSE_PUBLIC_KEY/SECRET_KEY).")
    count = 0
    for name, text in registry().items():
        c.create_prompt(name=name, prompt=text, labels=[PROMPT_LABEL])
        count += 1
    return count
```

(Implementer: verify `create_prompt`/`get_prompt` signatures on the installed SDK; `get_prompt` caches client-side with a TTL by default — keep that default.)

- [ ] **Step 2: consume at build/render time**

- `classification/chain.py build_classifier`: `ChatPromptTemplate.from_template(get_prompt("classify-article", PROMPT_TEMPLATE))`.
- `insights.py _box_chain`/generate_boxes: format string comes from `get_prompt("extract-insights", PROMPT_TEMPLATE)`.
- `analysts.py render_analyst`: template = `get_prompt(f"analyst-{role}", ANALYST_PROMPTS[role])`; `render_synthesis`: `get_prompt("synthesize-verdict", SYNTHESIS_PROMPT)`.
- NOTE: chains are lru_cached → prompt updates need a process restart; renders (analysts) fetch per call but Langfuse's client cache (TTL) absorbs it. Document both in a comment. ALL existing tests must keep passing — in disabled mode get_prompt returns the fallback, which is the previous behavior exactly.

- [ ] **Step 3: CLI**

`tests/test_root_cli.py`:

```python
def test_prompts_push_command(monkeypatch):
    monkeypatch.setattr("ticker_news.shared.prompts.push_all", lambda: 6)
    result = runner.invoke(cli.app, ["prompts", "push"])
    assert result.exit_code == 0, result.output
    assert "6" in result.output
```

`cli.py`:

```python
prompts_app = typer.Typer(help="Manage Langfuse prompt versions.")
app.add_typer(prompts_app, name="prompts")


@prompts_app.command("push")
def prompts_push() -> None:
    """Upsert the in-repo prompts to Langfuse with the production label."""
    from ticker_news.shared import prompts as prompts_mod

    n = prompts_mod.push_all()
    typer.echo(f"pushed {n} prompt(s)")
```

- [ ] **Step 4: run + commit**

Offline → 134 passed, 1 skipped (131 + 2 prompts + 1 CLI). Db → 12. Report real.

```bash
git add -A -- src tests
git commit -m "feat: Langfuse prompt management with in-repo fallbacks and prompts push CLI"
```

---

### Task 5: verification sweep + push

- [ ] Full offline + db suites — real counts (no Langfuse env set: this IS the no-op proof at scale).
- [ ] `ticker-news --help` (12 commands incl. prompts) + each `--help` exits 0.
- [ ] Lazy imports: `python -X importtime -c "import ticker_news.cli" 2>&1 | Select-String "langfuse|langchain|langgraph|google"` → no output.
- [ ] If `.env` contains LANGFUSE keys (check for the file/vars — do NOT print secret values): run a 1-article smoke (`ticker-news sentiment --limit 1` against the real DB IF it has data; it has 0 articles locally, so instead just `python -c` constructing the client and calling `client().auth_check()` if the SDK has it) and report whether a trace/auth round-trip succeeded. If no keys: state that cloud verification is pending user setup and list the exact .env lines the user must add.
- [ ] `git push`. No PR.

---

## Out of scope (Plan 6)

- research/ port, legacy deletion, CSV-backfill provider sentiment, CLAUDE.md update (must document LANGFUSE_* env vars + the prompts workflow), final PR.
- Eval datasets/experiments/LLM-as-judge (next milestone after MVP; this phase delivers the trace/naming/prompt substrate they need).
