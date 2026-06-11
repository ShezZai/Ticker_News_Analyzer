# E2E Pipeline Eval (Langfuse Experiment) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `ticker-news eval pipeline` command that re-runs articles through the real post-scrape stage chain against the shared DB and scores the resulting buy/sell verdict against the actual price move (entry at publication, exit at that day's regular close), as a Langfuse experiment.

**Architecture:** New `src/ticker_news/evals/pipeline_eval.py` module. The experiment task resets an article's derived fields and re-runs the unmodified `service/stages.py` adapters; two code evaluators fetch prices via `research/ticker_scan.py` (`fetch_prices` + `simulate`) and emit `directional_agreement` (1/0/None) and `price_move_pct` scores; a run-level evaluator averages agreement. Orchestrated by `langfuse.run_experiment` (local-data mode) or `dataset.run_experiment` (Langfuse dataset mode, run-over-run comparison).

**Tech Stack:** langfuse v4 experiment runner, psycopg 3 + pgvector, Typer CLI, existing Gemini/OpenAI stage code, Massive API price data.

**Spec:** `docs/superpowers/specs/2026-06-11-e2e-pipeline-eval-design.md`

**Reference — existing interfaces used (do not modify these files):**

- `ticker_news.service.stages`: `embed_stage(conn, url)`, `classify_stage(conn, url) -> str|None`, `tag_stage(conn, url, tag_ctx)`, `insights_stage(conn, url, tag_ctx)`, `sentiment_stage(conn, url) -> dict|None` (returns `{"ticker", "action", "confidence"}` or None when skipped), `TagContext.load(conn)`. All adapters skip silently when their output already exists — that's why the eval resets first.
- `ticker_news.research.ticker_scan`: `MARKET_TZ` (America/New_York), `fetch_prices(ticker, frm, to, key=None) -> TickerPrices`, `simulate(article, prices, include_after_hours=False) -> dict|None` where `article` needs `{"id", "ticker", "published_et"(aware dt)}` and the result has `"gain_pct"`. Raises `RuntimeError`/`ScanError` on API problems.
- `ticker_news.sentiment.store`: `ensure_schema(conn)` creates `article_sentiment`.
- `ticker_news.shared.observability` (`obs`): `client() -> Langfuse|None` (None when keys absent), `stage_span(name)` context manager, `flush()`.
- `ticker_news.shared.config.get_settings()`: `.database_url`, `.massive_api_key`, `.google_api_key`, `.openai_api_key`, `.langfuse_public_key`, `.langfuse_secret_key`.
- langfuse v4 (verified against installed 4.7.1): `Langfuse.run_experiment(*, name, run_name=None, description=None, data, task, evaluators=[], run_evaluators=[], max_concurrency=50, metadata=None)`; task signature `def task(*, item, **kwargs)` — `item` is a dict (`item["input"]`) for local data, a `DatasetItem` (`item.input`) for dataset runs; evaluator signature `def ev(*, input, output, expected_output=None, metadata=None, **kwargs) -> Evaluation`; run-evaluator `def rev(*, item_results, **kwargs)` where each result has `.evaluations`; `Langfuse.create_dataset(*, name, description=None, ...)`, `Langfuse.create_dataset_item(*, dataset_name, input=None, metadata=None, id=None, ...)` (deterministic `id` ⇒ upsert), `Langfuse.get_dataset(name)`, `dataset.run_experiment(...)` (same kwargs minus `data`).

**File structure:**

- Create: `src/ticker_news/evals/__init__.py` — empty package marker
- Create: `src/ticker_news/evals/pipeline_eval.py` — everything: schema healing, item building, reset, task, scoring, evaluators, orchestrator
- Create: `tests/evals/test_pipeline_eval.py` — offline unit tests (no DB, no network, no markers)
- Modify: `src/ticker_news/cli.py` — add `eval` sub-app with `pipeline` command (after the `jobs` sub-app, end of file)
- Modify: `CLAUDE.md` — one row in the commands table

---

### Task 1: Package scaffolding + pure scoring function

**Files:**
- Create: `src/ticker_news/evals/__init__.py`
- Create: `src/ticker_news/evals/pipeline_eval.py`
- Create: `tests/evals/test_pipeline_eval.py`

- [ ] **Step 1: Write the failing tests for the scoring matrix**

Create `tests/evals/test_pipeline_eval.py`:

```python
"""Offline unit tests for the E2E pipeline eval. No DB, no network."""

from ticker_news.evals.pipeline_eval import score_directional


class TestScoreDirectional:
    def test_buy_and_price_up_agrees(self):
        value, comment = score_directional("buy", 2.5)
        assert value == 1.0
        assert "agree" in comment

    def test_buy_and_price_down_disagrees(self):
        value, comment = score_directional("buy", -1.2)
        assert value == 0.0
        assert "disagree" in comment

    def test_buy_and_flat_price_disagrees(self):
        value, _ = score_directional("buy", 0.0)
        assert value == 0.0

    def test_sell_and_price_down_agrees(self):
        value, _ = score_directional("sell", -3.0)
        assert value == 1.0

    def test_sell_and_price_up_disagrees(self):
        value, _ = score_directional("sell", 1.7)
        assert value == 0.0

    def test_hold_is_excluded(self):
        value, comment = score_directional("hold", 2.0)
        assert value is None
        assert "hold" in comment

    def test_no_verdict_is_excluded_with_reason(self):
        value, comment = score_directional(None, None, skip_reason="category=recap/review")
        assert value is None
        assert "category=recap/review" in comment

    def test_missing_price_data_is_excluded(self):
        value, comment = score_directional("buy", None, skip_reason="no tradeable entry/exit bar")
        assert value is None
        assert "no price data" in comment

    def test_unknown_action_is_excluded(self):
        value, comment = score_directional("strong-buy", 1.0)
        assert value is None
        assert "strong-buy" in comment
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_pipeline_eval.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'ticker_news.evals'`

- [ ] **Step 3: Create the package and the scoring function**

Create `src/ticker_news/evals/__init__.py` (empty file).

Create `src/ticker_news/evals/pipeline_eval.py`:

```python
"""E2E pipeline eval: re-run articles through the real stage chain against the
shared DB, then score the verdict against the actual price move as a Langfuse
experiment.

Scoring is directional agreement: buy+up = 1, sell+down = 1, wrong direction
= 0; hold / no-verdict / no-price-data are excluded (value None) with an
explanatory comment. The raw entry->close move is recorded as a second score.

Design: docs/superpowers/specs/2026-06-11-e2e-pipeline-eval-design.md
"""

from __future__ import annotations


def score_directional(
    action: str | None, gain_pct: float | None, *, skip_reason: str | None = None
) -> tuple[float | None, str]:
    """Directional-agreement score for one verdict vs the realized move.

    Returns (value, comment) where value is 1.0 / 0.0, or None when the item
    cannot be scored (hold, no verdict, no price data) — None scores are
    excluded from Langfuse aggregates by the run evaluator.
    """
    if action is None:
        return None, f"no verdict ({skip_reason or 'sentiment skipped'})"
    if action == "hold":
        return None, "hold verdict - no direction to verify"
    if action not in ("buy", "sell"):
        return None, f"unknown action '{action}'"
    if gain_pct is None:
        return None, f"no price data ({skip_reason or 'unknown'})"
    correct = gain_pct > 0 if action == "buy" else gain_pct < 0
    verdict = "agree" if correct else "disagree"
    return (1.0 if correct else 0.0), f"{action} with {gain_pct:+.2f}% by close -> {verdict}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_pipeline_eval.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```powershell
git add src/ticker_news/evals tests/evals
git commit -m "feat: evals package with directional-agreement scoring"
```

---

### Task 2: Schema healing, item building, article reset

**Files:**
- Modify: `src/ticker_news/evals/pipeline_eval.py`
- Modify: `tests/evals/test_pipeline_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evals/test_pipeline_eval.py`:

```python
from datetime import datetime, timezone

import pytest

from ticker_news.evals.pipeline_eval import build_items, reset_article


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    """Records every execute; returns canned rows for SELECTs."""

    def __init__(self, rows=None):
        self.executed: list[tuple[str, tuple | None]] = []
        self._rows = rows or []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return FakeCursor(self._rows)

    def commit(self):
        self.executed.append(("COMMIT", None))


PUBLISHED = datetime(2026, 5, 28, 11, 10, tzinfo=timezone.utc)


def _row(aid=20512, ticker="MRVL", published=PUBLISHED, status="ok", has_content=True):
    return (aid, f"https://example.com/{aid}", ticker, published, "Title", status, has_content)


class TestBuildItems:
    def test_builds_langfuse_local_items(self):
        conn = FakeConn(rows=[_row()])
        items = build_items(conn, [20512])
        assert items == [{
            "input": {
                "article_id": 20512,
                "url": "https://example.com/20512",
                "published_utc": "2026-05-28T11:10:00+00:00",
                "title": "Title",
            },
            "metadata": {"seed_ticker": "MRVL"},
        }]

    def test_missing_id_raises(self):
        conn = FakeConn(rows=[_row()])
        with pytest.raises(ValueError, match="not found.*99999"):
            build_items(conn, [20512, 99999])

    def test_article_without_content_raises(self):
        conn = FakeConn(rows=[_row(status="empty", has_content=False)])
        with pytest.raises(ValueError, match="no scraped content"):
            build_items(conn, [20512])

    def test_article_without_published_raises(self):
        conn = FakeConn(rows=[_row(published=None)])
        with pytest.raises(ValueError, match="published_utc"):
            build_items(conn, [20512])


class TestResetArticle:
    def test_clears_derived_fields_and_dependent_rows(self):
        conn = FakeConn()
        reset_article(conn, 20512)
        sql = " ".join(s for s, _ in conn.executed)
        assert "DELETE FROM public.article_sentiment" in sql
        assert "DELETE FROM public.article_insights" in sql
        for col in ("embedding", "category", "category_reason", "primary_ticker",
                    "primary_segment", "more_tickers", "more_segments",
                    "insights_extracted_at"):
            assert col in sql
        assert conn.executed[-1] == ("COMMIT", None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_pipeline_eval.py -v`
Expected: ImportError — `cannot import name 'build_items'`

- [ ] **Step 3: Implement**

Append to `src/ticker_news/evals/pipeline_eval.py` (add the new imports at the top of the file, below `from __future__ import annotations`):

```python
from datetime import datetime, timezone

import psycopg
```

then the functions:

```python
def connect_eval(dsn: str | None = None) -> psycopg.Connection:
    """Fresh transactional connection to the eval target DB (DSN overridable).

    pgvector registration matches the worker convention (db.connect(vector=True));
    a separate helper because db.connect() cannot take an explicit DSN.
    """
    from pgvector.psycopg import register_vector

    from ticker_news.shared.config import get_settings

    conn = psycopg.connect(dsn or get_settings().database_url)
    register_vector(conn)
    return conn


def ensure_eval_schema(conn: psycopg.Connection) -> None:
    """Additively heal an older shared schema; safe to run every time."""
    from ticker_news.sentiment import store as sentiment_store

    conn.execute(
        "ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS insights_extracted_at timestamptz"
    )
    conn.execute(
        "ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS provider_sentiments jsonb"
    )
    conn.commit()
    sentiment_store.ensure_schema(conn)


def build_items(conn: psycopg.Connection, ids: list[int]) -> list[dict]:
    """Load articles as Langfuse local experiment items; reject unusable ones.

    The input payload is JSON-only (datetimes as ISO strings) so the same dict
    can be upserted as a Langfuse dataset item.
    """
    rows = conn.execute(
        "SELECT id, url, primary_ticker, published_utc, title, status, "
        "coalesce(content, '') <> '' "
        "FROM public.articles WHERE id = ANY(%s) ORDER BY id",
        (ids,),
    ).fetchall()
    missing = sorted(set(ids) - {r[0] for r in rows})
    if missing:
        raise ValueError(f"article ids not found: {missing}")
    items = []
    for aid, url, ticker, published, title, status, has_content in rows:
        if status != "ok" or not has_content:
            raise ValueError(f"article {aid} has no scraped content (status={status})")
        if published is None:
            raise ValueError(f"article {aid} has no published_utc; cannot price the entry")
        items.append({
            "input": {
                "article_id": aid,
                "url": url,
                "published_utc": published.astimezone(timezone.utc).isoformat(),
                "title": title or "",
            },
            "metadata": {"seed_ticker": ticker or ""},
        })
    return items


def reset_article(conn: psycopg.Connection, article_id: int) -> None:
    """Clear every derived field so the idempotent stage adapters re-run.

    Scraped content is untouched. One transaction: an eval article is never
    left half-reset.
    """
    conn.execute(
        "DELETE FROM public.article_sentiment WHERE article_id = %s", (article_id,)
    )
    conn.execute(
        "DELETE FROM public.article_insights WHERE article_id = %s", (article_id,)
    )
    conn.execute(
        "UPDATE public.articles SET embedding = NULL, category = NULL, "
        "category_reason = NULL, primary_ticker = NULL, primary_segment = NULL, "
        "more_tickers = NULL, more_segments = NULL, insights_extracted_at = NULL "
        "WHERE id = %s",
        (article_id,),
    )
    conn.commit()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_pipeline_eval.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```powershell
git add src/ticker_news/evals tests/evals
git commit -m "feat: eval schema healing, item building, article reset"
```

---

### Task 3: Price lookup + Langfuse evaluators

**Files:**
- Modify: `src/ticker_news/evals/pipeline_eval.py`
- Modify: `tests/evals/test_pipeline_eval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evals/test_pipeline_eval.py`:

```python
from types import SimpleNamespace

from ticker_news.evals import pipeline_eval
from ticker_news.evals.pipeline_eval import (
    avg_directional_agreement,
    directional_agreement_evaluator,
    price_move_evaluator,
)

ITEM_INPUT = {
    "article_id": 20512,
    "url": "https://example.com/20512",
    "published_utc": "2026-05-28T11:10:00+00:00",
    "title": "Title",
}


class TestItemEvaluators:
    def test_buy_with_rising_price_scores_one(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (2.5, None))
        pipeline_eval._cached_move.cache_clear()
        ev = directional_agreement_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.name == "directional_agreement"
        assert ev.value == 1.0

    def test_no_ticker_excluded(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (2.5, None))
        pipeline_eval._cached_move.cache_clear()
        ev = directional_agreement_evaluator(
            input=ITEM_INPUT,
            output={"action": None, "ticker": None, "skip_reason": "no primary ticker"},
        )
        assert ev.value is None
        assert "no primary ticker" in ev.comment

    def test_price_move_recorded_even_for_hold(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (-1.3, None))
        pipeline_eval._cached_move.cache_clear()
        ev = price_move_evaluator(
            input=ITEM_INPUT,
            output={"action": "hold", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.name == "price_move_pct"
        assert ev.value == -1.3

    def test_price_move_none_when_no_data(self, monkeypatch):
        monkeypatch.setattr(
            pipeline_eval, "realized_move", lambda t, p: (None, "no tradeable entry/exit bar")
        )
        pipeline_eval._cached_move.cache_clear()
        ev = price_move_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.value is None
        assert "no tradeable" in ev.comment


def _result(*evals):
    return SimpleNamespace(evaluations=list(evals))


class TestRunEvaluator:
    def test_averages_only_scored_items(self):
        results = [
            _result(SimpleNamespace(name="directional_agreement", value=1.0)),
            _result(SimpleNamespace(name="directional_agreement", value=0.0)),
            _result(SimpleNamespace(name="directional_agreement", value=None)),
            _result(SimpleNamespace(name="price_move_pct", value=5.0)),
        ]
        ev = avg_directional_agreement(item_results=results)
        assert ev.name == "avg_directional_agreement"
        assert ev.value == 0.5
        assert "2/4" in ev.comment

    def test_no_scorable_items(self):
        ev = avg_directional_agreement(item_results=[])
        assert ev.value is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_pipeline_eval.py -v`
Expected: ImportError — `cannot import name 'avg_directional_agreement'`

- [ ] **Step 3: Implement**

Add `from datetime import timedelta` and `from functools import lru_cache` to the imports of `pipeline_eval.py` (merge with the existing `datetime` import: `from datetime import datetime, timedelta, timezone`), plus `from langfuse import Evaluation`. Then append:

```python
def realized_move(ticker: str, published_utc: datetime) -> tuple[float | None, str | None]:
    """Entry->close move per the backtest rule. Returns (gain_pct, error).

    Entry = first tradeable minute bar at/after publication; exit = that day's
    regular close, or the next session's close for extended-hours entries
    (include_after_hours=True so evening articles still get scored).
    """
    from ticker_news.research import ticker_scan as ts

    pub_et = published_utc.astimezone(ts.MARKET_TZ)
    frm = (pub_et.date() - timedelta(days=1)).isoformat()
    to = (pub_et.date() + timedelta(days=7)).isoformat()
    try:
        prices = ts.fetch_prices(ticker, frm, to)
    except RuntimeError as exc:  # covers ScanError; missing MASSIVE_API_KEY etc.
        return None, str(exc)
    sim = ts.simulate(
        {"id": 0, "ticker": ticker, "published_et": pub_et}, prices,
        include_after_hours=True,
    )
    if sim is None:
        return None, "no tradeable entry/exit bar"
    return sim["gain_pct"], None


@lru_cache(maxsize=256)
def _cached_move(ticker: str, published_iso: str) -> tuple[float | None, str | None]:
    """Both item evaluators need the move; fetch Massive bars only once."""
    return realized_move(ticker, datetime.fromisoformat(published_iso))


def directional_agreement_evaluator(*, input, output, **kwargs) -> Evaluation:
    out = output or {}
    action, ticker = out.get("action"), out.get("ticker")
    gain_pct, price_err = None, "no primary ticker"
    if ticker:
        gain_pct, price_err = _cached_move(ticker, input["published_utc"])
    value, comment = score_directional(
        action, gain_pct, skip_reason=out.get("skip_reason") or price_err
    )
    return Evaluation(name="directional_agreement", value=value, comment=comment)


def price_move_evaluator(*, input, output, **kwargs) -> Evaluation:
    ticker = (output or {}).get("ticker")
    if not ticker:
        return Evaluation(name="price_move_pct", value=None, comment="no primary ticker")
    gain_pct, price_err = _cached_move(ticker, input["published_utc"])
    if gain_pct is None:
        return Evaluation(name="price_move_pct", value=None, comment=price_err)
    return Evaluation(
        name="price_move_pct", value=gain_pct,
        comment=f"entry->close move {gain_pct:+.2f}%",
    )


def avg_directional_agreement(*, item_results, **kwargs) -> Evaluation:
    values = [
        e.value
        for r in item_results
        for e in r.evaluations
        if e.name == "directional_agreement" and e.value is not None
    ]
    if not values:
        return Evaluation(
            name="avg_directional_agreement", value=None, comment="no scorable items"
        )
    avg = sum(values) / len(values)
    return Evaluation(
        name="avg_directional_agreement", value=avg,
        comment=f"scored {len(values)}/{len(item_results)} items, avg {avg:.2f}",
    )
```

Note: `monkeypatch.setattr(pipeline_eval, "realized_move", ...)` works because `_cached_move` calls `realized_move(...)` as a module global; the `cache_clear()` in each test prevents cross-test cache hits.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/evals/test_pipeline_eval.py -v`
Expected: 20 passed

- [ ] **Step 5: Commit**

```powershell
git add src/ticker_news/evals tests/evals
git commit -m "feat: price-vs-verdict evaluators for the pipeline eval"
```

---

### Task 4: Experiment task + orchestrator

**Files:**
- Modify: `src/ticker_news/evals/pipeline_eval.py`

This is integration glue over already-tested parts (stage adapters are production code; scoring/evaluators/reset are unit-tested above). No new unit tests; the smoke run in Task 6 verifies it end to end.

- [ ] **Step 1: Implement the task factory**

Append to `pipeline_eval.py`:

```python
def make_task(dsn: str | None):
    """Experiment task: reset the article, run the real stage chain, return the verdict.

    A fresh connection per invocation — sync psycopg connections must not be
    shared across the runner's concurrent task calls (same rule as the worker).
    """

    def run_pipeline_task(*, item, **kwargs) -> dict:
        from ticker_news.service import stages
        from ticker_news.shared import observability as obs

        data = item["input"] if isinstance(item, dict) else item.input
        article_id, url = data["article_id"], data["url"]
        conn = connect_eval(dsn)
        try:
            reset_article(conn, article_id)
            tag_ctx = stages.TagContext.load(conn)
            with obs.stage_span("embed"):
                stages.embed_stage(conn, url)
            with obs.stage_span("classify"):
                category = stages.classify_stage(conn, url)
            with obs.stage_span("tag"):
                stages.tag_stage(conn, url, tag_ctx)
            with obs.stage_span("insights"):
                stages.insights_stage(conn, url, tag_ctx)
            with obs.stage_span("sentiment"):
                verdict = stages.sentiment_stage(conn, url)
            if verdict is None:
                row = conn.execute(
                    "SELECT category, primary_ticker FROM public.articles WHERE id = %s",
                    (article_id,),
                ).fetchone()
                category, ticker = row if row else (None, None)
                reason = (
                    f"category={category}" if category != "real news"
                    else "no primary ticker" if not ticker
                    else "sentiment skipped"
                )
                return {"action": None, "confidence": None, "category": category,
                        "ticker": ticker, "skip_reason": reason}
            return {"action": verdict["action"], "confidence": verdict["confidence"],
                    "category": category, "ticker": verdict["ticker"],
                    "skip_reason": None}
        finally:
            conn.close()

    return run_pipeline_task
```

Notes for the implementer:
- `classify_stage` returns the category it just assigned (never skips here — reset cleared it). The deliberate omission of `obs.article_trace` is per spec §Tracing: the runner owns the trace; `stage_span`/`chain_config` nest under it via OTel context.
- The stage names passed to `stage_span` (`embed`, `classify`, `tag`, `insights`, `sentiment`) are the observation-name contract from CLAUDE.md — do not rename.

- [ ] **Step 2: Implement the orchestrator**

Append to `pipeline_eval.py`:

```python
EXPERIMENT_NAME = "pipeline-e2e"
_DESCRIPTION = (
    "Full post-scrape pipeline re-run per article; verdict scored against the "
    "realized entry->close price move (Massive)."
)


def run_eval(
    ids: list[int],
    *,
    dataset_name: str | None = None,
    dsn: str | None = None,
    run_name: str | None = None,
):
    """Run the E2E pipeline experiment; returns the langfuse ExperimentResult.

    Local-data mode runs exactly `ids`. Dataset mode upserts `ids` as items
    (deterministic id => idempotent) and runs over the WHOLE dataset, so the
    dataset acts as the growing eval suite.
    """
    from ticker_news.shared import observability as obs
    from ticker_news.shared.config import get_settings

    client = obs.client()
    if client is None:
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are required - "
            "eval results live in Langfuse."
        )
    s = get_settings()
    missing = [
        name
        for name, val in (
            ("MASSIVE_API_KEY", s.massive_api_key),
            ("GOOGLE_API_KEY", s.google_api_key),
            ("OPENAI_API_KEY", s.openai_api_key),
        )
        if not val
    ]
    if missing:
        raise SystemExit(f"missing required keys: {', '.join(missing)}")

    conn = connect_eval(dsn)
    try:
        ensure_eval_schema(conn)
        items = build_items(conn, ids) if ids else []
    finally:
        conn.close()

    common = dict(
        name=EXPERIMENT_NAME,
        run_name=run_name,
        description=_DESCRIPTION,
        task=make_task(dsn),
        evaluators=[directional_agreement_evaluator, price_move_evaluator],
        run_evaluators=[avg_directional_agreement],
        max_concurrency=2,  # each task fans out ~7 LLM calls already
        metadata={"entrypoint": "eval"},
    )
    try:
        if dataset_name:
            try:
                client.create_dataset(name=dataset_name, description=_DESCRIPTION)
            except Exception:  # noqa: BLE001 - already exists is fine
                pass
            for it in items:
                client.create_dataset_item(
                    dataset_name=dataset_name,
                    id=f"article-{it['input']['article_id']}",
                    input=it["input"],
                    metadata=it["metadata"],
                )
            dataset = client.get_dataset(dataset_name)
            if not dataset.items:
                raise SystemExit(f"dataset '{dataset_name}' has no items")
            return dataset.run_experiment(**common)
        if not items:
            raise SystemExit("no article ids given (use --ids, or --dataset with items)")
        return client.run_experiment(data=items, **common)
    finally:
        obs.flush()
```

Implementer caveats:
- If `dataset.run_experiment(**common)` rejects a kwarg (signature may differ slightly from `client.run_experiment`), check `inspect.signature(type(dataset).run_experiment)` and drop the unsupported key — do not silently swallow a TypeError.
- If `client.create_dataset` turns out to upsert rather than raise on duplicates, the try/except is harmless; leave it.

- [ ] **Step 3: Sanity-check imports and run the full offline suite**

Run: `.venv\Scripts\python.exe -c "from ticker_news.evals import pipeline_eval; print('ok')"`
Expected: `ok`

Run: `.venv\Scripts\python.exe -m pytest -m "not db and not integration"`
Expected: all pass (21 eval tests + existing suite), no regressions

- [ ] **Step 4: Commit**

```powershell
git add src/ticker_news/evals
git commit -m "feat: experiment task and run_eval orchestrator"
```

---

### Task 5: CLI command + docs

**Files:**
- Modify: `src/ticker_news/cli.py` (append after the `jobs` sub-app block at the end of the file)
- Modify: `CLAUDE.md` (commands table)

- [ ] **Step 1: Add the eval sub-app**

Append to `src/ticker_news/cli.py`:

```python
eval_app = typer.Typer(help="Pipeline quality evals (Langfuse experiments).")
app.add_typer(eval_app, name="eval")


@eval_app.command("pipeline")
def eval_pipeline(
    ids: str | None = typer.Option(None, help="Comma-separated article ids to evaluate."),
    dataset: str | None = typer.Option(None, "--dataset", help="Langfuse dataset name: upsert --ids as items, then run over the whole dataset."),
    dsn: str | None = typer.Option(None, "--dsn", help="Target DB DSN (default: DATABASE_URL)."),
    run_name: str | None = typer.Option(None, "--run-name", help="Experiment run name (default: auto-generated)."),
) -> None:
    """Re-run articles E2E through the pipeline; score verdicts against actual price moves."""
    from ticker_news.evals import pipeline_eval

    id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else []
    if not id_list and not dataset:
        raise typer.BadParameter("provide --ids, or --dataset with existing items")
    try:
        result = pipeline_eval.run_eval(
            id_list, dataset_name=dataset, dsn=dsn, run_name=run_name
        )
    except SystemExit as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    typer.echo(result.format())
```

- [ ] **Step 2: Verify the command is wired**

Run: `.venv\Scripts\python.exe -m ticker_news.cli eval pipeline --help` — if the module has no `__main__` guard, use `.venv\Scripts\ticker-news.exe eval pipeline --help` instead.
Expected: help text listing `--ids`, `--dataset`, `--dsn`, `--run-name`

Run: `.venv\Scripts\ticker-news.exe eval pipeline`
Expected: exits non-zero with `provide --ids, or --dataset with existing items`

- [ ] **Step 3: Document**

In `CLAUDE.md`, add one row to the commands table (after the `jobs status` row):

```markdown
| `ticker-news eval pipeline --ids N[,..]` | E2E eval: re-run articles through every stage, score verdict vs realized price move as a Langfuse experiment (`--dataset` for run-over-run comparison, `--dsn` for the shared DB) |
```

- [ ] **Step 4: Run the offline suite once more**

Run: `.venv\Scripts\python.exe -m pytest -m "not db and not integration"`
Expected: all pass

- [ ] **Step 5: Commit**

```powershell
git add src/ticker_news/cli.py CLAUDE.md
git commit -m "feat: ticker-news eval pipeline command"
```

---

### Task 6: Smoke run against the shared DB (manual integration)

**Files:** none (verification only)

Prerequisites: the cloudflared tunnel must be running (`cloudflared access tcp --hostname pg.ontimesite.com --url localhost:15432`), and `.env` must contain `MASSIVE_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.

- [ ] **Step 1: Pick the smoke article**

Article **20512** (MRVL, "Marvell CEO Says Data Center Business Is 'On Fire'", published 2026-05-28 11:10 UTC = 07:10 ET premarket -> same-day close exit; category `real news`). Confirm it still looks right:

```powershell
docker exec -e PGPASSWORD=<robert-password> news_pg psql "host=host.docker.internal port=15432 dbname=sharedproject user=robert sslmode=disable" -c "select id, primary_ticker, category, published_utc from articles where id = 20512"
```

- [ ] **Step 2: Run the eval (local-data mode)**

```powershell
.venv\Scripts\ticker-news.exe eval pipeline --ids 20512 --dsn "postgresql://robert:<robert-password>@localhost:15432/sharedproject"
```

Expected: stage logs, then `result.format()` output showing 1 item with a `directional_agreement` score (1.0, 0.0, or None-with-comment if it re-classified or returned hold) and a `price_move_pct` value.

- [ ] **Step 3: Verify shared-DB state**

```powershell
docker exec -e PGPASSWORD=<robert-password> news_pg psql "host=host.docker.internal port=15432 dbname=sharedproject user=robert sslmode=disable" -c "select category, primary_ticker, embedding is not null as embedded, insights_extracted_at is not null as insights from articles where id = 20512" -c "select action, confidence, ticker from article_sentiment where article_id = 20512" -c "select count(*) from article_insights where article_id = 20512"
```

Expected: category re-assigned, embedded = t, insights = t, one `article_sentiment` row (unless the verdict path skipped — then check the eval comment explains why).

- [ ] **Step 4: Verify in Langfuse**

Open the Langfuse project UI -> Traces: a new trace for the experiment item containing the stage spans (`embed`, `classify`, `tag`, `insights`, `sentiment`, `analyst:*`, `synthesize`) with the two scores attached. If running with `--dataset pipeline-e2e`, also check Datasets -> pipeline-e2e -> Runs.

- [ ] **Step 5: Fix anything the smoke run surfaced, then commit any fixes**

```powershell
git add -A src tests
git commit -m "fix: smoke-run fixes for the pipeline eval"   # only if changes were needed
```
