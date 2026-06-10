# Pipeline v2 — Plan 6: Research Port, Legacy Deletion & Release (final phase)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the research/analysis tools into `ticker_news.research`, kill every `sys.path` hack, finish the eval-readiness trace polish, delete the legacy `scripts/` tree, rewrite CLAUDE.md, and open the PR from `refactor/pipeline-v2` to `main`.

**Architecture:** New `src/ticker_news/research/` package (screaming layout: one module per research capability) importing the shared spine (`shared.db`, `shared.config`, `embedding.embedder`). A new `research/market_data.py` consolidates the Massive bars/market-time plumbing the legacy scripts each duplicated. The CSV backfill path gains provider-sentiment parity with the live poller. Worker traces gain verdict/category output and prompt-version metadata so traces are eval-ready.

**Tech Stack:** Existing stack (psycopg/pgvector, typer, LangChain/LangGraph, Langfuse 4.x). `requests` becomes a declared dependency (already used by `massive_rest.py`). Chart rendering deps (`pandas`, `mplfinance`, `matplotlib`) land in a `charts` optional extra with lazy imports.

---

## Decisions locked in this plan

| Decision | Rationale |
|---|---|
| Legacy `insight_sentiment.py`, `followup_sentiment.py`, `backtest_top2.py` are **deleted, not ported** | They are the *old* single-call Gemini sentiment flow, superseded by the analyst-panel orchestrator (`ticker_news.sentiment`, Plan 4 — explicit user requirement). The spec mandates a clean break: one-offs archived via git history, no compatibility shims. Porting them would create a second, untraced sentiment system. |
| New `research/backtest.py` backtests **`article_sentiment` verdicts** (the new system) against realized returns | This is the eval ground-truth generator for the evals milestone. Entry/exit simulation is reused from the `catalyst_returns` port. Writing scores back onto Langfuse traces is designed-for but out of scope (evals milestone). |
| `requirements.txt` is **deleted**; `pip install -e ".[dev,charts]"` is the setup path | pyproject.toml is already the canonical dependency list; requirements.txt is a stale legacy-scripts list (faiss, sentence-transformers, anthropic… none used). |
| `run_scrape.py` and `src/ticker_news/scraping/cli.py` are **deleted** | `ticker-news scrape` (typer) already covers them and does not import either. |
| True Langfuse prompt↔generation object linking is **out of scope** | A `prompt_versions` map in root-trace metadata gives A/B attribution at MVP cost. Object linking (langfuse_prompt in chain metadata) is an evals-milestone item. |
| `ticker-news fetch-news` writes the **legacy per-ticker CSV schema** (`ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name`) | Round-trips provider sentiment through `backfill`, and stays compatible with existing corpus CSVs. |

## Out of scope

- Evals/experiments implementation (trace→dataset extraction, experiment runner, score write-back).
- Real-time feed provider selection.
- Re-running historical backfills.

---

### Task 1: `research/market_data.py` + packaging deps

One copy of the Massive market-data plumbing that `scripts/checks_backtesting/ticker_candles.py` (line 51 `_api_key`, line 85 `fetch_bars`), `scripts/ticker_scan/scan_ranges.py` (lines 108–193), and `scripts/ticker_scan/catalyst_returns.py` (lines 117–152) each duplicate.

**Files:**
- Create: `src/ticker_news/research/__init__.py` (empty docstring module)
- Create: `src/ticker_news/research/market_data.py`
- Create: `tests/research/__init__.py`, `tests/research/test_market_data.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: pyproject changes**

In `[project] dependencies` add `"requests"` (already imported by `ticker_news/ingestion/massive_rest.py` but only installed transitively today). Add a new extra:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
charts = [
    "pandas",
    "matplotlib",
    "mplfinance",
]
```

- [ ] **Step 2: Write failing tests** (`tests/research/test_market_data.py`)

```python
from datetime import time as dtime

import pytest

from ticker_news.research import market_data as md


def test_session_of_boundaries():
    assert md.session_of(dtime(3, 59)) == "closed"
    assert md.session_of(dtime(4, 0)) == "premarket"
    assert md.session_of(dtime(9, 29)) == "premarket"
    assert md.session_of(dtime(9, 30)) == "regular"
    assert md.session_of(dtime(15, 59)) == "regular"
    assert md.session_of(dtime(16, 0)) == "after_hours"
    assert md.session_of(dtime(19, 59)) == "after_hours"
    assert md.session_of(dtime(20, 0)) == "closed"


def test_api_key_explicit_wins(monkeypatch):
    assert md.api_key("abc") == "abc"


def test_api_key_missing_raises(monkeypatch):
    monkeypatch.setattr(md, "_settings_key", lambda: "")
    with pytest.raises(RuntimeError):
        md.api_key(None)


def test_fetch_bars_paginates(monkeypatch):
    pages = [
        {"results": [{"t": 1}], "next_url": "https://api.massive.com/page2"},
        {"results": [{"t": 2}], "next_url": None},
    ]
    calls = []

    def fake_get_json(url, params):
        calls.append((url, dict(params)))
        return pages[len(calls) - 1]

    monkeypatch.setattr(md, "get_json", fake_get_json)
    bars = md.fetch_bars("NVDA", span="minute", frm="2025-01-02", to="2025-01-02",
                         key="k")
    assert [b["t"] for b in bars] == [1, 2]
    assert calls[0][1]["apiKey"] == "k"          # key on first call
    assert calls[1][1] == {"apiKey": "k"}        # next_url keeps only the key
```

- [ ] **Step 3: Implement `market_data.py`**

```python
"""Massive market-data plumbing shared by the research tools.

One copy of the market-time constants and bar fetching that the legacy
scripts (ticker_candles, scan_ranges, catalyst_returns) each duplicated.
"""

from __future__ import annotations

import time as _time
from datetime import time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

import requests

MARKET_TZ = ZoneInfo("America/New_York")
PREMARKET_OPEN = dtime(4, 0)
REGULAR_OPEN = dtime(9, 30)
REGULAR_CLOSE = dtime(16, 0)
AFTER_HOURS_CLOSE = dtime(20, 0)

AGGS_URL = "https://api.massive.com/v2/aggs/ticker/{ticker}/range/{multiplier}/{span}/{frm}/{to}"
MAX_RETRIES = 4
RETRY_BACKOFF = 1.5
REQUEST_TIMEOUT = 30


def _settings_key() -> str:
    from ticker_news.shared.config import get_settings

    return get_settings().massive_api_key or ""


def api_key(explicit: Optional[str] = None) -> str:
    """Explicit value wins; else MASSIVE_API_KEY from settings; else error."""
    key = explicit or _settings_key()
    if not key:
        raise RuntimeError("MASSIVE_API_KEY is not set (put it in .env or pass --api-key).")
    return key


def get_json(url: str, params: dict) -> dict:
    """GET with retry/backoff on 429/5xx/network errors (legacy `_get` port)."""
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"transient {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt < MAX_RETRIES - 1:
                _time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"Massive request failed for {url}: {last!r}")


def fetch_bars(
    ticker: str,
    *,
    span: str = "minute",
    multiplier: int = 1,
    frm: str,
    to: str,
    key: Optional[str] = None,
    adjusted: bool = True,
    limit: int = 50000,
) -> list[dict]:
    """All aggregate bars for `ticker` in [frm, to], following next_url pages."""
    k = api_key(key)
    url = AGGS_URL.format(ticker=ticker, multiplier=multiplier, span=span, frm=frm, to=to)
    params: dict = {"adjusted": str(adjusted).lower(), "sort": "asc",
                    "limit": limit, "apiKey": k}
    out: list[dict] = []
    while url:
        payload = get_json(url, params)
        out.extend(payload.get("results", []) or [])
        url = payload.get("next_url")
        params = {"apiKey": k}
    return out


def session_of(t: dtime) -> str:
    """premarket | regular | after_hours | closed (extended-hours convention)."""
    if PREMARKET_OPEN <= t < REGULAR_OPEN:
        return "premarket"
    if REGULAR_OPEN <= t < REGULAR_CLOSE:
        return "regular"
    if REGULAR_CLOSE <= t < AFTER_HOURS_CLOSE:
        return "after_hours"
    return "closed"
```

- [ ] **Step 4: Run** `pytest tests/research/test_market_data.py -q` → all pass; `pytest -m "not db and not integration" -q` → no regressions.

- [ ] **Step 5: Commit** `feat: shared Massive market-data module for research tools`

---

### Task 2: `ingestion/news_history.py` + `fetch-news` CLI

Port `scripts/data_getting_parsing/ticker_news.py` (`fetch_news_rows`/`fetch_news_csv`, 190 lines) into the package, replacing `run_universe.py` with a CLI flag. Reuse `massive_rest._request`-equivalent plumbing — do NOT duplicate retry logic: import `_request`, `BASE_URL`, `PAGE_LIMIT` from `ticker_news.ingestion.massive_rest`.

**Files:**
- Create: `src/ticker_news/ingestion/news_history.py`
- Create: `tests/ingestion/test_news_history.py`
- Modify: `src/ticker_news/cli.py` (add `fetch-news` command)

- [ ] **Step 1: Failing tests**

```python
from ticker_news.ingestion import news_history as nh


def _articles(ticker):
    return [{
        "article_url": f"https://x.com/{ticker.lower()}",
        "published_utc": "2025-01-02T03:04:05Z",
        "publisher": {"name": "Pub"},
        "insights": [
            {"ticker": ticker, "sentiment": "positive", "sentiment_reasoning": "why"},
            {"ticker": "OTHER", "sentiment": "neutral", "sentiment_reasoning": ""},
        ],
    }]


def test_fetch_news_rows_extracts_per_ticker_sentiment(monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles(t))
    rows = nh.fetch_news_rows(["NVDA"], "2025-01-01", "2025-01-31")
    assert rows == [{
        "ticker": "NVDA",
        "article_url": "https://x.com/nvda",
        "published_utc": "2025-01-02T03:04:05Z",
        "sentiment": "positive",
        "sentiment_reasoning": "why",
        "publisher_name": "Pub",
    }]


def test_fetch_news_rows_dedupes_ticker_url_pairs(monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles("NVDA") * 2)
    rows = nh.fetch_news_rows(["NVDA"], "2025-01-01", "2025-01-31")
    assert len(rows) == 1


def test_write_news_csv_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles(t))
    out = tmp_path / "news.csv"
    nh.fetch_news_csv(["NVDA"], "2025-01-01", "2025-01-31", output_path=str(out))
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == (
        "ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name"
    )
    assert "NVDA,https://x.com/nvda" in text
```

- [ ] **Step 2: Implement `news_history.py`**

Public API (port behavior faithfully from the legacy file — pagination via `next_url` with re-attached apiKey, dedupe of (ticker, url) pairs, sentiment extraction from `insights` matching the row's ticker):

```python
def fetch_range(ticker: str, start_iso: str, end_iso: str, *, key: str | None = None) -> list[dict]:
    """All articles for ticker in [start, end] via /v2/reference/news pagination.
    Uses massive_rest._request for retry/backoff; params include
    published_utc.gte / published_utc.lte, order=asc, sort=published_utc."""

def fetch_news_rows(tickers, start_date, end_date, *, key=None) -> list[dict]:
    """One CSV-shaped dict per (ticker, article) pair; dedupes pairs;
    sentiment fields come from the insights entry whose ticker matches."""

def fetch_news_csv(tickers, start_date, end_date, *, output_path="news.csv", key=None) -> str:
    """fetch_news_rows + csv.DictWriter with the legacy header order."""
```

- [ ] **Step 3: Add CLI command** (in `cli.py`, lazy imports as everywhere else)

```python
@app.command(name="fetch-news")
def fetch_news(
    tickers: str | None = typer.Option(None, help="Comma-separated tickers (default: ticker_data table)."),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: today)."),
    output: Path = typer.Option(None, "--output", "-o", help="Output CSV (default: news_<start>_<end>.csv)."),
) -> None:
    """Fetch news metadata + provider sentiment from Massive into a CSV."""
    from datetime import date
    from ticker_news.ingestion import news_history

    t = [x.strip().upper() for x in tickers.split(",") if x.strip()] if tickers else _universe_tickers()
    end_date = end or date.today().isoformat()
    out = str(output) if output else f"news_{start}_{end_date}.csv"
    path = news_history.fetch_news_csv(t, start, end_date, output_path=out)
    typer.echo(f"wrote {path}")
```

- [ ] **Step 4: Run** `pytest tests/ingestion -q` and the offline suite. Verify `ticker-news fetch-news --help` renders.

- [ ] **Step 5: Commit** `feat: port Massive news-history fetch into ingestion with fetch-news CLI`

---

### Task 3: CSV backfill provider sentiment

`CsvBackfillSource` currently delegates to `scraping.csv_source.read_jobs`, which (a) only reads a merged `tickers` column and (b) drops the `sentiment`/`sentiment_reasoning` columns — so backfilled corpora never get `articles.provider_sentiments`, the eval ground truth. Give the backfill source its own reader that groups per-ticker rows by URL and builds `source_meta` in the exact shape `massive_rest.py` produces (`{"sentiments": {TICKER: {"sentiment": ..., "sentiment_reasoning": ...}}, "provider": "csv"}`) so `stages._save_provider_sentiments` persists it.

**Files:**
- Modify: `src/ticker_news/ingestion/csv_backfill.py`
- Modify: `src/ticker_news/scraping/csv_source.py` (singular-`ticker` column fallback)
- Modify: `tests/ingestion/test_csv_backfill.py`, `tests/scraping/test_csv_source.py`

- [ ] **Step 1: Failing tests** — add to `tests/ingestion/test_csv_backfill.py`:

```python
LEGACY_CSV = """\
ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name
NVDA,https://example.com/a,2026-01-02T03:04:05Z,positive,beats guidance,Benzinga
AMD,https://example.com/a,2026-01-02T03:04:05Z,negative,loses share,Benzinga
NVDA,https://example.com/b,2026-01-03T00:00:00Z,,,Reuters
"""


async def test_legacy_rows_merge_by_url_with_sentiments(tmp_path):
    p = tmp_path / "legacy.csv"
    p.write_text(LEGACY_CSV, encoding="utf-8")
    items = await _collect(CsvBackfillSource(str(p)))
    assert len(items) == 2
    a = items[0]
    assert a.url == "https://example.com/a"
    assert a.tickers == ["NVDA", "AMD"]
    assert a.source_meta["provider"] == "csv"
    assert a.source_meta["sentiments"] == {
        "NVDA": {"sentiment": "positive", "sentiment_reasoning": "beats guidance"},
        "AMD": {"sentiment": "negative", "sentiment_reasoning": "loses share"},
    }
    b = items[1]
    assert b.source_meta["sentiments"] == {}  # blank sentiment columns -> no entry


async def test_limit_counts_articles_not_rows(tmp_path):
    p = tmp_path / "legacy.csv"
    p.write_text(LEGACY_CSV, encoding="utf-8")
    items = await _collect(CsvBackfillSource(str(p), limit=1))
    assert len(items) == 1 and items[0].tickers == ["NVDA", "AMD"]
```

And in `tests/scraping/test_csv_source.py`, a singular-column fallback test:

```python
def test_read_jobs_accepts_singular_ticker_column(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("ticker,article_url,published_utc,publisher_name\n"
                 "NVDA,https://x.com/a,2024-11-01T07:20:00Z,Pub\n", encoding="utf-8")
    job = list(read_jobs(str(p)))[0]
    assert job.tickers == ["NVDA"]
```

Existing merged-format tests must keep passing unchanged.

- [ ] **Step 2: Implement.** In `csv_source.py` change the tickers line to:

```python
tickers = [t.strip() for t in (row.get("tickers") or row.get("ticker") or "").split(",") if t.strip()]
```

Rewrite `csv_backfill.py` with its own reader (full file):

```python
"""CSV backfill source — replays a news CSV (legacy per-ticker rows or merged
rows) as a NewsFeedSource, carrying provider sentiment into source_meta."""

from __future__ import annotations

import csv
from datetime import datetime
from typing import AsyncIterator

from ticker_news.ingestion.feed import FeedItem


def _parse_dt(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_feed_items(path: str) -> list[FeedItem]:
    """Group CSV rows by URL (legacy CSVs carry one row per ticker-url pair).

    Tickers merge in first-seen order; non-blank sentiment columns become the
    same source_meta["sentiments"] shape the Massive poller produces.
    """
    merged: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            url = (row.get("article_url") or "").strip()
            if not url:
                continue
            entry = merged.setdefault(url, {
                "tickers": [], "sentiments": {},
                "published_utc": _parse_dt(row.get("published_utc")),
                "publisher": (row.get("publisher_name") or "").strip() or None,
            })
            raw = row.get("tickers") or row.get("ticker") or ""
            for t in (x.strip().upper() for x in raw.split(",")):
                if t and t not in entry["tickers"]:
                    entry["tickers"].append(t)
                    sentiment = (row.get("sentiment") or "").strip()
                    if sentiment:
                        entry["sentiments"][t] = {
                            "sentiment": sentiment,
                            "sentiment_reasoning": (row.get("sentiment_reasoning") or "").strip(),
                        }
    return [
        FeedItem(
            url=url,
            tickers=e["tickers"],
            published_utc=e["published_utc"],
            publisher=e["publisher"],
            source_meta={"sentiments": e["sentiments"], "provider": "csv"},
        )
        for url, e in merged.items()
    ]


class CsvBackfillSource:
    def __init__(self, csv_path: str, *, limit: int | None = None):
        self.csv_path = csv_path
        self.limit = limit

    async def stream(self) -> AsyncIterator[FeedItem]:
        # Blocking file I/O is intentional: backfill reads one bounded CSV at
        # startup. A high-throughput source should use asyncio.to_thread.
        items = read_feed_items(self.csv_path)
        for i, item in enumerate(items):
            if self.limit is not None and i >= self.limit:
                return
            yield item
```

Note the sentiment attaches only on the FIRST row that introduces a ticker (legacy CSVs have one row per ticker-url pair, so this is exact).

- [ ] **Step 3: Run** `pytest tests/ingestion tests/scraping/test_csv_source.py -q`, then the full offline suite.

- [ ] **Step 4: Commit** `feat: carry provider sentiment through CSV backfill into source_meta`

---

### Task 4: `research/search.py` + `search` CLI

Faithful port of `scripts/search/search_articles.py` (620 lines — pure psycopg + pgvector, **no Gemini despite old docs**). The legacy file is the behavioral spec; replace its plumbing with package imports:

- `from embed_articles import EMBED_DIM, MODEL_NAME, embed_query, get_conn` → `from ticker_news.embedding.embedder import embed_query` + `from ticker_news.shared.db import connect` (use `connect(vector=True)`); `EMBED_DIM` from `ticker_news.shared.llm`.
- Delete the `sys.path` hack and dotenv bootstrapping (settings handle .env).
- Keep: all search modes (free-text query, `--like ID`, `--like-url`, `--statement`), every filter flag, the `hnsw.ef_search`/`hnsw.iterative_scan` GUCs, the **two-query** seed-embedding form (fetch the seed embedding first, then `ORDER BY embedding <=> %s` — a join-embedded form defeats the HNSW index), `--months-before`/`--exclusive` windowing, `--min-similarity`, and the output formatting.

**Files:**
- Create: `src/ticker_news/research/search.py`
- Create: `tests/research/test_search.py`
- Modify: `src/ticker_news/cli.py`

- [ ] **Step 1:** Read the legacy file fully. Structure the port as: module-level pure helpers (filter/window SQL builders) + `run_search(...)` orchestrators + a `main(argv)`-free design (typer owns the CLI). Factor the WHERE-clause builder as a pure function so it is unit-testable:

```python
def build_filters(
    *, tickers: list[str] | None, segment: str | None, domain: str | None,
    since: str | None, until: str | None, exclusive: bool,
) -> tuple[str, list]:
    """Returns (SQL fragment starting with AND..., params) — pure, no DB."""
```

- [ ] **Step 2: Offline tests** (`tests/research/test_search.py`) — at minimum:

```python
from ticker_news.research.search import build_filters


def test_build_filters_empty():
    sql, params = build_filters(tickers=None, segment=None, domain=None,
                                since=None, until=None, exclusive=False)
    assert sql == "" and params == []


def test_build_filters_tickers_and_dates():
    sql, params = build_filters(tickers=["NVDA"], segment=None, domain=None,
                                since="2025-01-01", until=None, exclusive=False)
    assert "tickers" in sql and "published_utc" in sql
    assert params[0] == ["NVDA"]
```

(Adjust assertions to the exact fragments the port produces — the test pins the contract, the legacy SQL is the reference.)

- [ ] **Step 3: CLI command** `ticker-news search` exposing the legacy flags (`query` positional optional, `--like`, `--like-url`, `--statement`, `-k`, `--ticker` repeatable, `--segment`, `--domain`, `--since/--until`, `--months-before`, `--min-similarity`, `--exclusive`, `--ef-search`). Lazy import of the research module inside the command body.

- [ ] **Step 4: Verify** offline suite green; `ticker-news search --help` renders; importing `ticker_news.cli` still pulls no langchain/langfuse (the existing lazy-import test must stay green).

- [ ] **Step 5: Commit** `feat: port article semantic search into ticker_news.research`

---

### Task 5: `research/insight_search.py` + `search-insights` CLI

Same treatment for `scripts/search/search_articles_by_insights.py` (649 lines). Keep the dataclasses (`InsightHit`, `SeedInsight`, `InsightGroup`, `RelatedArticle`), `insights_of(conn, article_id)`, `search_insights(...)`, `search_by_insights(...)`, the `_seed_window` months-before logic, and the consolidation/ranking (group by article, rank by best similarity + matched-insight count). Replace plumbing exactly as in Task 4. The consolidation function must be pure (takes hit lists, returns groups) for offline tests.

**Files:**
- Create: `src/ticker_news/research/insight_search.py`
- Create: `tests/research/test_insight_search.py`
- Modify: `src/ticker_news/cli.py`

- [ ] **Step 1:** Port (legacy file is the spec; same import swaps; no sys.path).
- [ ] **Step 2: Offline tests** — pin `_seed_window` date math (months-before, exclusive bound) and the consolidation ranking with synthetic `InsightHit`s (e.g. two articles, one with higher best-similarity, one with more matches — assert documented order).
- [ ] **Step 3: CLI** `ticker-news search-insights` with the legacy flags (`query` positional optional, `--like`, `--like-url`, `-k`, `--per-insight`, `--top-articles`, filters, windowing, `--min-similarity`, `--ef-search`).
- [ ] **Step 4: Verify** offline suite + `--help` + lazy-import test.
- [ ] **Step 5: Commit** `feat: port insight-level semantic search into ticker_news.research`

---

### Task 6: `research/candles.py` + `research chart` CLI

Port `scripts/checks_backtesting/ticker_candles.py` (249 lines). Market-time constants and bar fetching come from `research.market_data` (delete the local `_api_key`/`fetch_bars` copies). pandas/mplfinance import **lazily** inside the rendering function behind a guard:

```python
def _require_charts():
    try:
        import mplfinance  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "chart rendering needs the charts extra: pip install -e \".[charts]\""
        ) from exc
```

Public API: `make_chart(ticker: str, ts: datetime | str, *, out: str | None = None, interval: int = 1, tz: str = "America/New_York", key: str | None = None) -> str` (returns the JPG path; same dark style, gold band + star marker, premarket/after-hours shading as legacy).

Create the `research` typer sub-app in `cli.py`:

```python
research_app = typer.Typer(help="On-demand research & backtesting tools.")
app.add_typer(research_app, name="research")


@research_app.command("chart")
def research_chart(
    ticker: str = typer.Argument(..., help="Ticker symbol."),
    timestamp: str = typer.Argument(..., help="Timestamp to mark (ISO, ET assumed if naive)."),
    output: str | None = typer.Option(None, "--output", "-o"),
    interval: int = typer.Option(1, help="Candle size in minutes."),
) -> None:
    """Render an intraday candlestick chart with the timestamp marked."""
    from ticker_news.research.candles import make_chart

    typer.echo(make_chart(ticker, timestamp, out=output, interval=interval))
```

**Files:** Create `src/ticker_news/research/candles.py`, `tests/research/test_candles.py`; modify `cli.py`.

- [ ] **Step 1:** Port; offline tests cover the timestamp parsing/market-time conversion helper (factor `to_market_time(ts: str|datetime, tz) -> datetime` pure) and the `_require_charts` error path (monkeypatch import failure or assert message when extras missing — use `pytest.importorskip` guards so the suite passes with or without the extra installed).
- [ ] **Step 2:** Offline suite + `ticker-news research chart --help`.
- [ ] **Step 3: Commit** `feat: port intraday candle rendering into ticker_news.research`

---

### Task 7: `research/ticker_scan.py` + scan CLIs

Consolidate the scan suite: `scripts/ticker_scan/scan_ranges.py` (405), `attach_articles.py` (148), `catalyst_returns.py` (361). All Massive HTTP goes through `market_data` (`fetch_bars`, `get_json`, `api_key`, `MARKET_TZ`, `session_of`); DB access through `shared.db.connect()`.

Public API to keep (signatures preserved from legacy so Task 8/9 can import them):

```python
# scan_ranges port
def daily_ranges(bars: list[dict]) -> dict[date, DayRange]   # collapse intraday bars per ET calendar date
def scan(tickers, *, start, end, threshold=5.0, bar="hour", index_ticker="I:COMP",
         nas_threshold=1.2, workers=8, key=None) -> list[dict]
def write_scan_csv(rows, path) -> None

# attach_articles port
def attach(input_csv: str, output_csv: str) -> int           # returns row count

# catalyst_returns port
class TickerPrices:                                           # minute+daily bars; .entry(ts), .next_trading_day(d)
def fetch_prices(ticker: str, frm: str, to: str, key: str | None = None) -> TickerPrices
def simulate(article: dict, prices: TickerPrices, include_after_hours: bool = False) -> dict | None
def catalyst_run(*, start, end, categories, workers=8, include_after_hours=False,
                 all_tickers=False, key=None) -> list[dict]
```

CLI commands on the `research` sub-app: `scan-ranges`, `attach-articles`, `catalyst-returns` — flags mirroring the legacy argparse interfaces (thresholds, dates, `--bar`, `--workers`, `--include-after-hours`, `--all-tickers`, output paths with the legacy default naming).

**Files:** Create `src/ticker_news/research/ticker_scan.py`, `tests/research/test_ticker_scan.py`; modify `cli.py`.

- [ ] **Step 1: Offline tests FIRST** for the pure logic (this is where legacy had zero coverage):
  - `daily_ranges`: synthetic hourly bars across two ET dates (include a bar at 23:30 ET to pin the date-collapse convention) → correct per-date low/high/range_pct/direction.
  - `simulate`: synthetic `TickerPrices` (construct directly from bar dicts) for three publication times — premarket (entry at first bar ≥ ts, exit same-day close), regular hours (same), after-hours (excluded by default; with `include_after_hours=True`, exit next trading day's close). Assert entry/exit prices and `gain_pct` arithmetic.
  - `TickerPrices.next_trading_day` skips a weekend gap in the daily bars.
- [ ] **Step 2:** Port the three modules into one file (~500 lines after dedupe), legacy files as the spec.
- [ ] **Step 3:** CLI commands; offline suite green; `--help` renders for all three.
- [ ] **Step 4: Commit** `feat: port ticker-scan suite (range scan, article attach, catalyst returns)`

---

### Task 8: `research/render.py` + render CLIs

Consolidate `render_bombs.py`, `render_catalyst_bombs.py`, `render_all_tickers.py` (315 lines total) into one module of three functions, all delegating chart drawing to `research.candles.make_chart` (lazy charts deps apply transitively):

```python
def render_bombs(input_csv: str, out_dir: str = "pics_bombs") -> int
def render_catalysts(input_csv: str, *, threshold: float = 3.0,
                     out_dir: str = "pics_bombs/catalysts") -> int
def render_all_tickers(csv_path: str, pics_dir: str, out_dir: str | None = None) -> int
```

Each returns the number of charts written, skips existing files, keeps the legacy filename convention `<ticker>_<article_id>_<date>.jpg` and the prev-after-hours → 04:00 ET marking rule. CLI: `research render-bombs`, `research render-catalysts`, `research render-all`.

**Files:** Create `src/ticker_news/research/render.py`, `tests/research/test_render.py`; modify `cli.py`.

- [ ] **Step 1:** Offline tests for the pure pieces: the articles-JSON-column parsing (a row with two articles yields two render jobs), the threshold filter, and the skip-existing logic (`tmp_path` with a pre-created file; monkeypatch `make_chart` to record calls).
- [ ] **Step 2:** Port; CLI; offline suite green.
- [ ] **Step 3: Commit** `feat: port chart-rendering batch tools into ticker_news.research`

---

### Task 9: `research/backtest.py` — backtest the NEW verdicts

New module (replaces legacy `backtest_top2.py`, which tested the deleted single-call sentiment). Backtests `article_sentiment` rows against realized returns using the Task 7 simulation. This is the eval ground-truth generator: each output row is (verdict, realized return) — the future evals milestone writes these as scores onto the article's Langfuse trace (out of scope here; note it in the module docstring).

```python
def load_verdicts(conn, start: str, end: str) -> list[dict]:
    """article_sentiment ⋈ articles: id, url, ticker, action, confidence,
    published_utc, title — real-news verdicts published in [start, end]."""

def evaluate(verdicts: list[dict], *, include_hold: bool = False,
             workers: int = 8, key: str | None = None) -> list[dict]:
    """Per verdict: simulate() entry at publication / exit next regular close
    via ticker_scan.fetch_prices; signed return = gain_pct for buy,
    -gain_pct for sell. Holds skipped unless include_hold (tracked unsigned)."""

def summarize(results: list[dict]) -> dict:
    """Win rate + avg signed return overall, per action, and per confidence
    bucket (<0.7, 0.7–0.85, >=0.85). Pure."""

def run_backtest(*, start, end, include_hold=False, out=None, workers=8, key=None) -> dict
```

CLI: `research backtest --start --end [--include-hold] [--out] [--workers]` printing the summary table and writing the per-verdict CSV (default `backtest_verdicts_<start>_<end>.csv`).

**Files:** Create `src/ticker_news/research/backtest.py`, `tests/research/test_backtest.py`; modify `cli.py`.

- [ ] **Step 1: Failing tests** for the pure parts:

```python
from ticker_news.research.backtest import summarize


def _r(action, conf, signed):
    return {"action": action, "confidence": conf, "signed_return_pct": signed}


def test_summarize_win_rate_and_buckets():
    s = summarize([_r("buy", 0.9, 2.0), _r("buy", 0.9, -1.0), _r("sell", 0.6, 3.0)])
    assert s["overall"]["n"] == 3
    assert s["overall"]["win_rate"] == pytest.approx(2 / 3)
    assert s["by_action"]["buy"]["avg_return_pct"] == pytest.approx(0.5)
    assert s["by_confidence"][">=0.85"]["n"] == 2
    assert s["by_confidence"]["<0.7"]["n"] == 1
```

Plus an `evaluate` test with a stubbed `fetch_prices`/`simulate` (monkeypatch) asserting sell verdicts get negated returns and holds are skipped by default.

- [ ] **Step 2:** Implement; CLI; offline suite green.
- [ ] **Step 3: Commit** `feat: backtest analyst-panel verdicts against realized returns`

---

### Task 10: Eval-readiness trace polish

Four precise changes; tests pin each.

**Files:**
- Modify: `src/ticker_news/service/stages.py` (classify_stage + sentiment_stage return values)
- Modify: `src/ticker_news/service/worker.py` (docstring + root output enrichment)
- Modify: `src/ticker_news/sentiment/batch.py` (per-article trace)
- Modify: `src/ticker_news/shared/prompts.py` (version recording)
- Modify: `src/ticker_news/shared/observability.py` (entrypoint metadata)
- Tests: `tests/service/test_stages.py`, `tests/service/test_worker.py`, `tests/shared/test_prompts.py`

- [ ] **Step 1: Stage return values.** `classify_stage` returns the category (str) when it classifies, `None` on skip paths. `sentiment_stage` returns a verdict summary dict after `save_verdict`:

```python
    verdict, analyses = judge_article(article, config=obs.chain_config() or None)
    sentiment_store.save_verdict(conn, aid, ticker, verdict, analyses, GEMINI_FLASH)
    return {"ticker": ticker, "action": verdict.action,
            "confidence": verdict.confidence}
```

(and `return None` on each existing skip/rollback path — make the returns explicit). Same for `classify_stage`: `return verdict.category` after the UPDATE, `return None` on skips.

- [ ] **Step 2: Worker root output.** In `process_article`, collect stage summaries and fix the stale docstrings:

Module docstring last sentence becomes: `process_article opens the per-article Langfuse trace; stages emit child spans.` Function docstring line "This is the single seam for the per-article Langfuse trace (Plan 5)." stays accurate — keep.

```python
    stage = job.stage
    ticker = job.tickers[0] if job.tickers else None
    summary: dict = {}
    with obs.article_trace(job.article_url, ticker=ticker) as root:
        try:
            while stage != DONE:
                runner = runners[stage]
                with obs.stage_span(stage):
                    result = await _run_stage(runner, job)
                if stage == "classify" and isinstance(result, str):
                    summary["category"] = result
                if stage == "sentiment" and isinstance(result, dict):
                    summary["verdict"] = result
                ...
            if root is not None:
                root.update(output={"final_stage": jobs.DONE, "ok": True, **summary},
                            metadata=_trace_metadata())
            return True
```

with the same `**summary` + `metadata=_trace_metadata()` on the scrape-empty early return and both error paths, and a tiny helper at module level:

```python
def _trace_metadata() -> dict:
    """Prompt versions actually used this process — A/B attribution on traces."""
    from ticker_news.shared import prompts

    versions = prompts.versions_seen()
    return {"prompt_versions": versions} if versions else {}
```

- [ ] **Step 3: Prompt version recording.** In `prompts.py` add a module-level `_seen_versions: dict[str, int] = {}`, record in `get_prompt`'s success path (`p = c.get_prompt(...); _seen_versions[name] = p.version; return p.prompt`), and expose:

```python
def versions_seen() -> dict[str, int]:
    """Langfuse prompt versions fetched by this process ({} when disabled)."""
    return dict(_seen_versions)
```

Test (in `test_prompts.py`): with a stubbed client returning a prompt object with `.prompt`/`.version`, `get_prompt` records the version and `versions_seen()` returns it; with Langfuse disabled it stays `{}`. Add `_seen_versions.clear()` to the existing autouse cache-clearing fixture in `tests/conftest.py` so tests stay isolated.

- [ ] **Step 4: Batch entrypoint traces.** `article_trace` gains an `entrypoint: str = "service"` keyword: `metadata["entrypoint"] = entrypoint` next to the existing url/ticker metadata. `batch.run_batch` selects `url, primary_ticker` pairs and wraps each judgment:

```python
        for url, ticker in bar:
            try:
                with obs.article_trace(url, ticker=ticker, entrypoint="batch"):
                    with obs.stage_span("sentiment"):
                        sentiment_stage(conn, url)
                done += 1
```

(The deterministic URL-seeded trace id means a batch re-judge lands in the article's existing trace — the by-design re-run correlation; `entrypoint` metadata distinguishes the runs.)

- [ ] **Step 5: Tests.** `test_stages.py`: sentiment stage returns the summary dict on success and `None` on skip (extend the existing stubbed-judge tests). `test_worker.py`: a fake runner map where sentiment returns `{"action": "buy"}` → assert the recorded root `update` call got `output["verdict"]` (the existing tests already stub `obs.article_trace`; extend the fake root to capture kwargs). Keyless no-op behavior must stay covered (everything still yields None / returns {}).

- [ ] **Step 6:** Offline + db suites green.

- [ ] **Step 7: Commit** `feat: eval-ready traces — verdict/category output, prompt versions, batch entrypoint`

---

### Task 11: Legacy deletion + requirements reconciliation

**Files:**
- Delete: entire `scripts/` tree, `run_scrape.py`, `src/ticker_news/scraping/cli.py`, `requirements.txt`, `tests/test_root_cli.py`, `tests/scraping/test_cli.py`
- Modify: `setup_venv.sh`, `.gitignore` (drop stale entries if any reference scripts/), `pyproject.toml` (nothing expected — verify)

- [ ] **Step 1: Pre-flight grep.** `rg -l "scripts/|run_scrape|scraping.cli|scraping import cli" src tests docs *.py *.sh *.md` — every hit must be either deleted in this task or rewritten (docs references to legacy commands get fixed in Task 12's CLAUDE.md rewrite; this task fixes any *code* references). `tests/scraping/test_cli.py` tests the argparse CLI being deleted — delete it; typer `scrape` coverage lives in the CLI lazy-import test and scraping pipeline tests (if reviewers find a coverage gap, add a thin typer-runner test invoking `scrape --help`).
- [ ] **Step 2: Delete** with `git rm -r scripts run_scrape.py src/ticker_news/scraping/cli.py requirements.txt tests/test_root_cli.py tests/scraping/test_cli.py`.
- [ ] **Step 3: setup_venv.sh** → replace `pip install -r requirements.txt` with `pip install -e ".[dev,charts]"` (keep the playwright install note).
- [ ] **Step 4: Verify**: full offline suite green; `python -c "import ticker_news.cli"` works; `ticker-news --help` lists all commands; `rg "embed_articles|PYTHONPATH" docs src tests` finds only historical docs (spec/plans are immutable records — leave them).
- [ ] **Step 5: Commit** `chore: delete legacy scripts tree, root entrypoint, and requirements.txt`

---

### Task 12: CLAUDE.md rewrite

Rewrite `CLAUDE.md` for the post-refactor reality. Required content (structure freely, keep it tight):

- **What this is** — same pipeline description, now: single package `src/ticker_news/`, screaming layout, one `articles` table + `pipeline_jobs` queue + `article_sentiment` verdicts + `article_insights`.
- **Architecture** — the stage chain `scrape → embed → classify → tag → insights → sentiment` driven by `ticker-news serve` (live Massive poll) / `ticker-news backfill` (CSV drain mode); Postgres-backed queue (SKIP LOCKED, backoff, permanent parking, LISTEN/NOTIFY); per-stage batch CLIs still exist for corpus work; sentiment = LangGraph analyst-panel (3 analysts fan-out via Send + synthesis judge); the `NewsFeedSource` protocol & where the future real-time provider plugs in; `research/` package for search/scan/backtest/charts.
- **Commands** — full `ticker-news` command table (scrape, embed, classify, tag, load-universe, load-overviews, insights, sentiment, serve, backfill, fetch-news, search, search-insights, jobs status/retry, prompts push, research chart/scan-ranges/attach-articles/catalyst-returns/backtest/render-*). Setup: `pip install -e ".[dev,charts]"` + `playwright install chromium`; `docker compose up -d` (note the existing `news_pg` container — `docker start news_pg` if name conflicts).
- **Configuration** — `.env` keys: `DATABASE_URL` (single DSN now — note `SCRAPER_DB_DSN` still accepted as a fallback alias), `MASSIVE_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` (cloud free tier; absent keys = tracing strictly off).
- **Observability & prompts** — one Langfuse trace per article (URL-seeded id, re-runs correlate), stable observation names as the eval contract (`process-article, scrape, embed, classify, tag, insights, sentiment, analyst:<role>, synthesize`), root output carries category/verdict, metadata carries prompt versions + entrypoint; prompt workflow: in-repo fallbacks are source of truth, `ticker-news prompts push` publishes, label `production`, edits require restart for lru_cached chains; **accepted gap: embedding costs are not traced** (OpenAI embeddings bypass the callback handler).
- **Testing** — `pytest -m "not db and not integration"` offline; `db` marker targets `news_test` ONLY (guard auto-creates it; `TICKER_NEWS_TEST_DSN` overrides; never points at the real `news` DB — past incident); conftest neutralizes `.env` and shell `LANGFUSE_*` so tests never export traces.
- **Critical gotchas (updated)** — pgvector extension; the universe CSV not committed; the two-query HNSW precedent-retrieval rule (join-embedded ORDER BY defeats the index); per-worker connections/Store (sync psycopg must not be shared across threads); per-thread CallbackHandler in thread pools; charts extra needed for `research chart`/`render-*`.
- **Remove**: the graphify section's structural claims (they describe the pre-refactor layout; keep at most a one-line note that `graphify-out/` is stale until regenerated), all `scripts/...` paths, the PYTHONPATH hack, the two-DSN-convention gotcha (now one DSN), `run_scrape.py`.

- [ ] **Step 1:** Write it (grounded in the actual CLI: run `ticker-news --help` and sub-app `--help`s to enumerate commands rather than trusting this plan).
- [ ] **Step 2:** `git add CLAUDE.md && git commit -m "docs: rewrite CLAUDE.md for the pipeline-v2 package"`

---

### Task 13: Final verification + PR to main

- [ ] **Step 1:** Full offline suite (`pytest -m "not db and not integration" -q`) and db suite (`pytest -m db -q`, Docker up) — both green.
- [ ] **Step 2:** Smoke the CLI surface: `ticker-news --help`, every sub-app `--help`, `python -c "import ticker_news.cli"` (lazy-import check).
- [ ] **Step 3:** Final integration review (subagent) over the whole Plan 6 diff; fix findings.
- [ ] **Step 4:** Push and open the PR: `gh pr create --base main --head refactor/pipeline-v2 --title "Pipeline v2: screaming-architecture refactor, continuous service, analyst-panel sentiment, Langfuse observability"` with a body summarizing all six phases (spec + plans linked), the eval-readiness state, and the user setup steps (.env keys, prompts push). No attribution footers or signatures anywhere — commits and PR body stay clean.
