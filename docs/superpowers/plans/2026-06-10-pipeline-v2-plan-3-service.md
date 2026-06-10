# Pipeline v2 — Plan 3: Continuous Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pipeline live: a Postgres-backed job queue (`pipeline_jobs`), an async worker service that drives each article through scrape → embed → classify → tag → insights, and a provider-agnostic `NewsFeedSource` port with two implementations (Massive REST poller, CSV backfill) — so the future real-time provider is one new file.

**Architecture:** Phase 3 of `docs/superpowers/specs/2026-06-10-pipeline-v2-design.md`. The queue is just Postgres: jobs keyed by `article_url` (matching `articles.url UNIQUE`), claimed with `FOR UPDATE SKIP LOCKED`, woken by LISTEN/NOTIFY with short-interval polling as fallback. Each worker owns its own DB connection (psycopg connections are not safe for concurrent cross-thread use). Stages reuse Plan 2's per-article callables; sync stage bodies run via `asyncio.to_thread`, the scraper stage is natively async. The `sentiment` stage slots in after `insights` in Plan 4 by adding one entry to the stage chain. Single-service assumption: exactly one `serve` process runs at a time (startup recovers orphaned `running` jobs); multi-process safety beyond SKIP LOCKED is out of scope.

**Tech Stack:** psycopg 3 (sync per-worker conns + one AsyncConnection for LISTEN), asyncio, pydantic (FeedItem), requests (Massive port), typer.

**Branch:** `refactor/pipeline-v2`. Baseline at start: **86 passed, 1 skipped** offline.

---

### Task 1: feed port + CSV backfill source

**Files:**
- Create: `src/ticker_news/ingestion/__init__.py` (empty)
- Create: `src/ticker_news/ingestion/feed.py`
- Create: `src/ticker_news/ingestion/csv_backfill.py`
- Create: `tests/ingestion/__init__.py`, `tests/ingestion/test_csv_backfill.py`

- [ ] **Step 1: Write failing tests**

`tests/ingestion/test_csv_backfill.py`:

```python
import pytest

from ticker_news.ingestion.csv_backfill import CsvBackfillSource
from ticker_news.ingestion.feed import FeedItem

CSV = """\
tickers,article_url,published_utc,publisher_name
"NVDA,AMD",https://example.com/a,2026-01-02T03:04:05Z,Benzinga
NVDA,https://example.com/b,,
,,2026-01-01T00:00:00Z,NoUrl
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "news.csv"
    p.write_text(CSV, encoding="utf-8")
    return str(p)


async def _collect(source):
    return [item async for item in source.stream()]


async def test_yields_feed_items_with_parsed_fields(csv_path):
    items = await _collect(CsvBackfillSource(csv_path))
    assert len(items) == 2  # the url-less row is dropped
    first = items[0]
    assert isinstance(first, FeedItem)
    assert first.url == "https://example.com/a"
    assert first.tickers == ["NVDA", "AMD"]
    assert first.published_utc is not None and first.published_utc.year == 2026
    assert first.publisher == "Benzinga"


async def test_missing_optional_fields_are_none(csv_path):
    items = await _collect(CsvBackfillSource(csv_path))
    second = items[1]
    assert second.url == "https://example.com/b"
    assert second.published_utc is None
    assert second.publisher is None


async def test_limit_caps_rows(csv_path):
    items = await _collect(CsvBackfillSource(csv_path, limit=1))
    assert len(items) == 1
```

(`asyncio_mode = "auto"` is already set in pyproject — plain `async def` tests work.)

Run: `pytest tests/ingestion -q` → FAIL (module missing).

- [ ] **Step 2: Implement feed.py**

```python
"""The provider-agnostic live-feed port.

Any news source — REST poller, websocket consumer, CSV backfill — implements
NewsFeedSource. The service consumes the stream and enqueues pipeline jobs;
nothing downstream knows or cares where items came from. When the real-time
provider is chosen, it becomes one new file implementing this protocol.
"""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class FeedItem(BaseModel):
    url: str
    tickers: list[str] = Field(default_factory=list)
    published_utc: datetime | None = None
    publisher: str | None = None
    source_meta: dict = Field(default_factory=dict)


@runtime_checkable
class NewsFeedSource(Protocol):
    def stream(self) -> AsyncIterator[FeedItem]: ...
```

- [ ] **Step 3: Implement csv_backfill.py**

Reuses the scraper's battle-tested CSV parsing (`read_jobs` handles BOM, blank URLs, the `tickers` column convention, ISO timestamps):

```python
"""CSV backfill source — wraps the legacy news-CSV flow as a NewsFeedSource."""

from __future__ import annotations

from typing import AsyncIterator

from ticker_news.ingestion.feed import FeedItem
from ticker_news.scraping.csv_source import read_jobs


class CsvBackfillSource:
    def __init__(self, csv_path: str, *, limit: int | None = None):
        self.csv_path = csv_path
        self.limit = limit

    async def stream(self) -> AsyncIterator[FeedItem]:
        produced = 0
        for job in read_jobs(self.csv_path):
            if self.limit is not None and produced >= self.limit:
                return
            produced += 1
            yield FeedItem(
                url=job.url,
                tickers=job.tickers,
                published_utc=job.published_utc,
                publisher=job.publisher,
            )
```

- [ ] **Step 4: Run tests, full suite, commit**

Run: `pytest tests/ingestion -q` → 3 passed. Full offline → 89 passed, 1 skipped.

```bash
git add src/ticker_news/ingestion tests/ingestion
git commit -m "feat: NewsFeedSource port with CSV backfill source"
```
(Standing rule for every commit: no Co-Authored-By trailers, no AI signatures.)

---

### Task 2: Massive REST poller source

Port the HTTP plumbing from `scripts/data_getting_parsing/ticker_news.py` (read it first; the legacy script stays untouched) and build a polling source on top. Design for testability: the source takes an injectable `fetch_articles(ticker, since_iso)` callable; production wiring uses the ported pagination functions.

**Files:**
- Create: `src/ticker_news/ingestion/massive_rest.py`
- Create: `tests/ingestion/test_massive_rest.py`

- [ ] **Step 1: Write failing tests**

`tests/ingestion/test_massive_rest.py`:

```python
from datetime import datetime, timedelta, timezone

from ticker_news.ingestion.massive_rest import MassiveRestSource

ARTICLE_A = {
    "article_url": "https://example.com/a",
    "published_utc": "2026-06-09T10:00:00Z",
    "publisher": {"name": "Benzinga"},
    "insights": [{"ticker": "NVDA", "sentiment": "positive", "sentiment_reasoning": "beat"}],
}
ARTICLE_A_AMD = {**ARTICLE_A, "insights": [{"ticker": "AMD", "sentiment": "neutral", "sentiment_reasoning": ""}]}
ARTICLE_B = {
    "article_url": "https://example.com/b",
    "published_utc": "2026-06-09T11:00:00Z",
    "publisher": {"name": "Reuters"},
    "insights": [],
}


def _source(pages, tickers=("NVDA", "AMD")):
    calls = []

    def fake_fetch(ticker, since_iso):
        calls.append((ticker, since_iso))
        return pages.get(ticker, [])

    src = MassiveRestSource(
        list(tickers), poll_interval_s=0.01, lookback=timedelta(days=1),
        fetch_articles=fake_fetch, max_polls=1,
    )
    return src, calls


async def _collect(src):
    return [item async for item in src.stream()]


async def test_groups_same_url_across_tickers_merging_tickers():
    src, _ = _source({"NVDA": [ARTICLE_A], "AMD": [ARTICLE_A_AMD]})
    items = await _collect(src)
    assert len(items) == 1
    assert items[0].url == "https://example.com/a"
    assert sorted(items[0].tickers) == ["AMD", "NVDA"]


async def test_sentiment_lands_in_source_meta_per_ticker():
    src, _ = _source({"NVDA": [ARTICLE_A], "AMD": []})
    items = await _collect(src)
    meta = items[0].source_meta
    assert meta["sentiments"]["NVDA"]["sentiment"] == "positive"


async def test_cursor_advances_past_seen_articles():
    src, calls = _source({"NVDA": [ARTICLE_A, ARTICLE_B], "AMD": []})
    await _collect(src)
    # after one poll, the per-ticker cursor moved to the newest published_utc
    assert src.cursor("NVDA") == datetime(2026, 6, 9, 11, 0, tzinfo=timezone.utc)


async def test_second_poll_does_not_reyield_seen_urls():
    pages = {"NVDA": [ARTICLE_A], "AMD": []}
    calls = []

    def fake_fetch(ticker, since_iso):
        calls.append((ticker, since_iso))
        return pages.get(ticker, [])

    src = MassiveRestSource(
        ["NVDA", "AMD"], poll_interval_s=0.01, lookback=timedelta(days=1),
        fetch_articles=fake_fetch, max_polls=2,
    )
    items = await _collect(src)
    assert len(items) == 1  # second poll returns the same article; not re-yielded
```

Run → FAIL.

- [ ] **Step 2: Implement massive_rest.py**

```python
"""Massive.com REST news poller — the 'live today' NewsFeedSource.

Polls the news endpoint per universe ticker on an interval, keeping a
per-ticker published_utc cursor and deduping URLs across tickers (one
FeedItem per article with all its tickers merged). The HTTP plumbing
(_request retry/backoff, cursor pagination) is ported from the legacy
scripts/data_getting_parsing/ticker_news.py.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Callable, Iterable, Optional

from ticker_news.ingestion.feed import FeedItem
from ticker_news.shared.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.massive.com/v2/reference/news"
PAGE_LIMIT = 1000
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
REQUEST_TIMEOUT = 30


class MassiveAPIError(RuntimeError):
    """Raised when the Massive API returns a non-recoverable error."""


# _request(session, url, params) -> dict
#   ← port VERBATIM from scripts/data_getting_parsing/ticker_news.py lines 61-75
#     (requests.Session GET with retries on 429/5xx/network)


def _parse_utc(value: str | None) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_articles_rest(ticker: str, since_iso: str) -> list[dict]:
    """One blocking fetch of every article for `ticker` since `since_iso`.

    Cursor pagination ported from the legacy _iter_articles: follow next_url,
    re-attaching the apiKey (next_url carries filters but not the key).
    """
    import requests

    key = get_settings().massive_api_key
    if not key:
        raise MassiveAPIError("MASSIVE_API_KEY is not set (put it in .env).")
    out: list[dict] = []
    params: Optional[dict] = {
        "ticker": ticker,
        "published_utc.gte": since_iso,
        "order": "asc",
        "sort": "published_utc",
        "limit": PAGE_LIMIT,
        "apiKey": key,
    }
    url = BASE_URL
    with requests.Session() as session:
        while url:
            payload = _request(session, url, params)
            out.extend(payload.get("results", []) or [])
            url = payload.get("next_url")
            params = {"apiKey": key} if url else None
    return out


class MassiveRestSource:
    """Poll-based NewsFeedSource over the Massive news API.

    max_polls is for tests/drain runs; None means poll forever.
    """

    def __init__(
        self,
        tickers: Iterable[str],
        *,
        poll_interval_s: float = 60.0,
        lookback: timedelta = timedelta(hours=24),
        fetch_articles: Callable[[str, str], list[dict]] = fetch_articles_rest,
        max_polls: int | None = None,
    ):
        self.tickers = [t.strip().upper() for t in tickers if t and t.strip()]
        self.poll_interval_s = poll_interval_s
        self.lookback = lookback
        self.fetch_articles = fetch_articles
        self.max_polls = max_polls
        self._cursors: dict[str, datetime] = {}
        self._seen_urls: set[str] = set()

    def cursor(self, ticker: str) -> datetime | None:
        return self._cursors.get(ticker)

    def _since(self, ticker: str, now: datetime) -> datetime:
        return self._cursors.get(ticker, now - self.lookback)

    async def stream(self) -> AsyncIterator[FeedItem]:
        polls = 0
        while self.max_polls is None or polls < self.max_polls:
            polls += 1
            now = datetime.now(timezone.utc)
            # url -> (tickers, article) accumulated across this poll
            batch: dict[str, tuple[list[str], dict]] = {}
            for ticker in self.tickers:
                since = self._since(ticker, now)
                try:
                    articles = await asyncio.to_thread(
                        self.fetch_articles, ticker, since.isoformat()
                    )
                except Exception as exc:
                    logger.warning("massive poll failed for %s: %r", ticker, exc)
                    continue
                newest = self._cursors.get(ticker)
                for article in articles:
                    published = _parse_utc(article.get("published_utc"))
                    if published and (newest is None or published > newest):
                        newest = published
                    url = (article.get("article_url") or "").strip()
                    if not url:
                        continue
                    tickers_for_url, _ = batch.setdefault(url, ([], article))
                    if ticker not in tickers_for_url:
                        tickers_for_url.append(ticker)
                if newest is not None:
                    self._cursors[ticker] = newest

            for url, (url_tickers, article) in batch.items():
                if url in self._seen_urls:
                    continue
                self._seen_urls.add(url)
                sentiments = {}
                for insight in article.get("insights") or []:
                    t = (insight.get("ticker") or "").upper()
                    if t:
                        sentiments[t] = {
                            "sentiment": insight.get("sentiment", ""),
                            "sentiment_reasoning": insight.get("sentiment_reasoning", ""),
                        }
                yield FeedItem(
                    url=url,
                    tickers=url_tickers,
                    published_utc=_parse_utc(article.get("published_utc")),
                    publisher=(article.get("publisher") or {}).get("name") or None,
                    source_meta={"sentiments": sentiments, "provider": "massive"},
                )

            if self.max_polls is None or polls < self.max_polls:
                await asyncio.sleep(self.poll_interval_s)
```

Note on `_seen_urls`: it grows unbounded over a long-running process; the DB enqueue is the real dedupe (ON CONFLICT DO NOTHING) — this set just avoids re-yield churn. Acceptable for now; document with a comment.

- [ ] **Step 3: Run tests, full suite, commit**

Run: `pytest tests/ingestion -q` → 7 passed. Full offline → 93 passed, 1 skipped.

```bash
git add src/ticker_news/ingestion/massive_rest.py tests/ingestion/test_massive_rest.py
git commit -m "feat: Massive REST polling feed source with per-ticker cursors"
```

---

### Task 3: the job queue (`service/jobs.py`)

**Files:**
- Create: `src/ticker_news/service/__init__.py` (empty)
- Create: `src/ticker_news/service/jobs.py`
- Create: `tests/service/__init__.py`, `tests/service/test_jobs_unit.py` (offline), `tests/service/test_jobs_db.py` (db-marked, runs against news_test)

- [ ] **Step 1: Write failing offline unit tests**

`tests/service/test_jobs_unit.py`:

```python
from ticker_news.service.jobs import (
    BACKOFF_CAP_S,
    BASE_BACKOFF_S,
    DONE,
    STAGES,
    backoff_delay,
    next_stage,
)


def test_stage_chain_order():
    assert STAGES == ["scrape", "embed", "classify", "tag", "insights"]


def test_next_stage_walks_the_chain_then_done():
    assert next_stage("scrape") == "embed"
    assert next_stage("insights") == DONE


def test_backoff_is_exponential_and_capped():
    assert backoff_delay(0) == BASE_BACKOFF_S
    assert backoff_delay(1) == BASE_BACKOFF_S * 2
    assert backoff_delay(50) == BACKOFF_CAP_S
```

- [ ] **Step 2: Implement jobs.py**

```python
"""Postgres-backed pipeline job queue.

One row per article URL. Workers claim with FOR UPDATE SKIP LOCKED, drive the
article through the stage chain, and advance `stage` after each step — a crash
resumes mid-article. Failures back off exponentially; over-cap jobs park as
'failed' and are requeueable via the CLI. NOTIFY 'pipeline_jobs' wakes the
service instantly on enqueue; polling is the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg

STAGES = ["scrape", "embed", "classify", "tag", "insights"]
DONE = "done"

MAX_ATTEMPTS = 5
BASE_BACKOFF_S = 30.0
BACKOFF_CAP_S = 3600.0

NOTIFY_CHANNEL = "pipeline_jobs"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    article_url     text PRIMARY KEY,
    stage           text NOT NULL DEFAULT 'scrape',
    status          text NOT NULL DEFAULT 'pending',
    attempts        int  NOT NULL DEFAULT 0,
    last_error      text,
    tickers         text[] NOT NULL DEFAULT '{}',
    published_utc   timestamptz,
    publisher       text,
    enqueued_at     timestamptz NOT NULL DEFAULT now(),
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pipeline_jobs_claim_idx
    ON pipeline_jobs (status, next_attempt_at);
"""


@dataclass
class Job:
    article_url: str
    stage: str
    attempts: int
    tickers: list[str]
    published_utc: datetime | None
    publisher: str | None


def ensure_schema(conn: psycopg.Connection) -> None:
    for statement in (s.strip() for s in _SCHEMA.split(";")):
        if statement:
            conn.execute(statement)
    conn.commit()


def next_stage(stage: str) -> str:
    idx = STAGES.index(stage)
    return STAGES[idx + 1] if idx + 1 < len(STAGES) else DONE


def backoff_delay(attempts: int) -> float:
    return min(BASE_BACKOFF_S * (2 ** attempts), BACKOFF_CAP_S)


def enqueue(conn: psycopg.Connection, item) -> bool:
    """Insert a job for a FeedItem; returns True if it was new."""
    cur = conn.execute(
        "INSERT INTO pipeline_jobs (article_url, tickers, published_utc, publisher) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (article_url) DO NOTHING",
        (item.url, item.tickers, item.published_utc, item.publisher),
    )
    new = cur.rowcount == 1
    if new:
        conn.execute(f"NOTIFY {NOTIFY_CHANNEL}")
    conn.commit()
    return new


def claim(conn: psycopg.Connection) -> Job | None:
    row = conn.execute(
        """
        UPDATE pipeline_jobs SET status = 'running', updated_at = now()
        WHERE article_url = (
            SELECT article_url FROM pipeline_jobs
            WHERE status = 'pending' AND next_attempt_at <= now()
            ORDER BY enqueued_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING article_url, stage, attempts, tickers, published_utc, publisher
        """
    ).fetchone()
    conn.commit()
    return Job(*row) if row else None


def advance(conn: psycopg.Connection, article_url: str, to_stage: str) -> None:
    """Record stage completion. to_stage == DONE finishes the job."""
    status = "done" if to_stage == DONE else "running"
    conn.execute(
        "UPDATE pipeline_jobs SET stage = %s, status = %s, attempts = 0, "
        "last_error = NULL, updated_at = now() WHERE article_url = %s",
        (to_stage, status, article_url),
    )
    conn.commit()


def fail(conn: psycopg.Connection, article_url: str, error: str,
         *, max_attempts: int = MAX_ATTEMPTS) -> None:
    row = conn.execute(
        "UPDATE pipeline_jobs SET attempts = attempts + 1, last_error = %s, "
        "updated_at = now() WHERE article_url = %s RETURNING attempts",
        (error[:2000], article_url),
    ).fetchone()
    attempts = row[0] if row else max_attempts
    if attempts >= max_attempts:
        conn.execute(
            "UPDATE pipeline_jobs SET status = 'failed' WHERE article_url = %s",
            (article_url,),
        )
    else:
        conn.execute(
            "UPDATE pipeline_jobs SET status = 'pending', "
            "next_attempt_at = now() + %s * interval '1 second' "
            "WHERE article_url = %s",
            (backoff_delay(attempts), article_url),
        )
    conn.commit()


def recover_orphans(conn: psycopg.Connection) -> int:
    """Reset 'running' jobs to 'pending' (call once at service startup —
    single-service assumption: any running row is an orphan of a dead run)."""
    cur = conn.execute(
        "UPDATE pipeline_jobs SET status = 'pending' WHERE status = 'running'"
    )
    conn.commit()
    return cur.rowcount


def requeue_failed(conn: psycopg.Connection, article_url: str | None = None) -> int:
    where, params = ("AND article_url = %s", (article_url,)) if article_url else ("", ())
    cur = conn.execute(
        f"UPDATE pipeline_jobs SET status = 'pending', attempts = 0, "
        f"next_attempt_at = now() WHERE status = 'failed' {where}",
        params,
    )
    conn.commit()
    return cur.rowcount


def counts(conn: psycopg.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, count(*) FROM pipeline_jobs GROUP BY status"
    ).fetchall()
    return {status: n for status, n in rows}


def queue_drained(conn: psycopg.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pipeline_jobs WHERE status IN ('pending', 'running') LIMIT 1"
    ).fetchone()
    return row is None
```

Run: `pytest tests/service/test_jobs_unit.py -q` → 3 passed.

- [ ] **Step 3: Write the db-marked queue tests**

`tests/service/test_jobs_db.py` — these run against `news_test` (the guarded fixture convention from `tests/scraping/conftest.py`); create a local conftest-free fixture by importing the helpers:

```python
import psycopg
import pytest

from ticker_news.ingestion.feed import FeedItem
from ticker_news.service import jobs

pytestmark = pytest.mark.db

# Reuse the guarded news_test connection logic from the scraping tests.
from tests.scraping.conftest import TEST_DSN, _connect_test_db


@pytest.fixture
def conn():
    try:
        c = _connect_test_db()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d`")
    jobs.ensure_schema(c)
    c.execute("TRUNCATE pipeline_jobs")
    c.commit()
    yield c
    c.execute("TRUNCATE pipeline_jobs")
    c.commit()
    c.close()


def _item(url="https://example.com/a"):
    return FeedItem(url=url, tickers=["NVDA"], publisher="Benzinga")


def test_enqueue_is_idempotent(conn):
    assert jobs.enqueue(conn, _item()) is True
    assert jobs.enqueue(conn, _item()) is False
    assert jobs.counts(conn) == {"pending": 1}


def test_claim_marks_running_and_returns_payload(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    assert job.article_url == "https://example.com/a"
    assert job.stage == "scrape"
    assert job.tickers == ["NVDA"]
    assert jobs.claim(conn) is None  # nothing else pending


def test_advance_to_done(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    jobs.advance(conn, job.article_url, jobs.DONE)
    assert jobs.counts(conn) == {"done": 1}
    assert jobs.queue_drained(conn) is True


def test_fail_backs_off_then_parks_as_failed(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    jobs.fail(conn, job.article_url, "boom")
    assert jobs.counts(conn) == {"pending": 1}
    # backed-off job is not claimable right now
    assert jobs.claim(conn) is None
    for _ in range(jobs.MAX_ATTEMPTS - 1):
        jobs.fail(conn, job.article_url, "boom again")
    assert jobs.counts(conn) == {"failed": 1}


def test_requeue_failed_resets(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    for _ in range(jobs.MAX_ATTEMPTS):
        jobs.fail(conn, job.article_url, "boom")
    assert jobs.requeue_failed(conn) == 1
    assert jobs.claim(conn) is not None


def test_recover_orphans(conn):
    jobs.enqueue(conn, _item())
    jobs.claim(conn)
    assert jobs.recover_orphans(conn) == 1
    assert jobs.claim(conn) is not None
```

Run if docker is up: `pytest tests/service/test_jobs_db.py -q` → 6 passed (else skipped — note which happened).

- [ ] **Step 4: Full suite + commit**

Run: `pytest -m "not db and not integration" -q` → 96 passed, 1 skipped.

```bash
git add src/ticker_news/service tests/service
git commit -m "feat: Postgres pipeline_jobs queue with skip-locked claims and backoff"
```

---

### Task 4: per-article stage adapters (`service/stages.py`)

Thin adapters over Plan 2's per-article callables. Read these before writing: `src/ticker_news/scraping/pipeline.py` (process_job), `src/ticker_news/embedding/embedder.py`, `src/ticker_news/classification/chain.py`, `src/ticker_news/enrichment/tagging.py` (load_ticker_data, build_matcher, build_annotator, compute_row), `src/ticker_news/enrichment/insights.py` (generate_boxes, _store_article_boxes).

**Files:**
- Create: `src/ticker_news/service/stages.py`
- Create: `tests/service/test_stages.py` (offline: scrape status mapping + skip logic via stubs)

- [ ] **Step 1: Write failing tests**

`tests/service/test_stages.py`:

```python
import pytest

from ticker_news.service import stages
from ticker_news.service.jobs import Job


def _job(url="https://example.com/a"):
    return Job(article_url=url, stage="scrape", attempts=0,
               tickers=["NVDA"], published_utc=None, publisher="Benzinga")


class _Resources:
    """Only what scrape_stage touches."""
    fetcher = store = settings = limiter = robots = None


async def test_scrape_error_raises_stage_error(monkeypatch):
    async def fake_process_job(job, fetcher, store, settings, limiter, robots):
        return "error"

    monkeypatch.setattr(stages, "process_job", fake_process_job)
    with pytest.raises(stages.StageError):
        await stages.scrape_stage(_job(), _Resources())


async def test_scrape_ok_and_empty_pass_through(monkeypatch):
    for result in ("ok", "empty"):
        async def fake_process_job(job, *a, _r=result, **kw):
            return _r

        monkeypatch.setattr(stages, "process_job", fake_process_job)
        assert await stages.scrape_stage(_job(), _Resources()) == result


async def test_scrape_builds_article_job_from_queue_payload(monkeypatch):
    seen = {}

    async def fake_process_job(job, *a, **kw):
        seen["job"] = job
        return "ok"

    monkeypatch.setattr(stages, "process_job", fake_process_job)
    await stages.scrape_stage(_job(), _Resources())
    assert seen["job"].url == "https://example.com/a"
    assert seen["job"].tickers == ["NVDA"]
    assert seen["job"].publisher == "Benzinga"
```

Run → FAIL.

- [ ] **Step 2: Implement stages.py**

```python
"""Per-article stage adapters for the worker service.

Each stage is idempotent: it checks whether its output already exists and
skips silently, so a crashed job re-runs from its recorded stage without
duplicating work. Sync stages run inside asyncio.to_thread; scrape is
natively async.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import psycopg

from ticker_news.classification.chain import classify_article
from ticker_news.embedding.embedder import build_text, embed_texts
from ticker_news.enrichment.insights import _store_article_boxes, generate_boxes
from ticker_news.enrichment.insights_text import DEFAULT_QUOTE_THRESHOLD
from ticker_news.enrichment.tagging import (
    build_annotator,
    build_matcher,
    compute_row,
    load_ticker_data,
)
from ticker_news.scraping.models import ArticleJob
from ticker_news.scraping.pipeline import process_job
from ticker_news.service.jobs import Job

logger = logging.getLogger(__name__)


class StageError(RuntimeError):
    """A stage failed in a way that should consume a retry attempt."""


async def scrape_stage(job: Job, resources) -> str:
    """Returns the scraper status: 'ok' | 'empty'. Raises StageError on 'error'."""
    article_job = ArticleJob(
        url=job.article_url,
        tickers=job.tickers,
        published_utc=job.published_utc,
        publisher=job.publisher,
    )
    status = await process_job(
        article_job, resources.fetcher, resources.store, resources.settings,
        resources.limiter, resources.robots,
    )
    if status == "error":
        raise StageError(f"scrape returned error for {job.article_url}")
    return status


def _fetch_article(conn: psycopg.Connection, url: str) -> tuple:
    row = conn.execute(
        "SELECT id, title, content FROM public.articles WHERE url = %s", (url,)
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    return row


def embed_stage(conn: psycopg.Connection, url: str) -> None:
    aid, title, content = _fetch_article(conn, url)
    done = conn.execute(
        "SELECT embedding IS NOT NULL FROM public.articles WHERE id = %s", (aid,)
    ).fetchone()[0]
    if done:
        return
    text = build_text(title, content)
    if not text:
        return
    vec = embed_texts([text])[0]
    conn.execute(
        "UPDATE public.articles SET embedding = %s WHERE id = %s", (vec, aid)
    )
    conn.commit()


def classify_stage(conn: psycopg.Connection, url: str) -> None:
    aid, title, content = _fetch_article(conn, url)
    current = conn.execute(
        "SELECT category FROM public.articles WHERE id = %s", (aid,)
    ).fetchone()[0]
    if current is not None:
        return
    verdict, _confirmed = classify_article(title, content or "")
    conn.execute(
        "UPDATE public.articles SET category = %s, category_reason = %s WHERE id = %s",
        (verdict.category, verdict.reason or None, aid),
    )
    conn.commit()


@dataclass
class TagContext:
    """Matcher/annotator built once per service run (read-only, thread-safe)."""

    data: dict
    find: Callable[[str], list[str]]
    annotate: Optional[Callable[[str], str]]

    @classmethod
    def load(cls, conn: psycopg.Connection) -> "TagContext":
        data = load_ticker_data(conn)
        return cls(data=data, find=build_matcher(data), annotate=build_annotator(data))


def tag_stage(conn: psycopg.Connection, url: str, tag_ctx: TagContext) -> None:
    row = conn.execute(
        "SELECT id, tickers, title, content, primary_ticker "
        "FROM public.articles WHERE url = %s", (url,),
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    aid, tickers, title, content, primary = row
    if primary is not None:
        return
    text = " ".join(p for p in (title, content) if p)
    p_ticker, p_segment, more_t, more_s = compute_row(
        tickers or [], text, tag_ctx.data, tag_ctx.find
    )
    conn.execute(
        "UPDATE public.articles SET primary_ticker = %s, primary_segment = %s, "
        "more_tickers = %s, more_segments = %s WHERE id = %s",
        (p_ticker, p_segment, more_t, more_s, aid),
    )
    conn.commit()


def insights_stage(conn: psycopg.Connection, url: str, tag_ctx: TagContext) -> None:
    row = conn.execute(
        "SELECT id, title, content, insights_extracted_at "
        "FROM public.articles WHERE url = %s", (url,),
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    aid, title, content, stamped = row
    if stamped is not None:
        return
    if not (content or "").strip():
        conn.execute(
            "UPDATE public.articles SET insights_extracted_at = now() WHERE id = %s",
            (aid,),
        )
        conn.commit()
        return
    boxes, model = generate_boxes(content)
    _store_article_boxes(
        conn, aid, url, title, content, boxes,
        reprocess=False, quote_threshold=DEFAULT_QUOTE_THRESHOLD,
        annotate=tag_ctx.annotate, model=model,
    )
    # embed this article's new insight boxes immediately
    rows = conn.execute(
        "SELECT id, box_text FROM public.article_insights "
        "WHERE article_id = %s AND embedding IS NULL", (aid,),
    ).fetchall()
    if rows:
        vectors = embed_texts([box for _bid, box in rows])
        for (bid, _box), vec in zip(rows, vectors):
            conn.execute(
                "UPDATE public.article_insights SET embedding = %s WHERE id = %s",
                (vec, bid),
            )
    conn.commit()
```

Implementer note: check `_store_article_boxes`'s actual signature in `insights.py` and match it exactly (parameter names/order may differ slightly from this sketch — the source is authoritative). Same for `compute_row`'s text-building convention: check how `tag_all` builds the text it passes (title + content concatenation) and mirror it exactly so service-tagged articles match batch-tagged ones. If `_store_article_boxes` does its own commit or expects an open transaction, follow its convention. Report any signature corrections you made.

- [ ] **Step 3: Run tests, full suite, commit**

Run: `pytest tests/service -q` → offline portion passes (3 unit + 3 stage tests + db tests skip without docker). Full offline → 99 passed, 1 skipped.

```bash
git add src/ticker_news/service/stages.py tests/service/test_stages.py
git commit -m "feat: per-article stage adapters for the worker service"
```

---

### Task 5: worker loop + serve/backfill/jobs CLI

**Files:**
- Create: `src/ticker_news/service/worker.py`
- Create: `tests/service/test_worker.py`
- Modify: `src/ticker_news/cli.py` (serve, backfill, jobs commands)
- Modify: `tests/test_root_cli.py`

- [ ] **Step 1: Write failing worker tests**

`tests/service/test_worker.py` — process_article with injected runners, and a drain-mode serve smoke test with everything faked:

```python
import asyncio

import pytest

from ticker_news.service import worker
from ticker_news.service.jobs import DONE, Job


def _job(stage="scrape"):
    return Job(article_url="https://example.com/a", stage=stage, attempts=0,
               tickers=["NVDA"], published_utc=None, publisher=None)


class FakeQueue:
    """Records advance/fail calls instead of touching Postgres."""

    def __init__(self):
        self.advanced = []
        self.failed = []

    def advance(self, conn, url, to_stage):
        self.advanced.append(to_stage)

    def fail(self, conn, url, error):
        self.failed.append(error)


async def test_process_article_runs_stages_in_order_and_advances():
    ran = []

    async def scrape(job):
        ran.append("scrape")
        return "ok"

    def sync_stage(name):
        def _run(job):
            ran.append(name)
        return _run

    runners = {
        "scrape": scrape,
        "embed": sync_stage("embed"),
        "classify": sync_stage("classify"),
        "tag": sync_stage("tag"),
        "insights": sync_stage("insights"),
    }
    q = FakeQueue()
    await worker.process_article(_job(), runners, q, conn=None)
    assert ran == ["scrape", "embed", "classify", "tag", "insights"]
    assert q.advanced == ["embed", "classify", "tag", "insights", DONE]


async def test_process_article_resumes_mid_chain():
    ran = []
    runners = {
        "tag": lambda job: ran.append("tag"),
        "insights": lambda job: ran.append("insights"),
    }
    q = FakeQueue()
    await worker.process_article(_job(stage="tag"), runners, q, conn=None)
    assert ran == ["tag", "insights"]
    assert q.advanced == ["insights", DONE]


async def test_empty_scrape_short_circuits_to_done():
    async def scrape(job):
        return "empty"

    q = FakeQueue()
    await worker.process_article(_job(), {"scrape": scrape}, q, conn=None)
    assert q.advanced == [DONE]


async def test_stage_failure_calls_fail_not_advance():
    async def scrape(job):
        raise worker.StageError("boom")

    q = FakeQueue()
    await worker.process_article(_job(), {"scrape": scrape}, q, conn=None)
    assert q.advanced == []
    assert len(q.failed) == 1 and "boom" in q.failed[0]
```

Run → FAIL.

- [ ] **Step 2: Implement worker.py**

```python
"""The continuous service: feed consumer + async worker pool over pipeline_jobs.

Each worker owns its own psycopg connection (sync psycopg connections must not
be shared across concurrently-running to_thread calls). One AsyncConnection
LISTENs on the notify channel to wake claimers instantly; a poll interval is
the fallback. process_article is the single seam where the per-article
Langfuse trace attaches in a later phase.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Mapping, Union

import psycopg

from ticker_news.ingestion.feed import NewsFeedSource
from ticker_news.service import jobs, stages
from ticker_news.service.jobs import DONE, Job, NOTIFY_CHANNEL
from ticker_news.service.stages import StageError, TagContext
from ticker_news.shared.config import get_settings
from ticker_news.shared.db import connect

logger = logging.getLogger(__name__)

StageRunner = Union[Callable[[Job], None], Callable[[Job], Awaitable[object]]]


async def process_article(
    job: Job,
    runners: Mapping[str, StageRunner],
    queue=jobs,
    *,
    conn,
) -> None:
    """Drive one article through its remaining stages, advancing after each."""
    stage = job.stage
    try:
        while stage != DONE:
            runner = runners[stage]
            if asyncio.iscoroutinefunction(runner):
                result = await runner(job)
            else:
                result = await asyncio.to_thread(runner, job)
            if stage == "scrape" and result == "empty":
                # nothing extracted: no content for downstream stages
                await asyncio.to_thread(queue.advance, conn, job.article_url, DONE)
                return
            stage = jobs.next_stage(stage)
            await asyncio.to_thread(queue.advance, conn, job.article_url, stage)
    except Exception as exc:
        logger.warning("article %s failed at stage %s: %r", job.article_url, stage, exc)
        await asyncio.to_thread(queue.fail, conn, job.article_url, repr(exc))


async def _listen_for_jobs(wake: asyncio.Event) -> None:
    aconn = await psycopg.AsyncConnection.connect(
        get_settings().database_url, autocommit=True
    )
    try:
        await aconn.execute(f"LISTEN {NOTIFY_CHANNEL}")
        async for _notice in aconn.notifies():
            wake.set()
    finally:
        await aconn.close()


async def serve(
    source: NewsFeedSource,
    *,
    workers: int = 4,
    poll_interval_s: float = 5.0,
    drain: bool = False,
) -> dict[str, int]:
    """Run the service: consume the feed, process jobs until stopped.

    drain=True exits once the feed is exhausted and the queue is empty
    (backfill mode). drain=False runs forever (live mode).
    """
    from ticker_news.scraping.config import Settings
    from ticker_news.scraping.fetch import Fetcher
    from ticker_news.scraping.pipeline import DomainLimiter
    from ticker_news.scraping.robots import RobotsCache
    from ticker_news.scraping.store.db import Store

    settings = Settings()
    store = Store(settings.db_dsn)
    store.init_schema()
    fetcher = Fetcher(settings)
    robots = RobotsCache(settings.user_agent) if settings.respect_robots else None
    limiter = DomainLimiter(settings.per_domain, settings.domain_delay_s)

    setup_conn = connect()
    jobs.ensure_schema(setup_conn)
    recovered = jobs.recover_orphans(setup_conn)
    if recovered:
        logger.info("recovered %d orphaned running job(s)", recovered)
    tag_ctx = TagContext.load(setup_conn)
    setup_conn.close()

    class _ScrapeResources:
        pass

    resources = _ScrapeResources()
    resources.fetcher, resources.store, resources.settings = fetcher, store, settings
    resources.limiter, resources.robots = limiter, robots

    wake = asyncio.Event()
    feed_done = asyncio.Event()
    processed = {"done": 0, "failed": 0}

    async def feed_task() -> None:
        feed_conn = connect()
        try:
            async for item in source.stream():
                new = await asyncio.to_thread(jobs.enqueue, feed_conn, item)
                if new:
                    wake.set()
        finally:
            feed_conn.close()
            feed_done.set()

    async def worker_task(worker_id: int) -> None:
        conn = connect(vector=True)
        runners: Mapping[str, StageRunner] = {
            "scrape": lambda job: stages.scrape_stage(job, resources),
            "embed": lambda job: stages.embed_stage(conn, job.article_url),
            "classify": lambda job: stages.classify_stage(conn, job.article_url),
            "tag": lambda job: stages.tag_stage(conn, job.article_url, tag_ctx),
            "insights": lambda job: stages.insights_stage(conn, job.article_url, tag_ctx),
        }
        try:
            while True:
                job = await asyncio.to_thread(jobs.claim, conn)
                if job is None:
                    if drain and feed_done.is_set() and await asyncio.to_thread(
                        jobs.queue_drained, conn
                    ):
                        return
                    wake.clear()
                    try:
                        await asyncio.wait_for(wake.wait(), timeout=poll_interval_s)
                    except asyncio.TimeoutError:
                        pass
                    continue
                await process_article(job, runners, conn=conn)
                processed["done"] += 1
        finally:
            conn.close()

    listener = asyncio.create_task(_listen_for_jobs(wake))
    feed = asyncio.create_task(feed_task())
    pool = [asyncio.create_task(worker_task(i)) for i in range(workers)]

    try:
        await asyncio.gather(*pool)
    finally:
        for task in (listener, feed):
            task.cancel()
        await fetcher.aclose()
        store.close()
    return processed
```

Implementer notes:
- The `scrape` runner returns a coroutine via the lambda — `asyncio.iscoroutinefunction` is False for a lambda returning a coroutine. Fix in process_article: detect awaitables by result, not by function: call the runner, and `if inspect.isawaitable(result): result = await result` for the async case, running SYNC runners via to_thread. Concretely: use a small helper that checks `asyncio.iscoroutinefunction(getattr(runner, "func", runner))` OR simpler — make the runners mapping store plain callables and have process_article do: `result = runner(job); result = await result if inspect.isawaitable(result) else await asyncio.to_thread(lambda: result)`. THINK THIS THROUGH and make the tests pass with both async and sync runners as written in the test file (async def scrape / plain lambdas); the test file is the contract. The cleanest shape: `maybe = runner(job)` inside to_thread is wrong for async — instead: if the call returns an awaitable, await it; else the runner already ran synchronously... but a sync runner doing DB/LLM work must NOT run on the event loop. Recommended final shape:

```python
import inspect

async def _run_stage(runner, job):
    if inspect.iscoroutinefunction(runner):
        return await runner(job)
    result = await asyncio.to_thread(runner, job)
    if inspect.isawaitable(result):  # sync wrapper returned a coroutine (lambda over async fn)
        return await result
    return result
```
  …and in serve's runners, keep the scrape lambda but mark it: `"scrape": (lambda job: stages.scrape_stage(job, resources))` — `to_thread` will call it, it returns a coroutine, `_run_stage` awaits it on the loop. CAUTION: a coroutine created inside to_thread but awaited on the loop is fine (creation is cheap and thread-agnostic). Verify the worker tests pass — they use a real `async def` for scrape and plain lambdas for sync stages.
- `drain` mode with multiple workers: a worker that returns leaves others running — `asyncio.gather` waits for all; each exits when it sees drained+feed_done. Fine.
- KeyboardInterrupt: let it propagate; the finally block closes resources. The CLI wraps `asyncio.run(serve(...))` in try/except KeyboardInterrupt to exit cleanly.

- [ ] **Step 3: CLI commands + tests**

Append to `tests/test_root_cli.py`:

```python
def test_backfill_command_enqueues_csv(monkeypatch):
    captured = {}

    class FakeSource:
        def __init__(self, csv_path, limit=None):
            captured["csv"] = csv_path
            captured["limit"] = limit

    async def fake_serve(source, *, workers, poll_interval_s, drain):
        captured["drain"] = drain
        captured["workers"] = workers
        return {"done": 0, "failed": 0}

    monkeypatch.setattr("ticker_news.ingestion.csv_backfill.CsvBackfillSource", FakeSource)
    monkeypatch.setattr("ticker_news.service.worker.serve", fake_serve)
    result = runner.invoke(cli.app, ["backfill", "--csv", "x.csv", "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert captured == {"csv": "x.csv", "limit": None, "drain": True, "workers": 2}


def test_serve_command_uses_massive_source(monkeypatch):
    captured = {}

    class FakeSource:
        def __init__(self, tickers, *, poll_interval_s, lookback, **kw):
            captured["tickers"] = list(tickers)
            captured["poll"] = poll_interval_s

    async def fake_serve(source, *, workers, poll_interval_s, drain):
        captured["drain"] = drain
        return {"done": 0, "failed": 0}

    monkeypatch.setattr("ticker_news.ingestion.massive_rest.MassiveRestSource", FakeSource)
    monkeypatch.setattr("ticker_news.service.worker.serve", fake_serve)
    monkeypatch.setattr(
        "ticker_news.cli._universe_tickers", lambda: ["NVDA", "AMD"]
    )
    result = runner.invoke(cli.app, ["serve", "--poll-interval", "30"])
    assert result.exit_code == 0, result.output
    assert captured["tickers"] == ["NVDA", "AMD"]
    assert captured["poll"] == 30.0
    assert captured["drain"] is False


def test_jobs_status_command(monkeypatch):
    monkeypatch.setattr("ticker_news.service.jobs.counts", lambda conn: {"pending": 2, "done": 5})
    monkeypatch.setattr("ticker_news.shared.db.connect", lambda **kw: _FakeConn())
    result = runner.invoke(cli.app, ["jobs", "status"])
    assert result.exit_code == 0, result.output
    assert "pending" in result.output and "2" in result.output


class _FakeConn:
    def close(self):
        pass
```

Add to cli.py (lazy imports throughout; `jobs` is a sub-typer):

```python
@app.command()
def serve(
    workers: int = typer.Option(4, min=1, help="Concurrent pipeline workers."),
    poll_interval: float = typer.Option(60.0, help="Feed poll interval, seconds."),
    lookback_hours: float = typer.Option(24.0, help="How far back the first poll reaches."),
    tickers: str | None = typer.Option(None, help="Comma-separated tickers (default: ticker_data table)."),
) -> None:
    """Run the live pipeline: poll the news feed, process articles end to end."""
    import asyncio
    from datetime import timedelta

    from ticker_news.ingestion.massive_rest import MassiveRestSource
    from ticker_news.service.worker import serve as run_service

    universe = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers else _universe_tickers()
    )
    source = MassiveRestSource(
        universe, poll_interval_s=poll_interval, lookback=timedelta(hours=lookback_hours)
    )
    try:
        asyncio.run(run_service(source, workers=workers,
                                poll_interval_s=5.0, drain=False))
    except KeyboardInterrupt:
        typer.echo("stopped.")


def _universe_tickers() -> list[str]:
    from ticker_news.shared.db import connect

    conn = connect()
    try:
        rows = conn.execute("SELECT ticker FROM public.ticker_data ORDER BY ticker").fetchall()
    finally:
        conn.close()
    if not rows:
        raise SystemExit(
            "ticker_data is empty - run `ticker-news load-universe` first or pass --tickers."
        )
    return [r[0] for r in rows]


@app.command()
def backfill(
    csv: str = typer.Option(..., help="News CSV to enqueue and process."),
    limit: int | None = typer.Option(None, help="Enqueue at most N rows."),
    workers: int = typer.Option(4, min=1, help="Concurrent pipeline workers."),
) -> None:
    """Enqueue a news CSV and process it to completion (drain mode)."""
    import asyncio

    from ticker_news.ingestion.csv_backfill import CsvBackfillSource
    from ticker_news.service.worker import serve as run_service

    source = CsvBackfillSource(csv, limit=limit)
    counts = asyncio.run(run_service(source, workers=workers,
                                     poll_interval_s=1.0, drain=True))
    typer.echo(f"backfill complete: {counts}")


jobs_app = typer.Typer(help="Inspect and manage the pipeline job queue.")
app.add_typer(jobs_app, name="jobs")


@jobs_app.command("status")
def jobs_status() -> None:
    """Show queue counts by status."""
    from ticker_news.service import jobs as jobs_mod
    from ticker_news.shared import db

    conn = db.connect()
    try:
        jobs_mod.ensure_schema(conn)
        for status, n in sorted(jobs_mod.counts(conn).items()):
            typer.echo(f"{status:>8}  {n}")
    finally:
        conn.close()


@jobs_app.command("retry")
def jobs_retry(
    url: str | None = typer.Option(None, help="Requeue one failed URL (default: all failed)."),
) -> None:
    """Requeue failed jobs."""
    from ticker_news.service import jobs as jobs_mod
    from ticker_news.shared import db

    conn = db.connect()
    try:
        n = jobs_mod.requeue_failed(conn, url)
        typer.echo(f"requeued {n} job(s).")
    finally:
        conn.close()
```

Implementer note on the jobs_status test: it monkeypatches `ticker_news.shared.db.connect` — the command must therefore resolve connect through the module (`db.connect()`), not a from-import, as written above. Adjust the test or command so they agree; the test file is the contract where reconcilable, but module-attribute resolution is the established lazy pattern.

- [ ] **Step 4: Run everything, commit**

Run: `pytest tests/service tests/test_root_cli.py -q` → all pass (4 worker + 3 stages + 3 unit + CLI tests; db tests skip offline).
Full offline → 110 passed, 1 skipped (96 + 4 worker + 3 CLI + ... verify the real arithmetic from the previous task's actual count and report).

```bash
git add src/ticker_news/service/worker.py tests/service/test_worker.py src/ticker_news/cli.py tests/test_root_cli.py
git commit -m "feat: continuous worker service with serve, backfill, and jobs CLI"
```

---

### Task 6: end-to-end db test + verification sweep + push

**Files:**
- Create: `tests/service/test_serve_db.py` (db-marked end-to-end)

- [ ] **Step 1: The end-to-end test (db-marked)**

A fake feed + fake LLM/network stages, real Postgres queue, real articles table — proves the loop: enqueue → claim → stages run in order → job done.

```python
import psycopg
import pytest

from ticker_news.ingestion.feed import FeedItem
from ticker_news.service import jobs, worker

pytestmark = pytest.mark.db

from tests.scraping.conftest import _connect_test_db


class OneShotFeed:
    def __init__(self, items):
        self.items = items

    async def stream(self):
        for item in self.items:
            yield item


@pytest.fixture
def conn():
    try:
        c = _connect_test_db()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    jobs.ensure_schema(c)
    c.execute("TRUNCATE pipeline_jobs")
    c.commit()
    yield c
    c.execute("TRUNCATE pipeline_jobs")
    c.commit()
    c.close()


async def test_drain_serve_processes_fake_feed(conn, monkeypatch):
    ran = []

    async def fake_scrape(job, resources):
        ran.append(("scrape", job.article_url))
        return "ok"

    def fake_sync(name):
        def _run(conn_, url, *a, **kw):
            ran.append((name, url))
        return _run

    monkeypatch.setattr(worker.stages, "scrape_stage", fake_scrape)
    monkeypatch.setattr(worker.stages, "embed_stage", fake_sync("embed"))
    monkeypatch.setattr(worker.stages, "classify_stage", fake_sync("classify"))
    monkeypatch.setattr(worker.stages, "tag_stage", fake_sync("tag"))
    monkeypatch.setattr(worker.stages, "insights_stage", fake_sync("insights"))
    monkeypatch.setattr(worker.stages.TagContext, "load", classmethod(lambda cls, c: None))
    # serve builds real scraper resources; keep that, it touches no network here.

    feed = OneShotFeed([FeedItem(url="https://example.com/e2e", tickers=["NVDA"])])
    counts = await worker.serve(feed, workers=2, poll_interval_s=0.2, drain=True)

    stages_run = [s for s, _ in ran]
    assert stages_run == ["scrape", "embed", "classify", "tag", "insights"]
    assert jobs.counts(conn).get("done") == 1
```

Implementer notes: `worker.serve` calls `Store(settings.db_dsn)` — under tests, `DATABASE_URL` should point at news_test... it does NOT by default (AppSettings default is the real news DB). The test must monkeypatch the env: `monkeypatch.setenv("DATABASE_URL", tests.scraping.conftest.TEST_DSN)` AND clear the settings cache (`get_settings.cache_clear()`) BEFORE calling serve, so every `connect()`/`Settings()` inside serve hits news_test. Add that to the test up front (import TEST_DSN). Also `Fetcher(settings)` and `RobotsCache` construct but make no network calls until used — scrape is faked, so nothing fires. If `store.init_schema()` or `TagContext.load` need adjusting (e.g. ticker_data missing in news_test — the monkeypatched `TagContext.load` returns None, so no), verify and report.

- [ ] **Step 2: Run db suite (if docker up)**

`docker ps` → if postgres is running: `pytest -m db -q` — all db tests pass (jobs queue + e2e + scraping store). Report counts. If docker is not running, do NOT start it; note it and continue — the offline gates below decide.

- [ ] **Step 3: Verification sweep**

- Full offline: `pytest -m "not db and not integration" -q` → expected count per Task 5 report.
- `ticker-news --help` lists: scrape, embed, classify, tag, load-universe, load-overviews, insights, serve, backfill, jobs. All `--help`s exit 0 (including `ticker-news jobs status --help`).
- Lazy-import check: `python -X importtime -c "import ticker_news.cli" 2>&1 | Select-String "langchain|google|psycopg"` → psycopg may appear (shared.db imports it at module level — acceptable, it's lightweight); langchain/google must NOT.
- Legacy: `python run_scrape.py --help` exits 0.

- [ ] **Step 4: Commit + push**

```bash
git add tests/service/test_serve_db.py
git commit -m "test: end-to-end drain-mode service test against news_test"
git push
```

---

## Out of scope (later plans)

- Plan 4: sentiment LangGraph stage — adds "sentiment" to `jobs.STAGES` between insights and done, one runner entry in worker.serve, one adapter in stages.py.
- Plan 5: Langfuse — `@observe` wrapper around `process_article`, spans per stage, compose services.
- Plan 6: research/ port, legacy deletion, websocket source when the provider is chosen (one new file implementing NewsFeedSource), CLAUDE.md update, final PR.
