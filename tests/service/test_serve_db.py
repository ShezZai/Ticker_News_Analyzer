"""End-to-end drain-mode service test against news_test.

Uses a one-shot fake feed + monkeypatched LLM/network stages, but a real
Postgres queue in news_test. Verifies the full loop:
    enqueue → claim → stages run in order → job marked done.

Safety: DATABASE_URL is monkeypatched to TEST_DSN and get_settings cache
is cleared BEFORE serve() runs, so every connect()/Settings() inside serve
hits news_test — never the real news DB.
"""
from __future__ import annotations

import psycopg
import pytest

from ticker_news.ingestion.feed import FeedItem
from ticker_news.service import jobs, worker
from ticker_news.shared.config import get_settings

pytestmark = pytest.mark.db

from tests.scraping.conftest import TEST_DSN, _connect_test_db


class OneShotFeed:
    """Yields a fixed list of FeedItems then stops (for drain mode)."""

    def __init__(self, items: list[FeedItem]) -> None:
        self.items = items

    async def stream(self):
        for item in self.items:
            yield item


@pytest.fixture
def conn():
    """Connect to news_test, ensure pipeline_jobs schema, and clean up around the test."""
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


async def test_drain_serve_processes_fake_feed(conn, monkeypatch):
    """serve() in drain mode: fake feed → real queue → fake stages → job done."""
    # --- Safety: redirect every DB connection inside serve() to news_test ---
    monkeypatch.setenv("DATABASE_URL", TEST_DSN)
    get_settings.cache_clear()

    # Verify the env monkeypatch propagated (scraper Settings reads the same cache).
    from ticker_news.scraping.config import Settings as ScraperSettings
    assert ScraperSettings().db_dsn == TEST_DSN, (
        "Scraper Settings.db_dsn did not pick up the TEST_DSN — "
        "cache_clear() must run before ScraperSettings() is instantiated."
    )

    ran: list[tuple[str, str]] = []

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
    # TagContext.load queries ticker_data which may not exist in news_test;
    # the patched stages don't use it, so return None.
    monkeypatch.setattr(
        worker.stages.TagContext, "load",
        classmethod(lambda cls, c: None),
    )

    feed = OneShotFeed([FeedItem(url="https://example.com/e2e", tickers=["NVDA"])])
    result = await worker.serve(feed, workers=2, poll_interval_s=0.2, drain=True)

    # All five stages ran for the one article (in order).
    stages_run = [s for s, _ in ran]
    assert stages_run == ["scrape", "embed", "classify", "tag", "insights"], (
        f"Expected all 5 stages in order, got: {stages_run}"
    )

    # The job must be marked done in the queue.
    queue_counts = jobs.counts(conn)
    assert queue_counts.get("done") == 1, (
        f"Expected 1 done job, got counts: {queue_counts}"
    )

    # serve() must report 1 processed article.
    assert result.get("done") == 1, f"serve() returned unexpected counts: {result}"
