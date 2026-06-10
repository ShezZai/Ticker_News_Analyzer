"""The continuous service: feed consumer + async worker pool over pipeline_jobs.

Each worker owns its own psycopg connection (sync psycopg connections must not
be shared across concurrently-running to_thread calls). One AsyncConnection
LISTENs on the notify channel to wake claimers instantly; a poll interval is
the fallback. process_article is the single seam where the per-article
Langfuse trace attaches in a later phase.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Awaitable, Callable, Mapping, Union

import psycopg

from ticker_news.ingestion.feed import NewsFeedSource
from ticker_news.service import jobs, stages
from ticker_news.service.jobs import DONE, Job, NOTIFY_CHANNEL
from ticker_news.service.stages import StageError, TagContext
from ticker_news.shared.config import get_settings
from ticker_news.shared import db

logger = logging.getLogger(__name__)

StageRunner = Union[Callable[[Job], None], Callable[[Job], Awaitable[object]]]


async def _run_stage(runner: StageRunner, job: Job) -> object:
    """Dispatch a stage runner regardless of whether it is async or sync.

    - True coroutine functions (async def) are awaited directly on the loop.
    - Sync callables run in asyncio.to_thread so they don't block the loop.
    - Sync wrappers that return a coroutine (e.g. a lambda over an async fn)
      are detected by inspect.isawaitable and awaited on the loop after the
      thread has returned the coroutine object.
    """
    if inspect.iscoroutinefunction(runner):
        return await runner(job)
    result = await asyncio.to_thread(runner, job)
    if inspect.isawaitable(result):
        # e.g. serve's "scrape" lambda: `lambda job: stages.scrape_stage(job, resources)`
        # — creating the coroutine is cheap and thread-safe; await it on the loop.
        return await result
    return result


async def process_article(
    job: Job,
    runners: Mapping[str, StageRunner],
    queue=jobs,
    *,
    conn,
) -> None:
    """Drive one article through its remaining stages, advancing after each.

    This is the single seam for the per-article Langfuse trace (Plan 5).
    """
    stage = job.stage
    try:
        while stage != DONE:
            runner = runners[stage]
            result = await _run_stage(runner, job)
            if stage == "scrape" and result == "empty":
                # Nothing extracted — no content for downstream stages.
                queue.advance(conn, job.article_url, DONE)
                return
            stage = jobs.next_stage(stage)
            queue.advance(conn, job.article_url, stage)
    except Exception as exc:
        logger.warning("article %s failed at stage %s: %r", job.article_url, stage, exc)
        queue.fail(conn, job.article_url, repr(exc))


async def _listen_for_jobs(dsn: str, wake: asyncio.Event) -> None:
    """Hold an AsyncConnection and SET wake on every NOTIFY."""
    aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
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
    Returns a summary dict with 'done' and 'failed' counts.
    """
    from ticker_news.scraping.config import Settings as ScraperSettings
    from ticker_news.scraping.fetch import Fetcher
    from ticker_news.scraping.pipeline import DomainLimiter
    from ticker_news.scraping.robots import RobotsCache
    from ticker_news.scraping.store.db import Store

    scraper_settings = ScraperSettings()
    store = Store(scraper_settings.db_dsn)
    store.init_schema()
    fetcher = Fetcher(scraper_settings)
    robots = (
        RobotsCache(scraper_settings.user_agent)
        if scraper_settings.respect_robots
        else None
    )
    limiter = DomainLimiter(scraper_settings.per_domain, scraper_settings.domain_delay_s)

    setup_conn = db.connect()
    jobs.ensure_schema(setup_conn)
    recovered = jobs.recover_orphans(setup_conn)
    if recovered:
        logger.info("recovered %d orphaned running job(s)", recovered)
    tag_ctx = TagContext.load(setup_conn)
    setup_conn.close()

    class _ScrapeResources:
        pass

    resources = _ScrapeResources()
    resources.fetcher = fetcher
    resources.store = store
    resources.settings = scraper_settings
    resources.limiter = limiter
    resources.robots = robots

    wake = asyncio.Event()
    feed_done = asyncio.Event()
    processed: dict[str, int] = {"done": 0, "failed": 0}

    async def feed_task() -> None:
        feed_conn = db.connect()
        try:
            async for item in source.stream():
                new = await asyncio.to_thread(jobs.enqueue, feed_conn, item)
                if new:
                    wake.set()
        finally:
            feed_conn.close()
            feed_done.set()

    async def worker_task(worker_id: int) -> None:
        conn = db.connect(vector=True)

        def _runners() -> dict[str, StageRunner]:
            return {
                "scrape": lambda job: stages.scrape_stage(job, resources),
                "embed": lambda job: stages.embed_stage(conn, job.article_url),
                "classify": lambda job: stages.classify_stage(conn, job.article_url),
                "tag": lambda job: stages.tag_stage(conn, job.article_url, tag_ctx),
                "insights": lambda job: stages.insights_stage(conn, job.article_url, tag_ctx),
            }

        runner_map = _runners()
        try:
            while True:
                job = await asyncio.to_thread(jobs.claim, conn)
                if job is None:
                    if drain and feed_done.is_set():
                        if await asyncio.to_thread(jobs.queue_drained, conn):
                            return
                    wake.clear()
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(wake.wait()), timeout=poll_interval_s
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue
                await process_article(job, runner_map, conn=conn)
                processed["done"] += 1
        finally:
            conn.close()

    dsn = get_settings().database_url
    listener = asyncio.create_task(_listen_for_jobs(dsn, wake))
    feed = asyncio.create_task(feed_task())
    pool = [asyncio.create_task(worker_task(i)) for i in range(workers)]

    try:
        await asyncio.gather(*pool)
    except Exception:
        raise
    finally:
        for task in (listener, feed):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await fetcher.aclose()
        store.close()

    return processed
