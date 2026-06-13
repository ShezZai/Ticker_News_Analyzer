"""The continuous service: feed consumer + async worker pool over pipeline_jobs.

Each worker owns its own psycopg connection (sync psycopg connections must not
be shared across concurrently-running to_thread calls). One AsyncConnection
LISTENs on the notify channel to wake claimers instantly; a poll interval is
the fallback. process_article opens the per-article Langfuse trace; stages
emit child spans.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Awaitable, Callable, Mapping, Union

import psycopg

from ticker_news.classification import pipeline as classify_pipeline
from ticker_news.ingestion.feed import NewsFeedSource
from ticker_news.service import jobs, stages
from ticker_news.service.jobs import DONE, Job, NOTIFY_CHANNEL
from ticker_news.service.stages import PermanentStageError, StageError, TagContext
from ticker_news.shared.config import get_settings
from ticker_news.shared import db, observability as obs
from ticker_news.sentiment import store as sentiment_store

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
) -> bool:
    """Drive one article through its remaining stages, advancing after each.

    This is the single seam for the per-article Langfuse trace (Plan 5).
    Returns True on success, False on failure.
    """
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
                if stage == "scrape" and result == "empty":
                    # Nothing extracted — no content for downstream stages.
                    await asyncio.to_thread(queue.advance, conn, job.article_url, DONE)
                    if root is not None:
                        root.update(output={"final_stage": jobs.DONE, "ok": True, **summary},
                                    metadata=obs.trace_metadata())
                    return True
                stage = jobs.next_stage(stage)
                await asyncio.to_thread(queue.advance, conn, job.article_url, stage)
            if root is not None:
                root.update(output={"final_stage": jobs.DONE, "ok": True, **summary},
                            metadata=obs.trace_metadata())
            return True
        except PermanentStageError as exc:
            logger.warning("article %s permanently failed at stage %s: %r", job.article_url, stage, exc)
            if root is not None:
                root.update(level="ERROR", status_message=repr(exc),
                            output={"final_stage": stage, "ok": False, **summary},
                            metadata=obs.trace_metadata())
            await asyncio.to_thread(queue.fail, conn, job.article_url, repr(exc), permanent=True)
            return False
        except Exception as exc:
            logger.warning("article %s failed at stage %s: %r", job.article_url, stage, exc)
            if root is not None:
                root.update(level="ERROR", status_message=repr(exc),
                            output={"final_stage": stage, "ok": False, **summary},
                            metadata=obs.trace_metadata())
            await asyncio.to_thread(queue.fail, conn, job.article_url, repr(exc))
            return False


async def _listen_for_jobs(dsn: str, wake: asyncio.Event) -> None:
    """Hold an AsyncConnection and SET wake on every NOTIFY."""
    try:
        aconn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
        try:
            await aconn.execute(f"LISTEN {NOTIFY_CHANNEL}")
            async for _notice in aconn.notifies():
                wake.set()
        finally:
            await aconn.close()
    except Exception:
        logger.warning("LISTEN connection lost; relying on poll fallback", exc_info=True)


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
    # One-shot setup store: only used for init_schema(), then closed immediately.
    setup_store = Store(scraper_settings.db_dsn)
    setup_store.init_schema()
    setup_store.close()

    fetcher = Fetcher(scraper_settings)
    robots = (
        RobotsCache(scraper_settings.user_agent)
        if scraper_settings.respect_robots
        else None
    )
    limiter = DomainLimiter(scraper_settings.per_domain, scraper_settings.domain_delay_s)

    setup_conn = db.connect()
    jobs.ensure_schema(setup_conn)
    sentiment_store.ensure_schema(setup_conn)
    classify_pipeline.ensure_schema(setup_conn)  # category + category_reason + is_act
    recovered = jobs.recover_orphans(setup_conn)
    if recovered:
        logger.info("recovered %d orphaned running job(s)", recovered)
    tag_ctx = TagContext.load(setup_conn)
    setup_conn.close()

    class _ScrapeResources:
        pass

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
        except Exception:
            logger.exception("feed task failed")
        finally:
            feed_conn.close()
            feed_done.set()

    async def worker_task(worker_id: int) -> None:
        conn = db.connect(vector=True)
        # Each worker owns its own Store (one sync psycopg connection per worker;
        # concurrent to_thread calls from different workers must not share a
        # single sync connection).
        worker_store = Store(scraper_settings.db_dsn)

        resources = _ScrapeResources()
        resources.fetcher = fetcher
        resources.store = worker_store
        resources.settings = scraper_settings
        resources.limiter = limiter
        resources.robots = robots

        def _runners() -> dict[str, StageRunner]:
            return {
                "scrape": lambda job: stages.scrape_stage(job, resources),
                "embed": lambda job: stages.embed_stage(conn, job.article_url),
                "classify": lambda job: stages.classify_stage(conn, job.article_url),
                "tag": lambda job: stages.tag_stage(conn, job.article_url, tag_ctx),
                "insights": lambda job: stages.insights_stage(conn, job.article_url, tag_ctx),
                "sentiment": lambda job: stages.sentiment_stage(conn, job.article_url),
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
                        await asyncio.wait_for(wake.wait(), timeout=poll_interval_s)
                    except asyncio.TimeoutError:
                        pass
                    continue
                ok = await process_article(job, runner_map, conn=conn)
                processed["done" if ok else "failed"] += 1
        finally:
            worker_store.close()
            conn.close()

    dsn = get_settings().database_url
    listener = asyncio.create_task(_listen_for_jobs(dsn, wake))
    feed = asyncio.create_task(feed_task())
    pool = [asyncio.create_task(worker_task(i)) for i in range(workers)]

    try:
        await asyncio.gather(*pool)
    finally:
        for task in (listener, feed):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await fetcher.aclose()
        # Bounded by the SDK's HTTP timeout — shutdown can take seconds when
        # Langfuse Cloud is unreachable; no-op when disabled.
        obs.flush()

    return processed
