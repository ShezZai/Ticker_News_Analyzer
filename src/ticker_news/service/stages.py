"""Per-article stage adapters for the worker service.

Each stage is idempotent: it checks whether its output already exists and
skips silently, so a crashed job re-runs from its recorded stage without
duplicating work. Sync stages run inside asyncio.to_thread; scrape is
natively async.
"""

from __future__ import annotations

import asyncio
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


class PermanentStageError(StageError):
    """A failure that will never succeed on retry — park the job immediately."""


def _stored_error(store, url: str) -> str | None:
    row = store.conn.execute(
        "SELECT error FROM articles WHERE url = %s", (url,)
    ).fetchone()
    return row[0] if row else None


async def scrape_stage(job: Job, resources) -> str:
    """Returns the scraper status: 'ok' | 'empty'. Raises StageError on 'error'.

    Already-ok articles are skipped — a re-enqueued URL must not re-fetch (a
    failing re-fetch would overwrite the good row via the store upsert).
    """
    if await asyncio.to_thread(resources.store.exists_ok, job.article_url):
        return "ok"
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
        reason = await asyncio.to_thread(_stored_error, resources.store, job.article_url)
        if reason == "blocked_by_robots":
            raise PermanentStageError(f"robots blocked {job.article_url}")
        raise StageError(f"scrape returned error for {job.article_url}")
    return status


def embed_stage(conn: psycopg.Connection, url: str) -> None:
    row = conn.execute(
        "SELECT id, title, content, embedding IS NOT NULL FROM public.articles WHERE url = %s",
        (url,),
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    aid, title, content, done = row
    if done:
        conn.rollback()
        return
    text = build_text(title, content)
    if not text:
        conn.rollback()
        return
    vec = embed_texts([text])[0]
    conn.execute(
        "UPDATE public.articles SET embedding = %s WHERE id = %s", (vec, aid)
    )
    conn.commit()


def classify_stage(conn: psycopg.Connection, url: str) -> None:
    row = conn.execute(
        "SELECT id, title, content, category FROM public.articles WHERE url = %s",
        (url,),
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    aid, title, content, current = row
    if current is not None:
        conn.rollback()
        return
    if not (content or "").strip():
        conn.rollback()
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
        conn.rollback()
        return
    # Mirror tag_all's exact text-building convention so service-tagged rows
    # match batch-tagged rows: "{title or ''}\n{content or ''}".
    text = f"{title or ''}\n{content or ''}"
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
        conn.rollback()
        return
    if not (content or "").strip():
        conn.execute(
            "UPDATE public.articles SET insights_extracted_at = now() WHERE id = %s",
            (aid,),
        )
        conn.commit()
        return
    boxes, model = generate_boxes(content)
    # _store_article_boxes commits internally; no second commit needed for the
    # box insert, but the embedding updates below require their own commit.
    _store_article_boxes(
        conn, aid, url, title, content, boxes,
        reprocess=False, quote_threshold=DEFAULT_QUOTE_THRESHOLD,
        annotate=tag_ctx.annotate, model=model,
    )
    # Embed this article's new insight boxes immediately so downstream searches
    # don't need a separate embedding pass.
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
