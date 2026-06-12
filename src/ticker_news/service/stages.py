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
from psycopg.types.json import Jsonb

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
from ticker_news.sentiment import store as sentiment_store
from ticker_news.sentiment.graph import judge_article
from ticker_news.service.jobs import Job
from ticker_news.shared import observability as obs
from ticker_news.shared.config import get_settings
from ticker_news.shared.llm import GEMINI_FLASH

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


def _save_provider_sentiments(store, url: str, source_meta: dict) -> None:
    sentiments = (source_meta or {}).get("sentiments")
    if not sentiments:
        return
    store.conn.execute(
        "UPDATE articles SET provider_sentiments = %s "
        "WHERE url = %s AND provider_sentiments IS NULL",
        (Jsonb(sentiments), url),
    )


async def scrape_stage(job: Job, resources) -> str:
    """Returns the scraper status: 'ok' | 'empty'. Raises StageError on 'error'.

    Already-ok articles are skipped — a re-enqueued URL must not re-fetch (a
    failing re-fetch would overwrite the good row via the store upsert).
    """
    if await asyncio.to_thread(resources.store.exists_ok, job.article_url):
        await asyncio.to_thread(
            _save_provider_sentiments, resources.store, job.article_url, job.source_meta
        )
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
    await asyncio.to_thread(
        _save_provider_sentiments, resources.store, job.article_url, job.source_meta
    )
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


def classify_stage(conn: psycopg.Connection, url: str) -> str | None:
    """Returns the assigned category, or None when the article was skipped."""
    row = conn.execute(
        "SELECT id, title, content, category FROM public.articles WHERE url = %s",
        (url,),
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    aid, title, content, current = row
    if current is not None:
        conn.rollback()
        return None
    if not (content or "").strip():
        conn.rollback()
        return None
    verdict, _confirmed = classify_article(title, content or "", config=obs.chain_config() or None)
    conn.execute(
        "UPDATE public.articles SET category = %s, category_reason = %s WHERE id = %s",
        (verdict.category, verdict.reason or None, aid),
    )
    conn.commit()
    return verdict.category


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
    boxes, model = generate_boxes(content, config=obs.chain_config() or None)
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


def article_similarity(conn: psycopg.Connection, article_id: int, k: int = 5) -> list[str]:
    """Cosine-nearest earlier real-news articles, using the stored embedding.

    Two-step on purpose: fetching the target embedding first lets the planner
    drive the second query through the HNSW index (a join-embedded ORDER BY
    <=> falls back to a sequential scan). Returns display lines for the
    historical-precedent analyst; empty when the target has no embedding or
    no published_utc.
    """
    target = conn.execute(
        "SELECT embedding, published_utc FROM public.articles "
        "WHERE id = %s AND embedding IS NOT NULL",
        (article_id,),
    ).fetchone()
    if target is None or target[1] is None:
        logger.debug("no embedding/published_utc for article %s; skipping precedents", article_id)
        return []
    embedding, published = target
    rows = conn.execute(
        "SELECT to_char(published_utc, 'YYYY-MM-DD'), primary_ticker, title "
        "FROM public.articles "
        "WHERE id != %s AND embedding IS NOT NULL "
        "  AND category = 'real news' AND published_utc < %s "
        "ORDER BY embedding <=> %s LIMIT %s",
        (article_id, published, embedding, k),
    ).fetchall()
    return [f"{d} [{t or '?'}] {title}" for d, t, title in rows]


def _apply_hnsw_gucs(cur, ef_search: int) -> None:
    """Per-transaction HNSW tuning so selective WHERE filters still return rows.

    SET LOCAL via set_config(..., true): scoped to the current transaction, never
    interpolated into SQL, never leaks onto the reused worker connection. The
    iterative_scan statement is savepoint-guarded so an older pgvector that lacks
    the GUC leaves the caller's transaction intact.
    """
    cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
    cur.execute("SAVEPOINT hnsw_guc")
    try:
        cur.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
    except Exception:  # noqa: BLE001 - pgvector < 0.8.0 has no such GUC
        cur.execute("ROLLBACK TO SAVEPOINT hnsw_guc")
    else:
        cur.execute("RELEASE SAVEPOINT hnsw_guc")


def insights_similarity(
    conn: psycopg.Connection,
    article_id: int,
    *,
    threshold: float = 0.7,
    limit: int = 40,
) -> list[str]:
    """Earlier real-news articles whose *insight boxes* echo this article's.

    For each embedded insight box of the target article, find earlier real-news
    insight boxes whose cosine similarity exceeds `threshold`; union the hits
    across all boxes (dedup on insight id, keeping the highest similarity), take
    the top `limit` boxes overall, then GROUP them by source article. Each
    returned element is one prior article: a `date [ticker] headline` line, with
    the matching insight excerpt(s) listed beneath when there is more than one
    (inlined after an em dash when there is exactly one). Same precedent
    discipline as article_similarity: only earlier-published 'real news'
    sources, and the target's own boxes are excluded.

    Empty when the target has no embedded boxes or no published_utc.
    """
    target = conn.execute(
        "SELECT published_utc FROM public.articles WHERE id = %s", (article_id,)
    ).fetchone()
    if target is None or target[0] is None:
        logger.debug("no published_utc for article %s; skipping insight precedents", article_id)
        return []
    published = target[0]
    box_rows = conn.execute(
        "SELECT embedding FROM public.article_insights "
        "WHERE article_id = %s AND embedding IS NOT NULL",
        (article_id,),
    ).fetchall()
    if not box_rows:
        logger.debug("no embedded insight boxes for article %s; skipping precedents", article_id)
        return []

    # Cosine distance bound is the algebraic mirror of the similarity floor
    # (similarity = 1 - distance), applied in SQL so the per-box scan returns
    # only matches we'd keep. One ANN query per target box.
    max_distance = 1.0 - threshold
    # insight_id -> (sim, source_article_id, date, ticker, headline, excerpt)
    best: dict[int, tuple[float, int, str, str | None, str | None, str]] = {}
    with conn.cursor() as cur:
        # The category + date + self filters are selective: HNSW returns ~ef_search
        # candidates BEFORE the WHERE applies, so without iterative_scan the post-
        # filtered yield falls well short of `limit`. GUCs are SET LOCAL (scoped to
        # this transaction), so they never leak onto the shared worker connection.
        _apply_hnsw_gucs(cur, ef_search=max(40, limit))
        for (box_embedding,) in box_rows:
            cur.execute(
                "SELECT ai.id, ai.article_id, to_char(a.published_utc, 'YYYY-MM-DD'), "
                "       a.primary_ticker, a.title, ai.insight, ai.topic, "
                "       1 - (ai.embedding <=> %s) AS similarity "
                "FROM public.article_insights ai "
                "JOIN public.articles a ON a.id = ai.article_id "
                "WHERE ai.article_id != %s AND ai.embedding IS NOT NULL "
                "  AND a.category = 'real news' AND a.published_utc < %s "
                "  AND (ai.embedding <=> %s) < %s "
                "ORDER BY ai.embedding <=> %s LIMIT %s",
                (box_embedding, article_id, published, box_embedding, max_distance,
                 box_embedding, limit),
            )
            for iid, aid, d, ticker, title, insight, topic, sim in cur.fetchall():
                sim = float(sim)
                prior = best.get(iid)
                if prior is None or sim > prior[0]:
                    # collapse internal whitespace so an excerpt is a single line.
                    excerpt = " ".join((insight or topic or "").split())
                    best[iid] = (sim, aid, d, ticker, title, excerpt)

    ranked = sorted(best.values(), key=lambda r: r[0], reverse=True)[:limit]

    # Group the top excerpts by source article, preserving similarity order so an
    # article ranks by its strongest matching insight and its excerpts read best
    # first. Each group renders as one prior article (the header line stays an
    # article, so "SIMILAR PAST ARTICLES" remains accurate).
    groups: dict[int, tuple[str, str | None, str | None, list[str]]] = {}
    order: list[int] = []
    for _sim, aid, d, ticker, title, excerpt in ranked:
        if aid not in groups:
            groups[aid] = (d, ticker, title, [])
            order.append(aid)
        groups[aid][3].append(excerpt)

    lines: list[str] = []
    for aid in order:
        d, ticker, title, excerpts = groups[aid]
        header = f"{d} [{ticker or '?'}] {(title or '').strip()}".rstrip()
        excerpts = [e for e in excerpts if e]
        if len(excerpts) == 1:
            lines.append(f"{header} — {excerpts[0]}")
        elif excerpts:
            sub = "\n".join(f"    - {e}" for e in excerpts)
            lines.append(f"{header}\n{sub}")
        else:
            lines.append(header)
    return lines


def gather_precedents(
    conn: psycopg.Connection, article_id: int, source: str | None = None
) -> list[str]:
    """Historical-precedent lines for the analyst panel.

    `source` ("article" | "insights") overrides the configured default — the
    eval harness uses it to compare retrieval strategies run-over-run without
    mutating process env.
    """
    s = get_settings()
    if (source or s.precedent_source) == "insights":
        return insights_similarity(
            conn, article_id,
            threshold=s.precedent_insights_threshold,
            limit=s.precedent_insights_limit,
        )
    return article_similarity(conn, article_id)


def own_article_insights(conn: psycopg.Connection, article_id: int) -> list[str]:
    """The target article's own distilled insights, in box order.

    Shown to the historical-precedent analyst under insights mode so it can
    judge overlap against the matched precedent excerpts. Whitespace is
    collapsed so each insight is one line.
    """
    rows = conn.execute(
        "SELECT insight, topic FROM public.article_insights "
        "WHERE article_id = %s ORDER BY box_index",
        (article_id,),
    ).fetchall()
    out = []
    for insight, topic in rows:
        text = " ".join((insight or topic or "").split())
        if text:
            out.append(text)
    return out


def sentiment_stage(
    conn: psycopg.Connection, url: str, precedent_source: str | None = None
) -> dict | None:
    """Judge buy/sell/hold for the article's primary ticker.

    Policy: only 'real news' articles with a tagged primary_ticker are judged;
    everything else skips (cheap, idempotent). Returns a verdict summary dict,
    or None when the article was skipped. `precedent_source` overrides the
    configured precedent-retrieval flow (used by the eval harness).
    """
    row = conn.execute(
        "SELECT id, title, content, category, primary_ticker, published_utc, "
        "provider_sentiments FROM public.articles WHERE url = %s", (url,),
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    aid, title, content, category, ticker, published, provider = row
    if category != "real news" or not ticker or not (content or "").strip():
        conn.rollback()
        return None
    if sentiment_store.has_verdict(conn, aid, ticker):
        conn.rollback()
        return None
    mode = precedent_source or get_settings().precedent_source
    precedents = gather_precedents(conn, aid, source=mode)
    # In insights mode, also surface the target's OWN insights so the analyst can
    # judge overlap against the matched precedent excerpts directly.
    own_insights = own_article_insights(conn, aid) if mode == "insights" else []
    provider_sentiment = ""
    if provider and isinstance(provider, dict):
        entry = provider.get(ticker) or {}
        provider_sentiment = entry.get("sentiment") or ""
    article = {
        "ticker": ticker,
        "title": title,
        "content": content,
        "published_utc": published,
        "provider_sentiment": provider_sentiment,
        "precedents": precedents,
        "own_insights": own_insights,
    }
    conn.rollback()  # release the read transaction; LLM calls can take minutes
    verdict, analyses = judge_article(article, config=obs.chain_config() or None)
    sentiment_store.save_verdict(conn, aid, ticker, verdict, analyses, GEMINI_FLASH)
    return {"ticker": ticker, "action": verdict.action,
            "confidence": verdict.confidence}
