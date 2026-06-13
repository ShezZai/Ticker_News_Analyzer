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
from datetime import timedelta
from typing import Callable, Optional

import psycopg
from psycopg.types.json import Jsonb

from ticker_news.classification.chain import classify_article_finegrained
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

logger = logging.getLogger(__name__)


class StageError(RuntimeError):
    """A stage failed in a way that should consume a retry attempt."""


# An article is "actionable" — judged for sentiment, eligible as a precedent —
# when the fine-grained classifier flagged it ACT (is_act = True). Rows
# classified before the fine-grained switch have is_act NULL, so we fall back to
# the legacy category == 'real news' label. Keep the SQL and Python forms in
# sync; both encode the same COALESCE(is_act, category = 'real news') rule.
def _actionable_sql(alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    return f"COALESCE({p}is_act, {p}category = 'real news')"


def _is_actionable(is_act: bool | None, category: str | None) -> bool:
    return bool(is_act) if is_act is not None else category == "real news"


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
    """Single-pass fine-grained classify; stores category + reason + is_act.

    Returns the assigned fine-grained category, or None when the article was
    skipped (already classified, or no content).
    """
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
    category, reason, is_act = classify_article_finegrained(
        title, content or "", config=obs.chain_config() or None
    )
    conn.execute(
        "UPDATE public.articles SET category = %s, category_reason = %s, is_act = %s "
        "WHERE id = %s",
        (category, reason, is_act, aid),
    )
    conn.commit()
    return category


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


def _lookback_clause(published, lookback_days: int | None) -> tuple[str, list]:
    """SQL fragment + params for the precedent lower time bound.

    Returns ("", []) when unbounded (lookback_days None or <= 0); otherwise a
    "published_utc >= %s" clause cutting off `lookback_days` before the target.
    Column-unqualified so it suits both the article and joined-insight queries
    (the latter aliases articles as `a`, so callers prefix as needed).
    """
    if not lookback_days or lookback_days <= 0:
        return "", []
    return " AND {col} >= %s", [published - timedelta(days=lookback_days)]


def article_similarity(
    conn: psycopg.Connection,
    article_id: int,
    k: int = 5,
    lookback_days: int | None = None,
    real_news_only: bool = False,
) -> list[str]:
    """Cosine-nearest earlier articles, using the stored embedding.

    Two-step on purpose: fetching the target embedding first lets the planner
    drive the second query through the HNSW index (a join-embedded ORDER BY
    <=> falls back to a sequential scan). Returns display lines for the
    historical-precedent analyst; empty when the target has no embedding or
    no published_utc. `lookback_days` bounds how far back precedents may be;
    `real_news_only` restricts candidates to category='real news'.
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
    lb_sql, lb_params = _lookback_clause(published, lookback_days)
    cat_sql = f" AND {_actionable_sql()}" if real_news_only else ""
    rows = conn.execute(
        "SELECT to_char(published_utc, 'YYYY-MM-DD'), primary_ticker, title "
        "FROM public.articles "
        "WHERE id != %s AND embedding IS NOT NULL AND published_utc < %s"
        + cat_sql
        + lb_sql.format(col="published_utc")
        + " ORDER BY embedding <=> %s LIMIT %s",
        (article_id, published, *lb_params, embedding, k),
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


@dataclass(frozen=True)
class _InsightSource:
    """How a precedent_source reads + labels an insight-box corpus.

    `table` and `label_col` are a fixed allowlist (below), so both are safe to
    interpolate into SQL. `label_col` None means the corpus is unlabelled (no
    tag). `filter_drop` excludes insights this column tags 'DROP'.
    """

    table: str
    label_col: str | None = None
    filter_drop: bool = False
    # When True, an article-level ANN prefilter selects a candidate article subset
    # first, and the insight-box ANN runs only within that subset (two-stage funnel).
    hybrid: bool = False


# precedent_source value -> its corpus + labelling. The two distilled modes read
# the SAME table but tag/filter by a different classification pass: first_label
# (no DROP in the data) vs second_label (DROP filtered out). The '-hybrid' modes
# add an article-level prefilter (see `ranked_precedent_hits`); distilled-hybrid
# uses the two-pass distilled corpus without DROP filtering.
INSIGHT_SOURCES = {
    "insights": _InsightSource("public.article_insights"),
    "distilled-first": _InsightSource(
        "public.distilled_article_insights", "first_label"
    ),
    "distilled-second": _InsightSource(
        "public.distilled_article_insights", "second_label", filter_drop=True
    ),
    "insights-hybrid": _InsightSource("public.article_insights", hybrid=True),
    # distilled-hybrid = two-pass: second_label with DROP-labelled boxes excluded.
    "distilled-hybrid": _InsightSource(
        "public.distilled_article_insights", "second_label", filter_drop=True, hybrid=True
    ),
    # distilled-hybrid-one-pass = first pass only: first_label, no DROP filtering.
    "distilled-hybrid-one-pass": _InsightSource(
        "public.distilled_article_insights", "first_label", hybrid=True
    ),
}


def _article_neighbors(
    conn: psycopg.Connection,
    article_id: int,
    *,
    threshold: float,
    lookback_days: int | None,
    real_news_only: bool,
) -> set[int]:
    """Article-level ANN prefilter for hybrid retrieval.

    Returns the article ids whose stored embedding is within `threshold` cosine
    similarity of the target's, under the precedent discipline (earlier-published,
    lookback window, optional real-news, self excluded). Exact distance filter (no
    LIMIT) over the windowed rows, so the candidate subset is complete.
    """
    target = conn.execute(
        "SELECT embedding, published_utc FROM public.articles "
        "WHERE id = %s AND embedding IS NOT NULL",
        (article_id,),
    ).fetchone()
    if target is None or target[1] is None:
        return set()
    embedding, published = target
    lb_tmpl, lb_params = _lookback_clause(published, lookback_days)
    cat_sql = f" AND {_actionable_sql()}" if real_news_only else ""
    rows = conn.execute(
        "SELECT id FROM public.articles "
        "WHERE id != %s AND embedding IS NOT NULL AND published_utc < %s"
        + cat_sql
        + lb_tmpl.format(col="published_utc")
        + " AND (embedding <=> %s) < %s",
        (article_id, published, *lb_params, embedding, 1.0 - threshold),
    ).fetchall()
    return {r[0] for r in rows}


def _label_tag(label: str | None) -> str:
    """Compact, prompt-ready tag for an insight's classification label."""
    return f"[{label}] " if label else ""


def _ranked_insight_hits(
    conn: psycopg.Connection,
    article_id: int,
    *,
    table: str,
    label_col: str | None,
    filter_drop: bool,
    threshold: float,
    limit: int,
    lookback_days: int | None,
    real_news_only: bool = False,
    restrict_articles: set[int] | None = None,
) -> list[tuple[float, int, str, str | None, str | None, str]]:
    """The deduped, similarity-ranked insight-box precedents for one article.

    Shared core of `insights_similarity` (display) and the cluster eval (recall):
    for each embedded box of the target, ANN-search the corpus under the full
    precedent discipline, union across boxes (dedup on insight id, max similarity
    wins), and return the top `limit` as (sim, article_id, date, ticker, headline,
    excerpt) tuples, ordered by descending similarity. Empty when the target has
    no embedded boxes or no published_utc.
    """
    target = conn.execute(
        "SELECT published_utc FROM public.articles WHERE id = %s", (article_id,)
    ).fetchone()
    if target is None or target[0] is None:
        logger.debug("no published_utc for article %s; skipping insight precedents", article_id)
        return []
    published = target[0]
    box_rows = conn.execute(
        f"SELECT embedding FROM {table} "
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
    # The label column exists only on labelled corpora; select NULL elsewhere so
    # the row shape (and unpack) stays fixed. label_col comes from the allowlisted
    # _InsightSource, so it is safe to interpolate.
    label_sql = f"ai.{label_col}" if label_col else "NULL::text"
    # filter_drop excludes candidates this column tags 'DROP'; IS DISTINCT FROM
    # keeps NULL/unlabelled rows (a plain != would drop them).
    drop_sql = (
        f" AND ai.{label_col} IS DISTINCT FROM 'DROP'" if filter_drop and label_col else ""
    )
    lb_tmpl, lb_params = _lookback_clause(published, lookback_days)
    lb_sql = lb_tmpl.format(col="a.published_utc")
    cat_sql = f" AND {_actionable_sql('a')}" if real_news_only else ""
    # Hybrid prefilter: restrict candidate boxes to the given article subset.
    restrict_sql = " AND ai.article_id = ANY(%s)" if restrict_articles is not None else ""
    restrict_param = [list(restrict_articles)] if restrict_articles is not None else []
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
                f"      {label_sql}, "
                "       1 - (ai.embedding <=> %s) AS similarity "
                f"FROM {table} ai "
                "JOIN public.articles a ON a.id = ai.article_id "
                f"WHERE ai.article_id != %s{restrict_sql} AND ai.embedding IS NOT NULL "
                "  AND a.published_utc < %s"
                f"{cat_sql}{lb_sql}"
                f"  AND (ai.embedding <=> %s) < %s{drop_sql} "
                "ORDER BY ai.embedding <=> %s LIMIT %s",
                [box_embedding, article_id, *restrict_param, published, *lb_params,
                 box_embedding, max_distance, box_embedding, limit],
            )
            for iid, aid, d, ticker, title, insight, topic, label, sim in cur.fetchall():
                sim = float(sim)
                prior = best.get(iid)
                if prior is None or sim > prior[0]:
                    # collapse internal whitespace so an excerpt is a single line,
                    # tag it with its classification label when there is text.
                    body = " ".join((insight or topic or "").split())
                    excerpt = (_label_tag(label) + body) if body else ""
                    best[iid] = (sim, aid, d, ticker, title, excerpt)

    return sorted(best.values(), key=lambda r: r[0], reverse=True)[:limit]


def ranked_precedent_hits(
    conn: psycopg.Connection,
    article_id: int,
    src: _InsightSource,
    *,
    threshold: float,
    limit: int,
    lookback_days: int | None,
    real_news_only: bool = False,
    article_threshold: float | None = None,
) -> list[tuple[float, int, str, str | None, str | None, str]]:
    """Ranked insight-box precedents for an `_InsightSource`, applying the hybrid
    article-level prefilter when `src.hybrid` (article ANN >= `article_threshold`,
    defaulting to `threshold`, selects the candidate subset). Single entry point
    shared by the production path and the cluster eval.
    """
    restrict = None
    if src.hybrid:
        restrict = _article_neighbors(
            conn, article_id,
            threshold=article_threshold if article_threshold is not None else threshold,
            lookback_days=lookback_days, real_news_only=real_news_only,
        )
        if not restrict:
            return []
    return _ranked_insight_hits(
        conn, article_id, table=src.table, label_col=src.label_col,
        filter_drop=src.filter_drop, threshold=threshold, limit=limit,
        lookback_days=lookback_days, real_news_only=real_news_only,
        restrict_articles=restrict,
    )


def insights_similarity(
    conn: psycopg.Connection,
    article_id: int,
    *,
    table: str = "public.article_insights",
    label_col: str | None = None,
    filter_drop: bool = False,
    threshold: float = 0.7,
    limit: int = 40,
    lookback_days: int | None = None,
    real_news_only: bool = False,
    restrict_articles: set[int] | None = None,
) -> list[str]:
    """Earlier articles whose *insight boxes* echo this article's.

    For each embedded insight box of the target article, find earlier
    insight boxes whose cosine similarity exceeds `threshold`; union the hits
    across all boxes (dedup on insight id, keeping the highest similarity), take
    the top `limit` boxes overall, then GROUP them by source article. Each
    returned element is one prior article: a `date [ticker] headline` line, with
    the matching insight excerpt(s) listed beneath when there is more than one
    (inlined after an em dash when there is exactly one). Same precedent
    discipline as article_similarity: only earlier-published sources, and the
    target's own boxes are excluded.

    Empty when the target has no embedded boxes or no published_utc.
    """
    ranked = _ranked_insight_hits(
        conn, article_id, table=table, label_col=label_col, filter_drop=filter_drop,
        threshold=threshold, limit=limit, lookback_days=lookback_days,
        real_news_only=real_news_only, restrict_articles=restrict_articles,
    )

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

    `source` ("article" | "insights" | "distilled-first" | "distilled-second")
    overrides the configured default — the eval harness uses it to compare
    retrieval strategies run-over-run without mutating process env. The insight
    modes are the same flow over a corpus, differing only in labelling/filtering.
    """
    s = get_settings()
    lookback = s.precedent_lookback_days
    real_news_only = s.precedent_real_news_only
    src = INSIGHT_SOURCES.get(source or s.precedent_source)
    if src:
        restrict = None
        if src.hybrid:
            threshold = s.precedent_hybrid_box_threshold
            restrict = _article_neighbors(
                conn, article_id, threshold=s.precedent_hybrid_article_threshold,
                lookback_days=lookback, real_news_only=real_news_only,
            )
            if not restrict:
                return []
        else:
            threshold = s.precedent_insights_threshold
        return insights_similarity(
            conn, article_id, table=src.table, label_col=src.label_col,
            filter_drop=src.filter_drop,
            threshold=threshold,
            limit=s.precedent_insights_limit,
            lookback_days=lookback,
            real_news_only=real_news_only,
            restrict_articles=restrict,
        )
    return article_similarity(conn, article_id, lookback_days=lookback, real_news_only=real_news_only)


def own_article_insights(
    conn: psycopg.Connection,
    article_id: int,
    table: str = "public.article_insights",
    label_col: str | None = None,
    filter_drop: bool = False,
    limit: int = 40,
) -> list[str]:
    """The target article's own insights, in box order, from `table`.

    Shown to the historical-precedent analyst under insight modes so it can
    judge overlap against the matched precedent excerpts. Tagged + DROP-filtered
    per the active source. Whitespace is collapsed so each insight is one line.
    Capped at `limit` boxes: most articles have a handful, but the distilled
    corpus has rare outliers (one article carries ~390 boxes) that would
    otherwise flood the prompt.
    """
    label_sql = f"{label_col}" if label_col else "NULL::text"
    drop_sql = (
        f" AND {label_col} IS DISTINCT FROM 'DROP'" if filter_drop and label_col else ""
    )
    rows = conn.execute(
        f"SELECT insight, topic, {label_sql} FROM {table} "
        f"WHERE article_id = %s{drop_sql} ORDER BY box_index LIMIT %s",
        (article_id, limit),
    ).fetchall()
    out = []
    for insight, topic, label in rows:
        text = " ".join((insight or topic or "").split())
        if text:
            out.append(_label_tag(label) + text)
    return out


def sentiment_stage(
    conn: psycopg.Connection, url: str, precedent_source: str | None = None
) -> dict | None:
    """Judge buy/sell/hold for the article's primary ticker.

    Policy: only ACTIONABLE articles with a tagged primary_ticker are judged;
    everything else skips (cheap, idempotent). Actionable = is_act (the
    fine-grained ACT/no-ACT collapse), falling back to the legacy
    category == 'real news' for rows classified before the fine-grained switch.
    Returns a verdict summary dict, or None when the article was skipped.
    `precedent_source` overrides the configured precedent-retrieval flow.
    """
    row = conn.execute(
        "SELECT id, title, content, category, is_act, primary_ticker, published_utc, "
        "provider_sentiments FROM public.articles WHERE url = %s", (url,),
    ).fetchone()
    if row is None:
        raise StageError(f"article row missing for {url}")
    aid, title, content, category, is_act, ticker, published, provider = row
    if not _is_actionable(is_act, category) or not ticker or not (content or "").strip():
        conn.rollback()
        return None
    if sentiment_store.has_verdict(conn, aid, ticker):
        conn.rollback()
        return None
    settings = get_settings()
    mode = precedent_source or settings.precedent_source
    with obs.stage_span("precedents") as span:
        precedents = gather_precedents(conn, aid, source=mode)
        # Under an insight mode, also surface the target's OWN insights (from the same
        # corpus, same labelling/filtering) so the analyst can judge overlap.
        src = INSIGHT_SOURCES.get(mode)
        own_insights = (
            own_article_insights(
                conn, aid, table=src.table, label_col=src.label_col,
                filter_drop=src.filter_drop, limit=settings.precedent_insights_limit,
            )
            if src else []
        )
        if span is not None:
            span.update(
                output={"n_precedents": len(precedents),
                        "n_own_insights": len(own_insights)},
                metadata={
                    "precedent_source": mode,
                    "threshold": settings.precedent_insights_threshold,
                    "limit": settings.precedent_insights_limit,
                },
            )
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
        # drives the [label] legend in the historical-precedent prompt: only the
        # labelled (distilled) corpora tag their excerpts.
        "precedent_labelled": bool(src and src.label_col),
    }
    conn.rollback()  # release the read transaction; LLM calls can take minutes
    verdict, analyses = judge_article(article, config=obs.chain_config() or None)
    sentiment_store.save_verdict(
        conn, aid, ticker, verdict, analyses, settings.sentiment_verdict_model
    )
    return {"ticker": ticker, "action": verdict.action,
            "confidence": verdict.confidence}
