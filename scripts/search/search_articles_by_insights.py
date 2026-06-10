"""Insight-level semantic search over ``public.article_insights`` (pgvector).

Where ``search_articles.py`` matches whole-article vectors, this tool works at
the *insight* granularity. Each article was chunked into "insight boxes" by
``extract_insights.py`` and every box was embedded with the same model used for
articles (text-embedding-3-small, 1536-dim), so insights and articles live in
one comparable vector space.

Two modes:

  * **by article** (``--like ID`` / ``--like-url URL``): take every insight of
    the seed article and, for each one, retrieve the top-`k` most similar insights
    elsewhere in the corpus (the seed's own insights are excluded). By default the
    retrieved insights are then *consolidated* into the distinct articles they
    came from — ranked by their best-matching insight and how many of the seed's
    insights they matched ("find related articles via insight overlap"). Pass
    ``--per-insight`` instead to see the per-insight breakdown (which specific
    ideas echo other coverage, with attribution to each matched insight).

  * **by text** (a positional query string): embed the query and return the
    top-`k` most similar insights in the corpus — a direct insight search.

So ``--k`` is the top-k retrieved *per seed insight*; ``--top-articles N`` caps
how many consolidated articles to show (default: all).

Python API:
    from search_articles_by_insights import (
        search_insights, insights_of, search_by_insights, related_articles,
    )

    for hit in search_insights("nvidia data center demand", k=5):
        print(hit.similarity, hit.topic, "->", hit.article_headline)

    # consolidated related articles (top-5 insight matches per seed insight):
    for ra in related_articles(article_id=11259, per_insight_k=5):
        print(ra.best_similarity, ra.matched_insights, ra.article_headline)

    # per-insight breakdown:
    for group in search_by_insights(article_id=11259, k=5):
        print(group.seed.topic)
        for hit in group.hits:
            print("  ", hit.similarity, hit.topic)

CLI:
    python search_articles_by_insights.py "ai chip export controls" --k 5
    python search_articles_by_insights.py --like 11259 --k 5        # consolidated articles
    python search_articles_by_insights.py --like 11259 --k 5 --top-articles 10
    python search_articles_by_insights.py --like 11259 --k 5 --per-insight  # breakdown
    python search_articles_by_insights.py --like 11259 --ticker NVDA --since 2025-01-01

    # only match insights from the 3 months before the seed's own publish date
    # (no lookahead — useful for backtesting):
    python search_articles_by_insights.py --like 11259 --months-before 3
    python search_articles_by_insights.py --like 11259 --months-before 3 --exclusive

Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

# embed_articles lives in the sibling scripts/embedding dir; put it on the path
# so this tool runs directly (without needing PYTHONPATH=scripts/embedding set).
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embedding"),
)

# Reuse the exact query embedder + connection helper from the article embedder,
# so query vectors are produced in the same space as the stored insight vectors.
from embed_articles import embed_query, get_conn  # noqa: E402

# HNSW ANN tuning (pgvector >= 0.8.0). The index returns its ~ef_search nearest
# rows *before* the WHERE filters apply, so a selective filter (a date window, a
# ticker) used to post-filter the candidate set down to nothing. We now enable
# `hnsw.iterative_scan`, which keeps scanning the index until it has enough rows
# satisfying the filter, so a small ef_search is fine. Override with --ef-search
# or the HNSW_EF_SEARCH env var.
EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "40"))
# "relaxed_order" is faster and may return rows slightly out of distance order;
# we re-sort hits by similarity in Python so the displayed ranking is exact.
ITERATIVE_SCAN = os.getenv("HNSW_ITERATIVE_SCAN", "relaxed_order")
# Default cosine-similarity floor: drop weak matches below this score.
DEFAULT_MIN_SIMILARITY = 0.7


def _apply_ann_gucs(cur) -> None:
    """Set the per-query HNSW GUCs (ef_search + iterative scan) on a cursor.

    Tolerates an older pgvector where `hnsw.iterative_scan` doesn't exist by
    rolling back the failed SET and continuing with just ef_search.
    """
    cur.execute(f"SET hnsw.ef_search = {int(EF_SEARCH)};")
    if ITERATIVE_SCAN and ITERATIVE_SCAN.lower() != "off":
        try:
            cur.execute(f"SET hnsw.iterative_scan = {ITERATIVE_SCAN};")
        except Exception:
            cur.connection.rollback()
            cur.execute(f"SET hnsw.ef_search = {int(EF_SEARCH)};")

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date_only(value: str) -> bool:
    """True for a bare 'YYYY-MM-DD' string (no time component)."""
    return bool(_DATE_ONLY.match(value.strip()))


@dataclass
class InsightHit:
    """One matched insight row (joined to its source article's metadata)."""

    insight_id: int
    article_id: int
    source_url: Optional[str]
    article_headline: Optional[str]
    topic: Optional[str]
    insight: Optional[str]
    box_text: Optional[str]
    tickers: List[str]
    published_utc: Optional[datetime]
    similarity: float  # cosine similarity in [-1, 1]; 1.0 == identical direction

    def __repr__(self) -> str:  # concise, useful in a REPL
        topic = (self.topic or "")[:60]
        return f"<{self.similarity:.3f} ai#{self.insight_id} a#{self.article_id} {topic!r}>"


@dataclass
class SeedInsight:
    """One insight of the seed article, used as a query."""

    insight_id: int
    box_index: int
    topic: Optional[str]
    insight: Optional[str]
    box_text: Optional[str]
    embedding: object  # the stored vector (np.float32 array)


@dataclass
class InsightGroup:
    """A seed insight together with its top-k matches elsewhere in the corpus."""

    seed: SeedInsight
    hits: List[InsightHit] = field(default_factory=list)


@dataclass
class RelatedArticle:
    """An article aggregated across the seed's per-insight hits."""

    article_id: int
    source_url: Optional[str]
    article_headline: Optional[str]
    tickers: List[str]
    published_utc: Optional[datetime]
    best_similarity: float       # highest insight-to-insight similarity seen
    matched_insights: int        # how many of the seed's insights matched here
    sum_similarity: float        # sum of (best) similarities, for ranking ties

    def __repr__(self) -> str:
        head = (self.article_headline or "")[:60]
        return (f"<a#{self.article_id} best={self.best_similarity:.3f} "
                f"n={self.matched_insights} {head!r}>")


# --------------------------------------------------------------------------- #
# Core query
# --------------------------------------------------------------------------- #
def _build_filters(
    tickers: Optional[Sequence[str]],
    domain: Optional[str],
    since: Optional[str],
    until: Optional[str],
    segment: Optional[str],
    exclude_article_id: Optional[int],
    until_exclusive: bool,
) -> "tuple[List[str], List[object]]":
    """Build SQL WHERE clauses + params over the joined articles row (alias a)."""
    clauses = ["ai.embedding IS NOT NULL"]
    params: List[object] = []
    if exclude_article_id is not None:
        clauses.append("ai.article_id <> %s")
        params.append(exclude_article_id)
    if tickers:
        clauses.append("a.tickers && %s")
        params.append([t.strip().upper() for t in tickers])
    if segment:
        clauses.append("(a.primary_segment = %s OR a.more_segments @> %s)")
        params.append(segment)
        params.append([segment])
    if domain:
        clauses.append("a.source_domain = %s")
        params.append(domain)
    if since:
        clauses.append("a.published_utc >= %s")
        params.append(since)
    if until:
        if until_exclusive:
            clauses.append("a.published_utc < %s")
        elif _is_date_only(until):
            clauses.append("a.published_utc < (%s::date + interval '1 day')")
        else:
            clauses.append("a.published_utc <= %s")
        params.append(until)
    return clauses, params


def _run_insight_search(
    conn,
    query_vec,
    k: int,
    tickers: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    min_similarity: Optional[float] = None,
    segment: Optional[str] = None,
    exclude_article_id: Optional[int] = None,
    until_exclusive: bool = False,
) -> List[InsightHit]:
    """Run one ANN query against article_insights, mapped to InsightHit objects."""
    clauses, params = _build_filters(
        tickers, domain, since, until, segment, exclude_article_id, until_exclusive
    )
    where = " AND ".join(clauses)
    # `<=>` is cosine distance; similarity = 1 - distance. Pass the query vector
    # for both the score expression and the ORDER BY so it uses the HNSW index.
    sql = (
        "SELECT ai.id, ai.article_id, ai.source_url, ai.article_headline, "
        "       ai.topic, ai.insight, ai.box_text, "
        "       a.tickers, a.published_utc, "
        "       1 - (ai.embedding <=> %s) AS similarity "
        "FROM public.article_insights ai "
        "JOIN public.articles a ON a.id = ai.article_id "
        f"WHERE {where} "
        "ORDER BY ai.embedding <=> %s LIMIT %s"
    )
    args = [query_vec, *params, query_vec, k]
    with conn.cursor() as cur:
        _apply_ann_gucs(cur)  # ef_search + iterative scan so filters still match
        cur.execute(sql, args)
        rows = cur.fetchall()

    hits = [
        InsightHit(
            insight_id=r[0], article_id=r[1], source_url=r[2], article_headline=r[3],
            topic=r[4], insight=r[5], box_text=r[6], tickers=list(r[7] or []),
            published_utc=r[8], similarity=float(r[9]),
        )
        for r in rows
    ]
    # relaxed_order may return rows slightly out of order; sort for an exact rank
    hits.sort(key=lambda h: h.similarity, reverse=True)
    if min_similarity is not None:
        hits = [h for h in hits if h.similarity >= min_similarity]
    return hits


# --------------------------------------------------------------------------- #
# Public: text-query insight search
# --------------------------------------------------------------------------- #
def search_insights(
    query: str,
    k: int = 10,
    tickers: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    min_similarity: Optional[float] = None,
    segment: Optional[str] = None,
    until_exclusive: bool = False,
    conn=None,
) -> List[InsightHit]:
    """Return the top-`k` insights most similar to a free-text `query`."""
    query_vec = embed_query(query)
    own_conn = conn is None
    conn = conn or get_conn()
    try:
        return _run_insight_search(
            conn, query_vec, k, tickers=tickers, domain=domain, since=since,
            until=until, min_similarity=min_similarity, segment=segment,
            until_exclusive=until_exclusive,
        )
    finally:
        if own_conn:
            conn.close()


# --------------------------------------------------------------------------- #
# Public: by-article insight search
# --------------------------------------------------------------------------- #
def _resolve_article_id(conn, *, article_id=None, url=None) -> "tuple[int, dict]":
    """Resolve a seed article to its id + a small metadata dict, by id or url."""
    if article_id is None and not url:
        raise ValueError("provide article_id or url")
    if article_id is not None:
        sql = ("SELECT id, title, url, tickers, published_utc "
               "FROM public.articles WHERE id = %s")
        args = (article_id,)
        label = f"article {article_id}"
    else:
        url = url.strip()
        sql = ("SELECT id, title, url, tickers, published_utc "
               "FROM public.articles WHERE url = %s OR url_canonical = %s LIMIT 1")
        args = (url, url)
        label = f"article with url {url!r}"
    with conn.cursor() as cur:
        cur.execute(sql, args)
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"{label} not found")
    meta = {
        "id": row[0], "title": row[1], "url": row[2],
        "tickers": list(row[3] or []), "published_utc": row[4],
    }
    return row[0], meta


def _seed_window(seed_pub, months_before, exclusive):
    """Compute (since, until) date bounds anchored on a seed article's date.

    Upper bound is the article's own publish time; lower bound is months_before
    earlier (or open if months_before is None). With ``exclusive`` the exact
    timestamp is used, otherwise the calendar day.
    """
    until = seed_pub.isoformat() if exclusive else seed_pub.date().isoformat()
    since = None
    if months_before is not None:
        from dateutil.relativedelta import relativedelta

        anchor = seed_pub if exclusive else seed_pub.date()
        since = (anchor - relativedelta(months=months_before)).isoformat()
    return since, until


def insights_of(conn, article_id: int) -> List[SeedInsight]:
    """Fetch the embedded insight boxes of an article, ordered by box_index."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, box_index, topic, insight, box_text, embedding "
            "FROM public.article_insights "
            "WHERE article_id = %s AND embedding IS NOT NULL "
            "ORDER BY box_index",
            (article_id,),
        )
        rows = cur.fetchall()
    return [
        SeedInsight(insight_id=r[0], box_index=r[1], topic=r[2], insight=r[3],
                    box_text=r[4], embedding=r[5])
        for r in rows
    ]


def search_by_insights(
    article_id: Optional[int] = None,
    url: Optional[str] = None,
    k: int = 5,
    tickers: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    min_similarity: Optional[float] = None,
    segment: Optional[str] = None,
    until_exclusive: bool = False,
    conn=None,
) -> List[InsightGroup]:
    """For each insight of the seed article, return its top-`k` similar insights.

    The seed article's own insights are excluded from every result set. Returns
    one InsightGroup per seed insight, in box order.
    """
    own_conn = conn is None
    conn = conn or get_conn()
    try:
        seed_id, _ = _resolve_article_id(conn, article_id=article_id, url=url)
        seeds = insights_of(conn, seed_id)
        groups: List[InsightGroup] = []
        for seed in seeds:
            hits = _run_insight_search(
                conn, seed.embedding, k, tickers=tickers, domain=domain,
                since=since, until=until, min_similarity=min_similarity,
                segment=segment, exclude_article_id=seed_id,
                until_exclusive=until_exclusive,
            )
            groups.append(InsightGroup(seed=seed, hits=hits))
        return groups
    finally:
        if own_conn:
            conn.close()


def related_articles(
    article_id: Optional[int] = None,
    url: Optional[str] = None,
    k: Optional[int] = None,
    per_insight_k: int = 5,
    tickers: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    min_similarity: Optional[float] = None,
    segment: Optional[str] = None,
    until_exclusive: bool = False,
    conn=None,
) -> List[RelatedArticle]:
    """Consolidate the seed's per-insight hits into ranked related articles.

    Every insight of the seed article is searched for its top ``per_insight_k``
    similar insights; all those hits are then pooled and grouped by their source
    article. Each candidate article is scored by its single best insight-to-insight
    similarity; ties broken by how many of the seed's insights it matched and the
    summed similarity. ``k`` caps how many articles to return (None = all).
    """
    own_conn = conn is None
    conn = conn or get_conn()
    try:
        groups = search_by_insights(
            article_id=article_id, url=url, k=per_insight_k, tickers=tickers,
            domain=domain, since=since, until=until, min_similarity=min_similarity,
            segment=segment, until_exclusive=until_exclusive, conn=conn,
        )
        agg: Dict[int, RelatedArticle] = {}
        for group in groups:
            # only count each candidate article once per seed insight (its best hit)
            best_for_seed: Dict[int, InsightHit] = {}
            for hit in group.hits:
                cur = best_for_seed.get(hit.article_id)
                if cur is None or hit.similarity > cur.similarity:
                    best_for_seed[hit.article_id] = hit
            for aid, hit in best_for_seed.items():
                ra = agg.get(aid)
                if ra is None:
                    agg[aid] = RelatedArticle(
                        article_id=aid, source_url=hit.source_url,
                        article_headline=hit.article_headline,
                        tickers=hit.tickers, published_utc=hit.published_utc,
                        best_similarity=hit.similarity, matched_insights=1,
                        sum_similarity=hit.similarity,
                    )
                else:
                    ra.matched_insights += 1
                    ra.sum_similarity += hit.similarity
                    ra.best_similarity = max(ra.best_similarity, hit.similarity)
        ranked = sorted(
            agg.values(),
            key=lambda r: (r.best_similarity, r.matched_insights, r.sum_similarity),
            reverse=True,
        )
        return ranked if k is None else ranked[:k]
    finally:
        if own_conn:
            conn.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt_meta(tickers: List[str], published_utc: Optional[datetime]) -> str:
    when = published_utc.strftime("%Y-%m-%d %H:%M:%S%z") if published_utc else "-" * 19
    tick = ",".join(tickers) if tickers else "-"
    return f"{when} | [{tick}]"


def _print_hit(hit: InsightHit, rank: int, indent: str) -> None:
    print(f"{indent}{rank}. {hit.similarity:.3f} | {hit.topic or '(no topic)'}")
    if hit.insight:
        print(f"{indent}   {hit.insight}")
    print(f"{indent}   {_fmt_meta(hit.tickers, hit.published_utc)}")
    head = hit.article_headline or "(no headline)"
    print(f"{indent}   from a#{hit.article_id}: {head}")
    if hit.source_url:
        print(f"{indent}   {hit.source_url}")
    print()


def _cmd_text(args) -> None:
    hits = search_insights(
        args.query, k=args.k, tickers=args.tickers, domain=args.domain,
        since=args.since, until=args.until, min_similarity=args.min_similarity,
        segment=args.segment,
    )
    if not hits:
        print(f"No matching insights for {args.query!r}.")
        return
    print(f"\nTop {len(hits)} insight(s) similar to {args.query!r}:\n")
    for rank, hit in enumerate(hits, 1):
        _print_hit(hit, rank, indent="")


def _cmd_article(args) -> None:
    conn = get_conn()
    try:
        try:
            seed_id, meta = _resolve_article_id(
                conn, article_id=args.like, url=args.like_url
            )
        except ValueError as exc:
            raise SystemExit(str(exc))

        print("\nSeed article:")
        print(f"  {meta['title'] or '(no title)'}")
        print(f"  {_fmt_meta(meta['tickers'], meta['published_utc'])}")
        print(f"  {meta['url']}")

        # When anchoring on the seed's own date, derive the window from it so we
        # only match insights published before the seed (no lookahead).
        since, until = args.since, args.until
        until_exclusive = args.exclusive
        if args.months_before is not None or args.exclusive:
            if meta["published_utc"] is None:
                raise SystemExit(
                    "seed article has no published_utc to anchor the window"
                )
            since, until = _seed_window(
                meta["published_utc"], args.months_before, args.exclusive
            )
            edge = "exclusive ts" if args.exclusive else "inclusive day"
            span = (f"{args.months_before} months before"
                    if args.months_before else "up to")
            print(f"\n[window] {since or '(open)'} .. {until} ({span} seed, {edge})")

        common = dict(
            tickers=args.tickers, domain=args.domain, since=since,
            until=until, min_similarity=args.min_similarity, segment=args.segment,
            until_exclusive=until_exclusive,
        )

        # Default: top-k similar insights per seed insight, consolidated into
        # the source articles those retrieved insights came from.
        if not args.per_insight:
            arts = related_articles(
                article_id=seed_id, k=args.top_articles, per_insight_k=args.k,
                conn=conn, **common,
            )
            if not arts:
                print("\nNo related articles found via insight overlap.")
                return
            print(f"\n{len(arts)} consolidated article(s) from top-{args.k} "
                  f"insight matches per seed insight"
                  f"{f' (capped at {args.top_articles})' if args.top_articles else ''}:\n")
            width = len(str(len(arts)))
            for rank, ra in enumerate(arts, 1):
                print(f"{rank:>{width}}. best {ra.best_similarity:.3f} | "
                      f"{ra.matched_insights} insight(s) matched | "
                      f"sum {ra.sum_similarity:.2f}")
                print(f"{'':>{width}}  {ra.article_headline or '(no headline)'}")
                print(f"{'':>{width}}  {_fmt_meta(ra.tickers, ra.published_utc)}")
                if ra.source_url:
                    print(f"{'':>{width}}  {ra.source_url}")
                print()
            return

        # --per-insight: detailed breakdown, top-k matches under each seed insight.
        groups = search_by_insights(article_id=seed_id, k=args.k, conn=conn, **common)
        if not groups:
            print("\nThis article has no embedded insights to search with.")
            return
        n_with_hits = sum(1 for g in groups if g.hits)
        print(f"\n{len(groups)} seed insight(s), {n_with_hits} with matches:\n")
        for gi, group in enumerate(groups, 1):
            seed = group.seed
            print(f"[{gi}] TOPIC: {seed.topic or '(no topic)'}")
            if seed.insight:
                print(f"    INSIGHT: {seed.insight}")
            if not group.hits:
                print("    (no similar insights matched the filters)\n")
                continue
            print(f"    Top {len(group.hits)} similar insight(s):")
            for rank, hit in enumerate(group.hits, 1):
                _print_hit(hit, rank, indent="      ")
    finally:
        conn.close()


def main() -> None:
    global EF_SEARCH
    p = argparse.ArgumentParser(
        description="Insight-level semantic search over article_insights (pgvector)."
    )
    p.add_argument("query", nargs="?", help="free-text insight search query")
    p.add_argument("--like", type=int, default=None, metavar="ID",
                   help="search using the insights of this article id")
    p.add_argument("--like-url", default=None, metavar="URL",
                   help="search using the insights of the article at this URL")
    p.add_argument("-k", "--k", type=int, default=5,
                   help="top-k similar insights retrieved per seed insight "
                        "(also total results in text mode; default 5)")
    p.add_argument("--per-insight", action="store_true",
                   help="with --like/--like-url, show the per-insight breakdown "
                        "instead of the consolidated article list")
    p.add_argument("--top-articles", type=int, default=None, metavar="N",
                   help="cap how many consolidated articles to show "
                        "(default: all)")
    p.add_argument("--ticker", action="append", dest="tickers", metavar="SYM",
                   help="filter matches by ticker (repeatable)")
    p.add_argument("--segment", default=None, metavar="NAME",
                   help="filter matches by AI segment (primary or more_segments)")
    p.add_argument("--domain", help="filter matches by source_domain")
    p.add_argument("--since", "--after", dest="since",
                   help="earliest published_utc of the matched article (YYYY-MM-DD)")
    p.add_argument("--until", "--before", dest="until",
                   help="latest published_utc of the matched article (YYYY-MM-DD)")
    p.add_argument("--months-before", type=int, default=None, metavar="N",
                   help="with --like/--like-url, only match insights from the N "
                        "months before the seed article's own publish date "
                        "(sets --after/--before automatically)")
    p.add_argument("--exclusive", action="store_true",
                   help="make the upper date bound strict on the exact timestamp; "
                        "with --months-before this anchors the window on the seed's "
                        "exact publish time (excludes same-day articles after it)")
    p.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY,
                   help=f"drop matches below this cosine similarity (default "
                        f"{DEFAULT_MIN_SIMILARITY}; pass 0 to show all)")
    p.add_argument("--ef-search", type=int, default=EF_SEARCH, metavar="N",
                   help=f"HNSW ANN candidate breadth (default {EF_SEARCH}); raise "
                        "it if a selective filter returns too few/no results")
    args = p.parse_args()
    EF_SEARCH = args.ef_search  # apply the requested ANN breadth for this run

    has_seed = args.like is not None or args.like_url
    if not has_seed and not args.query:
        p.error("provide a query string, --like ID, or --like-url URL")
    if args.per_insight and not has_seed:
        p.error("--per-insight requires --like or --like-url")
    if args.top_articles is not None and not has_seed:
        p.error("--top-articles requires --like or --like-url")
    if args.months_before is not None and not has_seed:
        p.error("--months-before requires --like or --like-url")
    if args.exclusive and not has_seed:
        p.error("--exclusive requires --like or --like-url (it anchors on the seed date)")

    if has_seed:
        _cmd_article(args)
    else:
        _cmd_text(args)


if __name__ == "__main__":
    main()
