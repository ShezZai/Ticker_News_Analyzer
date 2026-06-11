"""Insight-level semantic search over ``public.article_insights`` (pgvector).

Where ``research.search`` matches whole-article vectors, this tool works at
the *insight* granularity. Each article was chunked into "insight boxes" by
the enrichment stage and every box was embedded with the same model used for
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
    from ticker_news.research.insight_search import (
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

CLI (see ``ticker-news search-insights --help``):
    ticker-news search-insights "ai chip export controls" --k 5
    ticker-news search-insights --like 11259 --k 5        # consolidated articles
    ticker-news search-insights --like 11259 --k 5 --top-articles 10
    ticker-news search-insights --like 11259 --k 5 --per-insight  # breakdown
    ticker-news search-insights --like 11259 --ticker NVDA --since 2025-01-01

    # only match insights from the 3 months before the seed's own publish date
    # (no lookahead — useful for backtesting):
    ticker-news search-insights --like 11259 --months-before 3
    ticker-news search-insights --like 11259 --months-before 3 --exclusive

Connection comes from DATABASE_URL via ticker_news.shared.config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

# Query vectors are produced exactly like the stored ones — same model, same
# truncation — via the package embedder.
from ticker_news.embedding.embedder import embed_query

# The whole-article search tool already owns the HNSW ANN plumbing (GUCs via
# transaction-scoped set_config with a savepoint fallback) and the small date /
# seed-window helpers; the legacy insight tool duplicated them verbatim, so we
# import instead of forking.
from ticker_news.research.search import (
    DEFAULT_MIN_SIMILARITY,
    _apply_ann_gucs,
    _is_date_only,
    seed_window,
)
from ticker_news.shared.db import connect


def _get_conn():
    """A pgvector-aware connection (numpy arrays bind directly to vector cols)."""
    return connect(vector=True)


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
def build_filters(
    *,
    tickers: Optional[Sequence[str]] = None,
    segment: Optional[str] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    until_exclusive: bool = False,
    exclude_article_id: Optional[int] = None,
) -> tuple[List[str], list]:
    """Build SQL WHERE clauses + params for the insight search — pure, no DB.

    ``ai`` aliases ``article_insights``; ``a`` the joined ``articles`` row.
    The first clause (non-null embedding) is always present.
    """
    clauses = ["ai.embedding IS NOT NULL"]
    params: list = []
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
    ef_search: Optional[int] = None,
) -> List[InsightHit]:
    """Run one ANN query against article_insights, mapped to InsightHit objects."""
    clauses, params = build_filters(
        tickers=tickers, segment=segment, domain=domain, since=since, until=until,
        until_exclusive=until_exclusive, exclude_article_id=exclude_article_id,
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
        # ef_search + iterative scan so filters still match
        _apply_ann_gucs(cur, ef_search=ef_search)
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
    ef_search: Optional[int] = None,
) -> List[InsightHit]:
    """Return the top-`k` insights most similar to a free-text `query`."""
    query_vec = embed_query(query)
    own_conn = conn is None
    conn = conn or _get_conn()
    try:
        return _run_insight_search(
            conn, query_vec, k, tickers=tickers, domain=domain, since=since,
            until=until, min_similarity=min_similarity, segment=segment,
            until_exclusive=until_exclusive, ef_search=ef_search,
        )
    finally:
        if own_conn:
            conn.close()


# --------------------------------------------------------------------------- #
# Public: by-article insight search
# --------------------------------------------------------------------------- #
def _resolve_article_id(conn, *, article_id=None, url=None) -> tuple[int, dict]:
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
    ef_search: Optional[int] = None,
) -> List[InsightGroup]:
    """For each insight of the seed article, return its top-`k` similar insights.

    The seed article's own insights are excluded from every result set. Returns
    one InsightGroup per seed insight, in box order.
    """
    own_conn = conn is None
    conn = conn or _get_conn()
    try:
        seed_id, _ = _resolve_article_id(conn, article_id=article_id, url=url)
        seeds = insights_of(conn, seed_id)
        groups: List[InsightGroup] = []
        for seed in seeds:
            hits = _run_insight_search(
                conn, seed.embedding, k, tickers=tickers, domain=domain,
                since=since, until=until, min_similarity=min_similarity,
                segment=segment, exclude_article_id=seed_id,
                until_exclusive=until_exclusive, ef_search=ef_search,
            )
            groups.append(InsightGroup(seed=seed, hits=hits))
        return groups
    finally:
        if own_conn:
            conn.close()


def consolidate(groups: List[InsightGroup]) -> List[RelatedArticle]:
    """Pool per-insight hits into ranked RelatedArticle aggregates — pure.

    Each candidate article is counted at most once per seed insight (its best
    hit for that insight). Articles are scored by their single best
    insight-to-insight similarity; ties broken by how many of the seed's
    insights they matched, then by the summed (best) similarity.
    """
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
    return sorted(
        agg.values(),
        key=lambda r: (r.best_similarity, r.matched_insights, r.sum_similarity),
        reverse=True,
    )


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
    ef_search: Optional[int] = None,
) -> List[RelatedArticle]:
    """Consolidate the seed's per-insight hits into ranked related articles.

    Every insight of the seed article is searched for its top ``per_insight_k``
    similar insights; all those hits are then pooled and grouped by their source
    article (see ``consolidate``). ``k`` caps how many articles to return
    (None = all).
    """
    own_conn = conn is None
    conn = conn or _get_conn()
    try:
        groups = search_by_insights(
            article_id=article_id, url=url, k=per_insight_k, tickers=tickers,
            domain=domain, since=since, until=until, min_similarity=min_similarity,
            segment=segment, until_exclusive=until_exclusive, conn=conn,
            ef_search=ef_search,
        )
        ranked = consolidate(groups)
        return ranked if k is None else ranked[:k]
    finally:
        if own_conn:
            conn.close()


# --------------------------------------------------------------------------- #
# Output formatting (pure — every function returns the exact printed block)
# --------------------------------------------------------------------------- #
def format_meta(tickers: List[str], published_utc: Optional[datetime]) -> str:
    """'published | [tickers]' metadata line — pure formatting."""
    when = published_utc.strftime("%Y-%m-%d %H:%M:%S%z") if published_utc else "-" * 19
    tick = ",".join(tickers) if tickers else "-"
    return f"{when} | [{tick}]"


def _hit_lines(hit: InsightHit, rank: int, indent: str) -> List[str]:
    """The lines for one matched insight (ends with a blank separator line)."""
    lines = [f"{indent}{rank}. {hit.similarity:.3f} | {hit.topic or '(no topic)'}"]
    if hit.insight:
        lines.append(f"{indent}   {hit.insight}")
    lines.append(f"{indent}   {format_meta(hit.tickers, hit.published_utc)}")
    head = hit.article_headline or "(no headline)"
    lines.append(f"{indent}   from a#{hit.article_id}: {head}")
    if hit.source_url:
        lines.append(f"{indent}   {hit.source_url}")
    lines.append("")
    return lines


def format_text_results(hits: List[InsightHit], query: str) -> str:
    """The text-query mode listing (top-k insights for a free-text query)."""
    if not hits:
        return f"No matching insights for {query!r}."
    lines = ["", f"Top {len(hits)} insight(s) similar to {query!r}:", ""]
    for rank, hit in enumerate(hits, 1):
        lines.extend(_hit_lines(hit, rank, indent=""))
    return "\n".join(lines)


def format_consolidated(
    arts: List[RelatedArticle], per_insight_k: int, top_articles: Optional[int]
) -> str:
    """The consolidated related-articles listing (default --like view)."""
    if not arts:
        return "\nNo related articles found via insight overlap."
    lines = [
        "",
        f"{len(arts)} consolidated article(s) from top-{per_insight_k} "
        f"insight matches per seed insight"
        f"{f' (capped at {top_articles})' if top_articles else ''}:",
        "",
    ]
    width = len(str(len(arts)))
    for rank, ra in enumerate(arts, 1):
        lines.append(f"{rank:>{width}}. best {ra.best_similarity:.3f} | "
                     f"{ra.matched_insights} insight(s) matched | "
                     f"sum {ra.sum_similarity:.2f}")
        lines.append(f"{'':>{width}}  {ra.article_headline or '(no headline)'}")
        lines.append(f"{'':>{width}}  {format_meta(ra.tickers, ra.published_utc)}")
        if ra.source_url:
            lines.append(f"{'':>{width}}  {ra.source_url}")
        lines.append("")
    return "\n".join(lines)


def format_groups(groups: List[InsightGroup]) -> str:
    """The --per-insight breakdown (top-k matches under each seed insight)."""
    if not groups:
        return "\nThis article has no embedded insights to search with."
    n_with_hits = sum(1 for g in groups if g.hits)
    lines = ["", f"{len(groups)} seed insight(s), {n_with_hits} with matches:", ""]
    for gi, group in enumerate(groups, 1):
        seed = group.seed
        lines.append(f"[{gi}] TOPIC: {seed.topic or '(no topic)'}")
        if seed.insight:
            lines.append(f"    INSIGHT: {seed.insight}")
        if not group.hits:
            lines.append("    (no similar insights matched the filters)")
            lines.append("")
            continue
        lines.append(f"    Top {len(group.hits)} similar insight(s):")
        for rank, hit in enumerate(group.hits, 1):
            lines.extend(_hit_lines(hit, rank, indent="      "))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI orchestrator
# --------------------------------------------------------------------------- #
def run_cli(
    query: Optional[str] = None,
    *,
    like: Optional[int] = None,
    like_url: Optional[str] = None,
    k: int = 5,
    per_insight: bool = False,
    top_articles: Optional[int] = None,
    tickers: Optional[List[str]] = None,
    segment: Optional[str] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    months_before: Optional[int] = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    exclusive: bool = False,
    ef_search: Optional[int] = None,
) -> None:
    """The `ticker-news search-insights` orchestrator (legacy main(), minus argparse).

    Raises ValueError for every user-input problem; the typer command turns
    those into clean CLI errors.
    """
    has_seed = like is not None or bool(like_url)
    if not has_seed and not query:
        raise ValueError("provide a query string, --like ID, or --like-url URL")
    if per_insight and not has_seed:
        raise ValueError("--per-insight requires --like or --like-url")
    if top_articles is not None and not has_seed:
        raise ValueError("--top-articles requires --like or --like-url")
    if months_before is not None and not has_seed:
        raise ValueError("--months-before requires --like or --like-url")
    if exclusive and not has_seed:
        raise ValueError(
            "--exclusive requires --like or --like-url (it anchors on the seed date)"
        )

    # --- text mode: a direct top-k insight search for a free-text query ---
    if not has_seed:
        hits = search_insights(
            query, k=k, tickers=tickers, domain=domain, since=since, until=until,
            min_similarity=min_similarity, segment=segment, ef_search=ef_search,
        )
        print(format_text_results(hits, query))
        return

    # --- article mode: search with the seed article's own insights ---
    conn = _get_conn()
    try:
        seed_id, meta = _resolve_article_id(conn, article_id=like, url=like_url)

        print("\nSeed article:")
        print(f"  {meta['title'] or '(no title)'}")
        print(f"  {format_meta(meta['tickers'], meta['published_utc'])}")
        print(f"  {meta['url']}")

        # When anchoring on the seed's own date, derive the window from it so we
        # only match insights published before the seed (no lookahead).
        if months_before is not None or exclusive:
            if meta["published_utc"] is None:
                raise ValueError(
                    "seed article has no published_utc to anchor the window"
                )
            since, until = seed_window(meta["published_utc"], months_before, exclusive)
            edge = "exclusive ts" if exclusive else "inclusive day"
            span = (f"{months_before} months before"
                    if months_before else "up to")
            print(f"\n[window] {since or '(open)'} .. {until} ({span} seed, {edge})")

        common = dict(
            tickers=tickers, domain=domain, since=since, until=until,
            min_similarity=min_similarity, segment=segment,
            until_exclusive=exclusive, ef_search=ef_search,
        )

        # Default: top-k similar insights per seed insight, consolidated into
        # the source articles those retrieved insights came from.
        if not per_insight:
            arts = related_articles(
                article_id=seed_id, k=top_articles, per_insight_k=k,
                conn=conn, **common,
            )
            print(format_consolidated(arts, k, top_articles))
            return

        # --per-insight: detailed breakdown, top-k matches under each seed insight.
        groups = search_by_insights(article_id=seed_id, k=k, conn=conn, **common)
        print(format_groups(groups))
    finally:
        conn.close()
