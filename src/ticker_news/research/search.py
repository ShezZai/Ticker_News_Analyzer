"""Semantic search over the embedded articles (text-embedding-3-small + pgvector).

Embeds a text query with the same model used for indexing and returns the most
similar articles by cosine similarity, using the HNSW index on
``public.articles.embedding``.

Python API:
    from ticker_news.research.search import search, similar_to, similar_to_url

    for hit in search("nvidia data center demand", k=5, tickers=["NVDA"]):
        print(hit.similarity, hit.title)

    for hit in similar_to(11259, k=5):       # "more like this article" (by id)
        print(hit.similarity, hit.url)

    for hit in similar_to_url("https://www.fool.com/...", k=5):  # by URL
        print(hit.similarity, hit.title)

CLI (see ``ticker-news search --help``):
    ticker-news search "nvidia data center demand" --k 5
    ticker-news search "ai chip export controls" --ticker NVDA --ticker AMD
    ticker-news search "rate cuts" --domain fool.com --since 2025-01-01
    ticker-news search --like 11259 --k 10              # similar by article id
    ticker-news search --like-url "https://..." --k 10  # similar by URL

    # similarity to a statement, but dated relative to a linked article:
    ticker-news search --like-url "https://..." \
        --statement "nvidia faces chinese ai competition" --months-before 3

    # a JSON file of scoped statements (each run as its own section):
    ticker-news search --like-url "https://..." \
        --statement statements.json --months-before 3

    # statements.json — an array of {statement, scope, value} objects:
    #   [
    #     {"statement": "...", "scope": "ticker",  "value": "NVDA"},  # only NVDA-tagged
    #     {"statement": "...", "scope": "sector"},                     # whole DB
    #     {"statement": "...", "scope": "segment", "value": "Memory & Storage"}  # by AI segment
    #   ]

Connection comes from DATABASE_URL via ticker_news.shared.config.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

# Query vectors are produced exactly like the stored ones — same model, same
# truncation — via the package embedder.
from ticker_news.embedding.embedder import embed_query
from ticker_news.shared.db import connect

# HNSW ANN tuning (pgvector >= 0.8.0). The index returns its ~ef_search nearest
# rows *before* the WHERE filters apply, so a selective filter (a date window, a
# ticker) used to post-filter the candidate set down to nothing. We enable
# `hnsw.iterative_scan`, which keeps scanning the index until it has enough rows
# satisfying the filter, so a small ef_search is fine. Override with --ef-search
# or the HNSW_EF_SEARCH env var.
EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "40"))
# "relaxed_order" is faster and may return rows slightly out of distance order;
# we re-sort hits by similarity in Python so the displayed ranking is exact.
ITERATIVE_SCAN = os.getenv("HNSW_ITERATIVE_SCAN", "relaxed_order")
# Default cosine-similarity floor: drop weak matches below this score.
DEFAULT_MIN_SIMILARITY = 0.7

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _get_conn():
    """A pgvector-aware connection (numpy arrays bind directly to vector cols)."""
    return connect(vector=True)


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


def _is_date_only(value: str) -> bool:
    """True for a bare 'YYYY-MM-DD' string (no time component)."""
    return bool(_DATE_ONLY.match(value.strip()))


@dataclass
class SearchHit:
    """One search result row."""

    id: int
    url: str
    title: Optional[str]
    source_domain: Optional[str]
    publisher: Optional[str]
    published_utc: Optional[datetime]
    tickers: List[str]
    similarity: float  # cosine similarity in [-1, 1]; 1.0 == identical direction

    def __repr__(self) -> str:  # concise, useful in a REPL
        title = (self.title or "")[:70]
        return f"<{self.similarity:.3f} #{self.id} {title!r}>"


def build_filters(
    *,
    tickers: Optional[Sequence[str]] = None,
    segment: Optional[str] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    until_exclusive: bool = False,
    exclude_id: Optional[int] = None,
) -> tuple[str, list]:
    """Build the optional WHERE clauses for a search — pure, no DB.

    Returns ``(fragment, params)`` where the fragment is either empty or starts
    with ``" AND ..."`` so it appends directly after ``embedding IS NOT NULL``.
    """
    clauses: List[str] = []
    params: list = []
    if tickers:
        clauses.append("tickers && %s")
        params.append([t.strip().upper() for t in tickers])
    if segment:
        # an article belongs to a segment if it's the primary or a secondary one
        clauses.append("(primary_segment = %s OR more_segments @> %s)")
        params.append(segment)
        params.append([segment])
    if domain:
        clauses.append("source_domain = %s")
        params.append(domain)
    if since:
        clauses.append("published_utc >= %s")
        params.append(since)
    if until:
        if until_exclusive:
            # strict upper bound on the exact timestamp (everything strictly before)
            clauses.append("published_utc < %s")
        elif _is_date_only(until):
            # For a bare YYYY-MM-DD upper bound, include the entire day rather than
            # just its midnight, so "before 2025-08-31" covers all of Aug 31.
            clauses.append("published_utc < (%s::date + interval '1 day')")
        else:
            clauses.append("published_utc <= %s")
        params.append(until)
    if exclude_id is not None:
        clauses.append("id <> %s")
        params.append(exclude_id)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def _run_search(
    conn,
    query_vec,
    k: int,
    tickers: Optional[Sequence[str]],
    domain: Optional[str],
    since: Optional[str],
    until: Optional[str],
    min_similarity: Optional[float],
    exclude_id: Optional[int],
    until_exclusive: bool = False,
    segment: Optional[str] = None,
) -> List[SearchHit]:
    """Execute the ANN query and map rows to SearchHit objects."""
    extra, params = build_filters(
        tickers=tickers, segment=segment, domain=domain, since=since,
        until=until, until_exclusive=until_exclusive, exclude_id=exclude_id,
    )
    # `<=>` is cosine distance; similarity = 1 - distance. Pass the query vector
    # for both the score expression and the ORDER BY so it uses the HNSW index.
    sql = (
        f"SELECT id, url, title, source_domain, publisher, published_utc, "
        f"tickers, 1 - (embedding <=> %s) AS similarity "
        f"FROM public.articles WHERE embedding IS NOT NULL{extra} "
        f"ORDER BY embedding <=> %s LIMIT %s"
    )
    args = [query_vec, *params, query_vec, k]

    with conn.cursor() as cur:
        _apply_ann_gucs(cur)  # ef_search + iterative scan so filters still match
        cur.execute(sql, args)
        rows = cur.fetchall()

    hits = [
        SearchHit(
            id=r[0], url=r[1], title=r[2], source_domain=r[3], publisher=r[4],
            published_utc=r[5], tickers=list(r[6] or []), similarity=float(r[7]),
        )
        for r in rows
    ]
    # relaxed_order may return rows slightly out of order; sort for an exact rank
    hits.sort(key=lambda h: h.similarity, reverse=True)
    if min_similarity is not None:
        hits = [h for h in hits if h.similarity >= min_similarity]
    return hits


def search(
    query: str,
    k: int = 10,
    tickers: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    min_similarity: Optional[float] = None,
    until_exclusive: bool = False,
    segment: Optional[str] = None,
    conn=None,
) -> List[SearchHit]:
    """Return the top-`k` articles most similar to `query`.

    Args:
        query:           free-text search string.
        k:               number of results to return.
        tickers:         keep only articles tagged with any of these symbols.
        domain:          restrict to a single source_domain.
        since / until:   published_utc bounds ("YYYY-MM-DD" or full timestamp).
        min_similarity:  drop hits below this cosine similarity.
        until_exclusive: treat `until` as a strict (<) bound on the exact
                         timestamp instead of an inclusive whole-day bound.
        segment:         keep only articles in this AI segment (primary or more).
        conn:            reuse an existing connection (one is opened otherwise).
    """
    query_vec = embed_query(query)
    own_conn = conn is None
    conn = conn or _get_conn()
    try:
        return _run_search(
            conn, query_vec, k, tickers, domain, since, until,
            min_similarity, exclude_id=None, until_exclusive=until_exclusive,
            segment=segment,
        )
    finally:
        if own_conn:
            conn.close()


def get_article(article_id=None, url=None, conn=None) -> SearchHit:
    """Fetch a single article (by id or url) as a SearchHit (similarity=1.0)."""
    if article_id is None and not url:
        raise ValueError("provide article_id or url")
    own_conn = conn is None
    conn = conn or _get_conn()
    try:
        if article_id is not None:
            where, params = "id = %s", (article_id,)
            label = f"article {article_id}"
        else:
            where, params = "url = %s OR url_canonical = %s", (url.strip(), url.strip())
            label = f"article with url {url.strip()!r}"
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, url, title, source_domain, publisher, published_utc, "
                f"tickers FROM public.articles WHERE {where} LIMIT 1",
                params,
            )
            r = cur.fetchone()
        if r is None:
            raise ValueError(f"{label} not found")
        return SearchHit(
            id=r[0], url=r[1], title=r[2], source_domain=r[3], publisher=r[4],
            published_utc=r[5], tickers=list(r[6] or []), similarity=1.0,
        )
    finally:
        if own_conn:
            conn.close()


def _resolve_article(conn, *, article_id=None, url=None):
    """Look up an article by id or url; return (id, embedding, published_utc)."""
    if article_id is not None:
        sql = "SELECT id, embedding, published_utc FROM public.articles WHERE id = %s"
        key, label = article_id, f"article {article_id}"
    else:
        # match either the raw or the canonicalized URL
        sql = (
            "SELECT id, embedding, published_utc FROM public.articles "
            "WHERE url = %s OR url_canonical = %s LIMIT 1"
        )
        url = url.strip()
        key, label = url, f"article with url {url!r}"

    with conn.cursor() as cur:
        cur.execute(sql, (key, key) if article_id is None else (key,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"{label} not found")
    if row[1] is None:
        raise ValueError(f"{label} has no embedding")
    return row[0], row[1], row[2]


def similar_to(
    article_id: int,
    k: int = 10,
    tickers: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    min_similarity: Optional[float] = None,
    until_exclusive: bool = False,
    segment: Optional[str] = None,
    conn=None,
) -> List[SearchHit]:
    """Return the top-`k` articles most similar to an existing article's vector.

    Two queries on purpose: fetch the seed's embedding first, then run a
    separate ANN query ordered by it — a join-embedded form would defeat the
    HNSW index.
    """
    own_conn = conn is None
    conn = conn or _get_conn()
    try:
        found_id, embedding, _ = _resolve_article(conn, article_id=article_id)
        return _run_search(
            conn, embedding, k, tickers, domain, since, until,
            min_similarity, exclude_id=found_id, until_exclusive=until_exclusive,
            segment=segment,
        )
    finally:
        if own_conn:
            conn.close()


def similar_to_url(
    url: str,
    k: int = 10,
    tickers: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    min_similarity: Optional[float] = None,
    until_exclusive: bool = False,
    segment: Optional[str] = None,
    conn=None,
) -> List[SearchHit]:
    """Like `similar_to`, but locate the seed article by its URL.

    Matches against both ``url`` and ``url_canonical``.
    """
    own_conn = conn is None
    conn = conn or _get_conn()
    try:
        found_id, embedding, _ = _resolve_article(conn, url=url)
        return _run_search(
            conn, embedding, k, tickers, domain, since, until,
            min_similarity, exclude_id=found_id, until_exclusive=until_exclusive,
            segment=segment,
        )
    finally:
        if own_conn:
            conn.close()


def seed_window(seed_pub, months_before, exclusive) -> tuple[Optional[str], str]:
    """Compute (since, until) date bounds anchored on a seed article's date.

    Upper bound is the article's own publish time; lower bound is months_before
    earlier (or open if months_before is None). With ``exclusive`` the exact
    timestamp is used, otherwise the calendar day.
    """
    if exclusive:
        until = seed_pub.isoformat()
    else:
        until = seed_pub.date().isoformat()
    since = None
    if months_before is not None:
        from dateutil.relativedelta import relativedelta

        anchor = seed_pub if exclusive else seed_pub.date()
        since = (anchor - relativedelta(months=months_before)).isoformat()
    return since, until


def search_with_seed_dates(
    statement: str,
    article_id: Optional[int] = None,
    url: Optional[str] = None,
    months_before: Optional[int] = None,
    exclusive: bool = False,
    k: int = 10,
    tickers: Optional[Sequence[str]] = None,
    domain: Optional[str] = None,
    min_similarity: Optional[float] = None,
    segment: Optional[str] = None,
    conn=None,
) -> List[SearchHit]:
    """Search articles by similarity to `statement`, dated relative to an article.

    The similarity is computed against `statement` (free text), while the
    published_utc window is anchored on the seed article identified by
    `article_id` or `url`: the upper bound is the article's publish date and the
    lower bound is `months_before` earlier (open-ended if omitted).
    """
    if article_id is None and not url:
        raise ValueError("provide article_id or url to anchor the dates")
    own_conn = conn is None
    conn = conn or _get_conn()
    try:
        seed = get_article(article_id=article_id, url=url, conn=conn)
        if seed.published_utc is None:
            raise ValueError("seed article has no published_utc to anchor the window")
        since, until = seed_window(seed.published_utc, months_before, exclusive)
        return search(
            statement, k=k, tickers=tickers, domain=domain,
            since=since, until=until, min_similarity=min_similarity,
            until_exclusive=exclusive, segment=segment, conn=conn,
        )
    finally:
        if own_conn:
            conn.close()


def scope_to_filter(scope: str, value: Optional[str]) -> dict:
    """Translate a statement item's scope into search() filter kwargs.

      - "sector":  {} (no filter — the entire corpus)
      - "ticker":  {"tickers": [value]} (articles labelled with that ticker)
      - "segment": {"segment": value} (articles whose primary or secondary AI
                   segment is `value`, via the primary_segment/more_segments cols)
    """
    kind = (scope or "sector").strip().lower()
    if kind == "sector":
        return {}
    if kind == "ticker":
        if not value:
            raise ValueError("ticker scope requires a 'value'")
        return {"tickers": [value]}
    if kind == "segment":
        if not value:
            raise ValueError("segment scope requires a 'value'")
        return {"segment": value}
    raise ValueError(f"unknown scope {kind!r} (expected ticker/segment/sector)")


def load_statements(path: str) -> List[dict]:
    """Load a JSON file: an array of {statement, scope, value?} objects."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("statement file must be a JSON array of objects")
    items: List[dict] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict) or not obj.get("statement"):
            raise ValueError(f"item {i}: each entry needs a non-empty 'statement'")
        items.append(
            {
                "statement": obj["statement"],
                "scope": obj.get("scope", "sector"),
                "value": obj.get("value"),
            }
        )
    return items


def format_meta(h: SearchHit) -> str:
    """'published | [tickers]' metadata line for one hit — pure formatting."""
    when = (
        h.published_utc.strftime("%Y-%m-%d %H:%M:%S%z")
        if h.published_utc else "-------------------"
    )
    tickers = ",".join(h.tickers) if h.tickers else "-"
    return f"{when} | [{tickers}]"


def format_hits(hits: List[SearchHit], subject: str) -> str:
    """Human-readable result listing (headline first, metadata + link beneath)."""
    if not hits:
        return f"No matching articles for {subject}."
    lines = [f"Found {len(hits)} article(s) similar to {subject}:", ""]
    width = len(str(len(hits)))
    for rank, h in enumerate(hits, 1):
        lines.append(f"{rank:>{width}}. {h.title or '(no title)'}")
        lines.append(f"{'':>{width}}  {h.similarity:.3f} | {format_meta(h)}")
        lines.append(f"{'':>{width}}  {h.url}")
        lines.append("")
    return "\n".join(lines)


def run_cli(
    query: Optional[str] = None,
    *,
    like: Optional[int] = None,
    like_url: Optional[str] = None,
    statement: Optional[str] = None,
    k: int = 10,
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
    """The `ticker-news search` orchestrator (legacy main(), minus argparse).

    Raises ValueError for every user-input problem; the typer command turns
    those into clean CLI errors.
    """
    global EF_SEARCH
    if ef_search is not None:
        EF_SEARCH = ef_search  # apply the requested ANN breadth for this run

    has_seed = like is not None or bool(like_url)
    if not has_seed and not query and not statement:
        raise ValueError("provide a query string, --like ID, or --like-url URL")
    if statement and not has_seed:
        raise ValueError("--statement requires --like or --like-url for the article dating")
    if months_before is not None and not has_seed:
        raise ValueError("--months-before requires --like or --like-url")

    # When a seed anchors the dates (months-before, or a statement search), resolve
    # the seed's publish time and derive the window from it.
    if has_seed and (months_before is not None or statement):
        seed = get_article(article_id=like, url=like_url)
        if seed.published_utc is None:
            raise ValueError("seed article has no published_utc to anchor the window")
        since, until = seed_window(seed.published_utc, months_before, exclusive)
        edge = "exclusive ts" if exclusive else "inclusive day"
        span = f"{months_before} months before" if months_before else "up to"
        print(f"[window] {since or '(open)'} .. {until} ({span} seed, {edge})")

    date_common = dict(
        k=k, domain=domain, since=since, until=until,
        min_similarity=min_similarity, until_exclusive=exclusive,
    )

    # When a seed article is involved, show it first for context.
    if has_seed:
        seed = get_article(article_id=like, url=like_url)
        label = "Dating-anchor article" if statement else "Seed article"
        print(f"\n{label}:")
        print(f"  {seed.title or '(no title)'}")
        print(f"  {format_meta(seed)}")
        print(f"  {seed.url}")

    # --- statement mode: a literal string or a JSON file of scoped statements ---
    if statement:
        if os.path.isfile(statement):
            try:
                items = load_statements(statement)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"statement file: {exc}") from exc
        else:
            # a bare statement string uses any --ticker/--segment filter as scope
            items = [{"statement": statement, "_literal": True}]

        for idx, item in enumerate(items, 1):
            if item.get("_literal"):      # literal-string case
                scope_filter = {"tickers": tickers, "segment": segment}
                scope_label = "statement"
            else:
                try:
                    scope_filter = scope_to_filter(item["scope"], item.get("value"))
                except ValueError as exc:
                    raise ValueError(f"statement file item {idx}: {exc}") from exc
                kind = (item["scope"] or "sector").lower()
                scope_label = kind if kind == "sector" else f"{kind} {item['value']}"

            print(f"\n=== [{idx}] {scope_label} ===")
            print(f"statement: {item['statement']!r}")
            hits = search(item["statement"], **scope_filter, **date_common)
            print(format_hits(hits, repr(item["statement"])))
        return

    # --- plain similarity / text-query mode ---
    scope = dict(tickers=tickers, segment=segment)
    if like is not None:
        hits = similar_to(like, **scope, **date_common)
    elif like_url:
        hits = similar_to_url(like_url, **scope, **date_common)
    else:
        hits = search(query, **scope, **date_common)

    print()
    print(format_hits(hits, "it" if has_seed else repr(query)))
