"""Semantic search over the embedded articles (BAAI/bge-m3 + pgvector).

Embeds a text query with the same model used for indexing and returns the most
similar articles by cosine similarity, using the HNSW index on
``public.articles.embedding``.

Python API:
    from search_articles import search, similar_to

    for hit in search("nvidia data center demand", k=5, tickers=["NVDA"]):
        print(hit.similarity, hit.title)

    for hit in similar_to(11259, k=5):       # "more like this article" (by id)
        print(hit.similarity, hit.url)

    for hit in similar_to_url("https://www.fool.com/...", k=5):  # by URL
        print(hit.similarity, hit.title)

CLI:
    python search_articles.py "nvidia data center demand" --k 5
    python search_articles.py "ai chip export controls" --ticker NVDA --ticker AMD
    python search_articles.py "rate cuts" --domain fool.com --since 2025-01-01
    python search_articles.py --like 11259 --k 10            # similar by article id
    python search_articles.py --like-url "https://..." --k 10  # similar by URL

    # similarity to a statement, but dated relative to a linked article:
    python search_articles.py --like-url "https://..." \
        --statement "nvidia faces chinese ai competition" --months-before 3

    # a JSON file of scoped statements (each run as its own section):
    python search_articles.py --like-url "https://..." \
        --statement statements.json --months-before 3

    # statements.json — an array of {statement, scope, value} objects:
    #   [
    #     {"statement": "...", "scope": "ticker",  "value": "NVDA"},  # only NVDA-tagged
    #     {"statement": "...", "scope": "sector"},                     # whole DB
    #     {"statement": "...", "scope": "segment", "value": "Memory & Storage"}  # by AI segment
    #   ]

Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

# Reuse the model name, embedding dim and connection helper from the embedder so
# query vectors are produced exactly like the stored ones.
from embed_articles import EMBED_DIM, MODEL_NAME, get_conn

_model = None  # lazily-loaded SentenceTransformer singleton
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


def _get_model():
    """Load BAAI/bge-m3 once (fp16 on GPU) for encoding queries."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        use_cuda = torch.cuda.is_available()
        model_kwargs = {"torch_dtype": torch.float16} if use_cuda else None
        _model = SentenceTransformer(
            MODEL_NAME,
            device="cuda" if use_cuda else "cpu",
            model_kwargs=model_kwargs,
        )
    return _model


def embed_query(text: str):
    """Return the L2-normalized query embedding (numpy array of EMBED_DIM)."""
    if not text or not text.strip():
        raise ValueError("query text is empty")
    return _get_model().encode(
        text.strip(), normalize_embeddings=True, show_progress_bar=False
    )


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
    clauses = ["embedding IS NOT NULL"]
    params: List[object] = []
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

    where = " AND ".join(clauses)
    # `<=>` is cosine distance; similarity = 1 - distance. Pass the query vector
    # for both the score expression and the ORDER BY so it uses the HNSW index.
    sql = (
        f"SELECT id, url, title, source_domain, publisher, published_utc, "
        f"tickers, 1 - (embedding <=> %s) AS similarity "
        f"FROM public.articles WHERE {where} "
        f"ORDER BY embedding <=> %s LIMIT %s"
    )
    args = [query_vec, *params, query_vec, k]

    with conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()

    hits = [
        SearchHit(
            id=r[0], url=r[1], title=r[2], source_domain=r[3], publisher=r[4],
            published_utc=r[5], tickers=list(r[6] or []), similarity=float(r[7]),
        )
        for r in rows
    ]
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
    conn = conn or get_conn()
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
    conn = conn or get_conn()
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
    """Return the top-`k` articles most similar to an existing article's vector."""
    own_conn = conn is None
    conn = conn or get_conn()
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
    conn = conn or get_conn()
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


def _seed_window(seed_pub, months_before, exclusive):
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
    conn = conn or get_conn()
    try:
        seed = get_article(article_id=article_id, url=url, conn=conn)
        if seed.published_utc is None:
            raise ValueError("seed article has no published_utc to anchor the window")
        since, until = _seed_window(seed.published_utc, months_before, exclusive)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over embedded articles (bge-m3 + pgvector)."
    )
    parser.add_argument("query", nargs="?", help="free-text search query")
    parser.add_argument(
        "--like", type=int, default=None, metavar="ID",
        help="find articles similar to this article id instead of a text query",
    )
    parser.add_argument(
        "--like-url", default=None, metavar="URL",
        help="find articles similar to the article at this URL",
    )
    parser.add_argument(
        "--statement", default=None, metavar="TEXT|FILE",
        help="search by similarity to a statement, dated relative to the "
             "--like/--like-url article. Either a literal string, or a path to a "
             "JSON file: an array of {statement, scope(ticker|segment|sector), "
             "value} objects (ticker->that ticker, sector->whole DB, segment->stub)",
    )
    parser.add_argument("-k", "--k", type=int, default=10, help="number of results")
    parser.add_argument(
        "--ticker", action="append", dest="tickers", metavar="SYM",
        help="filter by ticker (repeatable, e.g. --ticker NVDA --ticker AMD)",
    )
    parser.add_argument(
        "--segment", default=None, metavar="NAME",
        help="filter by AI segment (matches primary_segment or more_segments)",
    )
    parser.add_argument("--domain", help="filter by source_domain")
    parser.add_argument(
        "--since", "--after", dest="since",
        help="earliest published_utc, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--until", "--before", dest="until",
        help="latest published_utc, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--months-before", type=int, default=None, metavar="N",
        help="with --like/--like-url, only search the N months before the seed "
             "article's own publish date (sets --after/--before automatically)",
    )
    parser.add_argument(
        "--min-similarity", type=float, default=None,
        help="drop results below this cosine similarity",
    )
    parser.add_argument(
        "--exclusive", action="store_true",
        help="make the upper date bound strict on the exact timestamp; with "
             "--months-before this anchors the window on the seed's exact "
             "publish time (excludes same-day articles published after it)",
    )
    args = parser.parse_args()

    has_seed = args.like is not None or args.like_url
    if not has_seed and not args.query and not args.statement:
        parser.error("provide a query string, --like ID, or --like-url URL")
    if args.statement and not has_seed:
        parser.error("--statement requires --like or --like-url for the article dating")
    if args.months_before is not None and not has_seed:
        parser.error("--months-before requires --like or --like-url")

    since, until = args.since, args.until
    # When a seed anchors the dates (months-before, or a statement search), resolve
    # the seed's publish time and derive the window from it.
    if has_seed and (args.months_before is not None or args.statement):
        try:
            seed = get_article(article_id=args.like, url=args.like_url)
        except ValueError as exc:
            parser.error(str(exc))
        if seed.published_utc is None:
            parser.error("seed article has no published_utc to anchor the window")
        since, until = _seed_window(
            seed.published_utc, args.months_before, args.exclusive
        )
        edge = "exclusive ts" if args.exclusive else "inclusive day"
        span = f"{args.months_before} months before" if args.months_before else "up to"
        print(f"[window] {since or '(open)'} .. {until} ({span} seed, {edge})")

    def fmt_meta(h: SearchHit) -> str:
        when = (
            h.published_utc.strftime("%Y-%m-%d %H:%M:%S%z")
            if h.published_utc else "-------------------"
        )
        tickers = ",".join(h.tickers) if h.tickers else "-"
        return f"{when} | [{tickers}]"

    def print_hits(hits: List[SearchHit], subject: str) -> None:
        if not hits:
            print(f"No matching articles for {subject}.")
            return
        print(f"Found {len(hits)} article(s) similar to {subject}:\n")
        width = len(str(len(hits)))
        for rank, h in enumerate(hits, 1):
            # headline first, prominently, then metadata + link beneath it
            print(f"{rank:>{width}}. {h.title or '(no title)'}")
            print(f"{'':>{width}}  {h.similarity:.3f} | {fmt_meta(h)}")
            print(f"{'':>{width}}  {h.url}\n")

    date_common = dict(
        k=args.k, domain=args.domain, since=since, until=until,
        min_similarity=args.min_similarity, until_exclusive=args.exclusive,
    )

    # When a seed article is involved, show it first for context.
    if has_seed:
        seed = get_article(article_id=args.like, url=args.like_url)
        label = "Dating-anchor article" if args.statement else "Seed article"
        print(f"\n{label}:")
        print(f"  {seed.title or '(no title)'}")
        print(f"  {fmt_meta(seed)}")
        print(f"  {seed.url}")

    # --- statement mode: a literal string or a JSON file of scoped statements ---
    if args.statement:
        if os.path.isfile(args.statement):
            try:
                items = load_statements(args.statement)
            except (ValueError, json.JSONDecodeError) as exc:
                parser.error(f"statement file: {exc}")
        else:
            # a bare statement string uses any --ticker/--segment filter as scope
            items = [{"statement": args.statement, "_literal": True}]

        for idx, item in enumerate(items, 1):
            if item.get("_literal"):      # literal-string case
                scope_filter = {"tickers": args.tickers, "segment": args.segment}
                scope_label = "statement"
            else:
                try:
                    scope_filter = scope_to_filter(item["scope"], item.get("value"))
                except ValueError as exc:
                    parser.error(f"statement file item {idx}: {exc}")
                kind = (item["scope"] or "sector").lower()
                scope_label = kind if kind == "sector" else f"{kind} {item['value']}"

            print(f"\n=== [{idx}] {scope_label} ===")
            print(f"statement: {item['statement']!r}")
            try:
                hits = search(item["statement"], **scope_filter, **date_common)
            except ValueError as exc:
                parser.error(str(exc))
            print_hits(hits, repr(item["statement"]))
        return

    # --- plain similarity / text-query mode ---
    scope = dict(tickers=args.tickers, segment=args.segment)
    try:
        if args.like is not None:
            hits = similar_to(args.like, **scope, **date_common)
        elif args.like_url:
            hits = similar_to_url(args.like_url, **scope, **date_common)
        else:
            hits = search(args.query, **scope, **date_common)
    except ValueError as exc:
        parser.error(str(exc))

    print()
    print_hits(hits, "it" if has_seed else repr(args.query))


if __name__ == "__main__":
    main()
