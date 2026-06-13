"""Classification pipeline: classify articles into content categories.

Resumable: only rows with category IS NULL are processed by default.
--reprocess re-classifies every row; --ids restricts to specific article ids.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Tuple

import psycopg

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from ticker_news.classification.chain import classify_article_finegrained
from ticker_news.classification.variants import FINEGRAINED_CATEGORIES
from ticker_news.shared import observability as obs
from ticker_news.shared.db import connect


def ensure_schema(conn: psycopg.Connection) -> None:
    """Add the category columns + a filter index if they don't already exist.

    `is_act` is the ACT/no-ACT collapse of the fine-grained category written by
    the production classifier; it is NULL for rows classified before the
    fine-grained switch, which downstream gates handle via
    COALESCE(is_act, category = 'real news').
    """
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE public.articles "
            "ADD COLUMN IF NOT EXISTS category        text, "
            "ADD COLUMN IF NOT EXISTS category_reason text, "
            "ADD COLUMN IF NOT EXISTS is_act          boolean"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_category_idx "
            "ON public.articles (category)"
        )
    conn.commit()


def articles_to_process(
    conn: psycopg.Connection, reprocess: bool, limit: Optional[int],
    ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, str, str]]:
    """Return (id, title, content) for articles needing classification.

    Only content-bearing rows are considered. By default rows that already have a
    ``category`` are skipped; ``reprocess`` includes them; ``ids`` restricts to
    specific article ids (and re-classifies them regardless of current category).
    """
    clauses = ["content IS NOT NULL", "char_length(content) > 0"]
    params: List[object] = []
    if ids:
        clauses.append("id = ANY(%s)")
        params.append(list(ids))
    elif not reprocess:
        clauses.append("category IS NULL")
    where = " AND ".join(clauses)
    lim = f"LIMIT {int(limit)}" if limit else ""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, title, content FROM public.articles "
            f"WHERE {where} ORDER BY id {lim}",
            params,
        )
        return cur.fetchall()


def classify_all(
    reprocess: bool = False,
    limit: Optional[int] = None,
    ids: Optional[Sequence[int]] = None,
    workers: int = 8,
    batch_size: int = 200,
) -> int:
    """Classify articles and write the labels back. Returns the number classified.

    Gemini calls run concurrently across ``workers`` threads (network-bound); DB
    writes are batched and committed on the main thread.
    """
    conn = connect()
    try:
        ensure_schema(conn)
        rows = articles_to_process(conn, reprocess, limit, ids=ids)
        if not rows:
            print("Nothing to classify.")
            return 0

        print(f"Classifying {len(rows)} article(s) with {workers} worker(s) "
              f"(single-pass fine-grained; ACT/no-ACT collapse) …")

        done = 0
        n_failed = 0
        counts = {c: 0 for c in FINEGRAINED_CATEGORIES}
        pending: List[tuple] = []  # (category, reason, is_act, id) awaiting UPDATE

        def flush() -> None:
            if not pending:
                return
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE public.articles "
                    "SET category = %s, category_reason = %s, is_act = %s "
                    "WHERE id = %s",
                    pending,
                )
            conn.commit()
            pending.clear()

        def _classify_with_cfg(title, content):
            return classify_article_finegrained(
                title, content, config=obs.chain_config() or None
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_classify_with_cfg, title, content or ""):
                    (aid, title)
                for aid, title, content in rows
            }
            completed = as_completed(futures)
            pbar = tqdm(completed, total=len(futures), unit="article") if tqdm else completed
            for fut in pbar:
                aid, _title = futures[fut]
                try:
                    category, reason, is_act = fut.result()
                except Exception as exc:  # exhausted retries on a single article
                    n_failed += 1
                    print(f"  article {aid}: {exc}")
                    continue
                counts[category] += 1
                done += 1
                pending.append((category, reason, is_act, aid))
                if len(pending) >= batch_size:
                    flush()
                if tqdm and hasattr(pbar, "set_postfix_str"):
                    pbar.set_postfix_str(f"{n_failed} failed")
            flush()

        print(f"Classified {done} article(s); {n_failed} failed.")
        for c in FINEGRAINED_CATEGORIES:
            print(f"  {c:<28} {counts[c]}")
        return done
    finally:
        obs.flush()
        conn.close()
