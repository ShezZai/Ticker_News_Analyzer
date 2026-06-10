"""Batch sentiment over already-ingested articles (no service required)."""

from __future__ import annotations

from ticker_news.sentiment import store
from ticker_news.service.stages import sentiment_stage
from ticker_news.shared.db import connect

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def run_batch(*, limit: int | None = None, reprocess: bool = False) -> int:
    conn = connect(vector=True)
    try:
        store.ensure_schema(conn)
        not_done = "" if reprocess else (
            "AND NOT EXISTS (SELECT 1 FROM article_sentiment s "
            "WHERE s.article_id = a.id AND s.ticker = a.primary_ticker)"
        )
        lim = f"LIMIT {int(limit)}" if limit else ""
        urls = [r[0] for r in conn.execute(
            f"SELECT a.url FROM public.articles a "
            f"WHERE a.category = 'real news' AND a.primary_ticker IS NOT NULL "
            f"AND a.content IS NOT NULL AND a.content <> '' {not_done} ORDER BY a.published_utc {lim}"
        ).fetchall()]
        conn.commit()
        if reprocess:
            # Delete existing verdicts for the selected articles up front so
            # re-judging actually overwrites rather than being silently skipped
            # by save_verdict's DO NOTHING clause.
            conn.execute(
                "DELETE FROM article_sentiment s "
                "USING public.articles a "
                "WHERE s.article_id = a.id "
                "AND a.url = ANY(%s)",
                (urls,),
            )
            conn.commit()
        done = 0
        bar = tqdm(urls, unit="article") if tqdm else urls
        for url in bar:
            try:
                sentiment_stage(conn, url)
                done += 1
            except Exception as exc:
                print(f"  {url}: {exc!r}")
        print(f"Judged {done}/{len(urls)} article(s).")
        return done
    finally:
        conn.close()
