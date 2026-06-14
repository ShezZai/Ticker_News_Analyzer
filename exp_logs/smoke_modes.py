"""Read-only smoke test: run the new ticker-scoped precedent modes against the
real DB on a handful of actionable articles. No LLM calls. Verifies the SQL is
valid and the modes return sensible precedent lines."""
import psycopg

from ticker_news.service import stages
from ticker_news.shared.config import get_settings

s = get_settings()
dsn = s.database_url
modes = ["ticker-history", "ticker-history-fallback", "ticker-blend", "ticker-relevant"]

with psycopg.connect(dsn) as conn:
    # pick a few actionable, tagged articles that have insights (eligible for sentiment)
    rows = conn.execute(
        "SELECT a.id, a.primary_ticker "
        "FROM public.articles a "
        "WHERE a.primary_ticker IS NOT NULL AND a.embedding IS NOT NULL "
        "  AND EXISTS (SELECT 1 FROM public.article_insights ai WHERE ai.article_id = a.id) "
        "ORDER BY a.published_utc DESC NULLS LAST LIMIT 5"
    ).fetchall()
    print(f"sample articles: {rows}")
    for aid, tkr in rows:
        print(f"\n=== article {aid} [{tkr}] ===")
        for mode in modes:
            try:
                lines = stages.gather_precedents(conn, aid, source=mode)
                conn.rollback()
                preview = lines[0][:80] if lines else "(none)"
                print(f"  {mode:24s} -> {len(lines):2d} lines | first: {preview}")
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                print(f"  {mode:24s} -> ERROR: {type(exc).__name__}: {exc}")
print("\nsmoke done")
