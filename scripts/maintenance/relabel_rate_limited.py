"""One-time migration: relabel rate-limit failures as retryable.

Rows where the HTTP fetch returned 429 were historically stored with
``status='empty'`` (which reads like a dead-end). They are actually retryable
throttles. This script relabels them to ``status='error', error='rate_limited'``
so the adaptive-throttle re-run (`run_scrape.py --retry-failed`) can drain them.

Idempotent: re-running it just relabels any new 429/empty rows.

Usage (Postgres must be up):
    python scripts/maintenance/relabel_rate_limited.py            # apply
    python scripts/maintenance/relabel_rate_limited.py --dry-run  # count only
"""
import argparse
import os

import psycopg

DSN = os.environ.get("SCRAPER_DB_DSN", "postgresql://scraper:scraper@localhost:5432/news")

_MATCH = "status = 'empty' AND http_status = 429"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Report the count without writing.")
    ap.add_argument("--dsn", default=DSN, help="Postgres DSN (default: $SCRAPER_DB_DSN).")
    args = ap.parse_args()

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        affected = conn.execute(
            f"SELECT count(*) FROM articles WHERE {_MATCH}"
        ).fetchone()[0]
        print(f"[relabel] {affected} rows match (empty + http 429)")
        if args.dry_run or affected == 0:
            return
        conn.execute(
            f"UPDATE articles SET status = 'error', error = 'rate_limited' WHERE {_MATCH}"
        )
        print(f"[relabel] relabeled {affected} rows -> status=error, error=rate_limited")


if __name__ == "__main__":
    main()
