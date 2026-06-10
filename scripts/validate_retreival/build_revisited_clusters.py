#!/usr/bin/env python3
"""Build a 'revisited' cluster CSV of TIGHT, single-catalyst events from the DB.

The original article_clusters.csv mixed tight events (Intel Lip-Bu Tan CEO,
Nvidia H20 curbs, Palantir earnings) with broad market-wide narratives (TSMC
Arizona, Trump tariffs). §7 of the validation showed retrieval scores well on the
tight events and poorly on the broad ones.

This rebuilds the ground truth with only tight events, each defined by a precise
keyword anchor + a short active-coverage window, and re-finds its articles from
the DB. Output matches the original schema so it drops straight into
validate_retrieval.py / compare_hybrids.py:

    cluster theme, article ids, news_articles_ids, later_news_article_ids,
    all_less_than_3_months_before_later, news_less_than_3_months_before_later

For each event: the matched articles are the cluster; the latest one that has
embedded insights is the seed (later_news_article_ids); the rest is the ground
truth. The news_* columns hold the current 'real news' subset (small post-
reclassification — the harness defaults to the 'all' column).

Usage:
    python build_revisited_clusters.py                          # -> default path
    python build_revisited_clusters.py -o some/where.csv

Requires NEWS_DB_DSN / DATABASE_URL (content-rich DB).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import psycopg

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"
DEFAULT_OUT = "docs/validations/article_clusters_revisited.csv"

# Earnings-headline terms — for ticker-anchored quarterly events.
EARN = ("(title ILIKE '%earnings%' OR title ILIKE '%revenue%' OR title ILIKE '%results%' "
        "OR title ILIKE '%quarter%' OR title ILIKE '% Q1 %' OR title ILIKE '% Q3 %' "
        "OR title ILIKE '% Q4 %' OR title ILIKE '%beats%' OR title ILIKE '%tops%' "
        "OR title ILIKE '%guidance%' OR title ILIKE '%forecast%' OR title ILIKE '%blowout%')")

# Each event: theme, precise WHERE over articles (alias a), a title_anchor used to
# pick a centrally-on-topic seed, and a tight coverage window [since, until].
EVENTS = [
    ("Stargate $500B AI infrastructure announcement (Jan 2025)",
     "content ILIKE '%stargate%'", "title ILIKE '%stargate%'", "2025-01-21", "2025-01-24"),
    ("DeepSeek R1 model launch & AI selloff (Jan 2025)",
     "content ILIKE '%deepseek%'", "title ILIKE '%deepseek%'", "2025-01-27", "2025-01-29"),
    ("Palantir Q4 2024 earnings & rally (Feb 2025)",
     "a.primary_ticker='PLTR' AND content ILIKE '%palantir%'", "title ILIKE '%palantir%'",
     "2025-02-04", "2025-02-11"),
    ("Nvidia Q3 FY2025 earnings (Nov 2024)",
     f"a.primary_ticker='NVDA' AND {EARN}", "title ILIKE '%nvidia%'", "2024-11-20", "2024-11-25"),
    ("Nvidia Q4 FY2025 earnings (Feb-Mar 2025)",
     f"a.primary_ticker='NVDA' AND {EARN}", "title ILIKE '%nvidia%'", "2025-02-26", "2025-03-03"),
    ("Nvidia Q1 FY2026 earnings (May 2025)",
     f"a.primary_ticker='NVDA' AND {EARN}", "title ILIKE '%nvidia%'", "2025-05-28", "2025-06-02"),
    ("Broadcom Q1 FY2025 earnings (Mar 2025)",
     f"a.primary_ticker='AVGO' AND {EARN}", "title ILIKE '%broadcom%'", "2025-03-06", "2025-03-12"),
    ("Intel CEO Gelsinger departure (Dec 2024)",
     "content ILIKE '%gelsinger%'", "title ILIKE '%intel%' OR title ILIKE '%gelsinger%'",
     "2024-12-02", "2024-12-13"),
    ("Intel names Lip-Bu Tan CEO (Mar 2025)",
     "content ILIKE '%lip-bu tan%'", "title ILIKE '%intel%' OR title ILIKE '%lip-bu tan%'",
     "2025-03-13", "2025-03-20"),
    ("Nvidia H20 China export ban & $5.5B charge (Apr 2025)",
     "content ILIKE '%h20%' AND content ILIKE '%nvidia%'", "title ILIKE '%nvidia%'",
     "2025-04-13", "2025-04-23"),
    ("CoreWeave IPO debut (Mar-Apr 2025)",
     "content ILIKE '%coreweave%'", "title ILIKE '%coreweave%'", "2025-03-28", "2025-04-03"),
]

HEADER = ["cluster theme", "article ids", "news_articles_ids", "later_news_article_ids",
          "all_less_than_3_months_before_later", "news_less_than_3_months_before_later"]


def fetch(cur, where: str, title_anchor: str, since: str, until: str):
    """Matched (id, published_utc, category, has_insights, title_match), by id."""
    sql = (
        "SELECT a.id, a.published_utc, a.category, "
        "  EXISTS (SELECT 1 FROM public.article_insights ai "
        "          WHERE ai.article_id = a.id AND ai.embedding IS NOT NULL) AS has_ins, "
        f"  ({title_anchor}) AS title_match "
        f"FROM public.articles a WHERE ({where}) "
        f"AND (a.published_utc AT TIME ZONE 'UTC')::date BETWEEN '{since}' AND '{until}' "
        "AND a.content IS NOT NULL AND a.embedding IS NOT NULL "
        "ORDER BY a.id"
    )
    cur.execute(sql)
    return cur.fetchall()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", default=DEFAULT_OUT)
    args = p.parse_args(argv)

    conn = psycopg.connect(DB_DSN)
    rows_out = []
    try:
        with conn.cursor() as cur:
            for theme, where, title_anchor, since, until in EVENTS:
                rows = fetch(cur, where, title_anchor, since, until)
                if not rows:
                    print(f"  WARN no matches: {theme}", file=sys.stderr)
                    continue
                ids = [r[0] for r in rows]
                news = [r[0] for r in rows if r[2] == "real news"]
                # seed = latest insight-bearing row whose TITLE carries the event
                # anchor (central to the event); fall back to latest insight-bearing.
                insightful = [r for r in rows if r[3] and r[1] is not None]
                if not insightful:
                    print(f"  WARN no insight-bearing seed: {theme}", file=sys.stderr)
                    continue
                central = [r for r in insightful if r[4]]
                seed = max(central or insightful, key=lambda r: r[1])
                seed_id, seed_pub = seed[0], seed[1]
                # all matched fall inside the tight window (< 3 months before seed),
                # so the "all_before" set is every matched article (seed included,
                # the validator strips it). news_before = its real-news subset.
                all_before = ids
                news_before = news
                rows_out.append({
                    "cluster theme": theme,
                    "article ids": " ".join(map(str, ids)),
                    "news_articles_ids": " ".join(map(str, news)),
                    "later_news_article_ids": str(seed_id),
                    "all_less_than_3_months_before_later": " ".join(map(str, all_before)),
                    "news_less_than_3_months_before_later": " ".join(map(str, news_before)),
                })
                span = f"{rows[0][1].date()}..{rows[-1][1].date()}"
                print(f"  {theme[:52]:<54} n={len(ids):>3} news={len(news):>2} "
                      f"seed=a#{seed_id} ({seed_pub.date()})  span~{span}", file=sys.stderr)
    finally:
        conn.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {len(rows_out)} tight event(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
