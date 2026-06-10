#!/usr/bin/env python3
"""Attach the day's matching articles to each row of a ticker_range_scan CSV.

For every (date, ticker) row in the input CSV this finds the articles whose
ticker is either the article's ``primary_ticker`` or appears in its
``more_tickers`` (the "primary or other" match) and that landed for that trading
day, namely:

  * articles published on that ET date, plus
  * articles published the previous calendar day AFTER the 16:00 ET close
    (after-hours / overnight news that hits before the next session).

Those articles are written into a new ``articles`` column as a JSON list of
``{id, published_et, title, url, when}`` objects (``when`` is "same-day" or
"prev-after-hours"), with an ``article_count`` column alongside for quick
filtering.

The article<->day match is done on the article's date/time in US market time
(America/New_York) so it lines up with the trading day the scan used.

Usage:
    python attach_articles.py INPUT.csv OUTPUT.csv

Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Set, Tuple

# Regular session closes at 16:00 ET; anything at/after this is "after-hours".
AFTER_HOURS_HOUR = 16

import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"


def read_rows(path: str) -> Tuple[List[dict], List[str]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "date" not in fields or "ticker" not in fields:
        raise SystemExit(f"{path} must have 'date' and 'ticker' columns; got {fields}")
    return rows, fields


def fetch_articles_by_ticker_day(
    tickers: Set[str], start: date, end: date
) -> Dict[Tuple[str, date], List[dict]]:
    """Map (ticker, trading_date) -> [article dicts] for the scan's tickers.

    An article is indexed under every in-scope ticker it touches (primary or
    more_tickers) for the trading day it belongs to: its own ET date, and -- when
    it was published after the 16:00 ET close -- also the next calendar day, so a
    row picks up the prior evening's after-hours news. The window is widened one
    day earlier so the first row can still see its prior-day after-hours news.
    """
    tlist = sorted(tickers)
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id,
                   (published_utc AT TIME ZONE 'America/New_York')::date AS et_date,
                   EXTRACT(HOUR FROM published_utc AT TIME ZONE 'America/New_York')::int
                       AS et_hour,
                   to_char(published_utc AT TIME ZONE 'America/New_York',
                           'YYYY-MM-DD HH24:MI') AS published_et,
                   title, url, primary_ticker, more_tickers
            FROM public.articles
            WHERE (published_utc AT TIME ZONE 'America/New_York')::date
                      BETWEEN %s AND %s
              AND (primary_ticker = ANY(%s) OR more_tickers && %s::text[])
            ORDER BY published_utc
            """,
            (start - timedelta(days=1), end, tlist, tlist),
        )
        rows = cur.fetchall()

    out: Dict[Tuple[str, date], List[dict]] = defaultdict(list)
    for aid, et_date, et_hour, published_et, title, url, primary, more in rows:
        touched = set()
        if primary and primary.upper() in tickers:
            touched.add(primary.upper())
        for t in more or []:
            if t and t.upper() in tickers:
                touched.add(t.upper())
        base = {"id": aid, "published_et": published_et, "title": title, "url": url}
        for t in touched:
            out[(t, et_date)].append({**base, "when": "same-day"})
        if et_hour >= AFTER_HOURS_HOUR:  # after-hours -> also the next trading day
            nxt = et_date + timedelta(days=1)
            for t in touched:
                out[(t, nxt)].append({**base, "when": "prev-after-hours"})
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", help="ticker_range_scan CSV to read")
    p.add_argument("output", help="destination CSV (input + articles columns)")
    args = p.parse_args(argv)

    rows, fields = read_rows(args.input)
    if not rows:
        raise SystemExit("input CSV has no data rows.")

    tickers = {r["ticker"].strip().upper() for r in rows if r.get("ticker")}
    dates = [date.fromisoformat(r["date"]) for r in rows]
    by_day = fetch_articles_by_ticker_day(tickers, min(dates), max(dates))

    out_fields = fields + [f for f in ("article_count", "articles") if f not in fields]
    total = 0
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for r in rows:
            key = (r["ticker"].strip().upper(), date.fromisoformat(r["date"]))
            arts = by_day.get(key, [])
            total += len(arts)
            r["article_count"] = len(arts)
            r["articles"] = json.dumps(arts, ensure_ascii=False)
            writer.writerow(r)

    n_with = sum(1 for r in rows if r["article_count"])
    print(
        f"Wrote {len(rows)} row(s) to {args.output}: "
        f"{n_with} had matching articles ({total} article-attachments total)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
