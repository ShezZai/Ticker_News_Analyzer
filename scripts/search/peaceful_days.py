#!/usr/bin/env python3
"""Build a CSV pool of articles published on "peaceful" market days.

A trading day is "peaceful" when the market has no clear direction / low stress:

  * VIX close < --vix-max (default 25), and
  * the mean ABSOLUTE daily % move of BOTH the S&P 500 and the NASDAQ over the
    trailing --lookback trading days (default 10) is < --max-move (default 1.0%).

Each article is attached to the latest trading day on/before its publish date (so
a weekend headline inherits Friday's regime), and kept only if that day is
peaceful. The output CSV is a pool of articles to test sentiment on calm,
directionless tape.

Data sources:
  * VIX daily close  -> FRED series VIXCLS (free, no key).
  * S&P 500 / NASDAQ -> Massive daily bars (SPY proxy + I:COMP composite). The
    VIX/SPX indices themselves are not licensed on this Massive key, but daily %
    moves of SPY/I:COMP track the indices. (--sp-ticker / --ndq-ticker override.)
  * Articles -> NEWS_DB_DSN / DATABASE_URL (content-rich DB).

Usage:
    python peaceful_days.py                                   # real news, full span
    python peaceful_days.py --start 2025-01-01 --end 2025-03-31
    python peaceful_days.py --categories "real news" "legal solicitation"
    python peaceful_days.py --vix-max 20 --max-move 0.8 --out calm.csv

Requires MASSIVE_API_KEY + NEWS_DB_DSN in env / .env.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import psycopg
import requests
from dotenv import load_dotenv

load_dotenv()

MARKET_TZ = ZoneInfo("America/New_York")
DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"
AGGS_URL = "https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{frm}/{to}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5

DEFAULT_CATEGORIES = ["real news"]
DEFAULT_VIX_MAX = 25.0
DEFAULT_MAX_MOVE = 1.0   # percent
DEFAULT_LOOKBACK = 10    # trading days


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"exhausted retries for {url}")


def fetch_daily_closes(ticker: str, frm: str, to: str, key: str) -> Dict[date, float]:
    """ET-dated daily closes from Massive for one ticker."""
    payload = _get_json(AGGS_URL.format(ticker=ticker, frm=frm, to=to),
                        {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key})
    out: Dict[date, float] = {}
    for b in payload.get("results") or []:
        d = datetime.fromtimestamp(b["t"] / 1000, MARKET_TZ).date()
        out[d] = b["c"]
    if not out:
        raise RuntimeError(f"no daily bars for {ticker} ({frm}..{to})")
    return out


def fetch_vix(frm: str, to: str) -> Dict[date, float]:
    """VIX daily close from FRED VIXCLS."""
    resp = requests.get(FRED_URL, params={"id": "VIXCLS", "cosd": frm, "coed": to},
                        timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    out: Dict[date, float] = {}
    for line in resp.text.splitlines()[1:]:
        ds, _, vs = line.partition(",")
        if vs and vs != ".":
            try:
                out[date.fromisoformat(ds)] = float(vs)
            except ValueError:
                continue
    if not out:
        raise RuntimeError("FRED returned no VIX observations")
    return out


def trailing_abs_move(closes: Dict[date, float], lookback: int) -> Dict[date, float]:
    """Mean |daily % move| over the trailing `lookback` trading days, per day."""
    days = sorted(closes)
    rets: List[Tuple[date, float]] = []
    for i in range(1, len(days)):
        prev, cur = closes[days[i - 1]], closes[days[i]]
        if prev:
            rets.append((days[i], abs(cur / prev - 1.0)))
    out: Dict[date, float] = {}
    for i in range(lookback - 1, len(rets)):
        window = rets[i - lookback + 1:i + 1]
        out[rets[i][0]] = sum(r for _, r in window) / lookback
    return out


# --------------------------------------------------------------------------- #
# Articles
# --------------------------------------------------------------------------- #
def load_articles(start: str, end: str, categories: Sequence[str]):
    """(id, published_et, primary, more_tickers, category, title) in range."""
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, published_utc, primary_ticker, more_tickers, category, title
            FROM public.articles
            WHERE category = ANY(%s)
              AND (published_utc AT TIME ZONE 'America/New_York')::date BETWEEN %s AND %s
            ORDER BY published_utc
            """,
            (list(categories), start, end),
        )
        rows = cur.fetchall()
    out = []
    for aid, pub_utc, primary, more, cat, title in rows:
        more_up = [t.strip().upper() for t in (more or []) if t and t.strip()]
        out.append((aid, pub_utc.astimezone(MARKET_TZ), (primary or "").strip().upper(),
                    more_up, cat, title or ""))
    return out


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None, help="ET date YYYY-MM-DD (default: earliest article)")
    p.add_argument("--end", default=None, help="ET date YYYY-MM-DD (default: latest article)")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                   help=f"article categories (default: {DEFAULT_CATEGORIES})")
    p.add_argument("--vix-max", type=float, default=DEFAULT_VIX_MAX)
    p.add_argument("--max-move", type=float, default=DEFAULT_MAX_MOVE,
                   help="max trailing mean |daily move| in PERCENT (default 1.0)")
    p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    p.add_argument("--sp-ticker", default="SPY", help="S&P 500 proxy (default SPY)")
    p.add_argument("--ndq-ticker", default="I:COMP", help="NASDAQ proxy (default I:COMP)")
    p.add_argument("--require", choices=["both", "either"], default="both",
                   help="require the move threshold on both indices or either (default both)")
    p.add_argument("--out", default="docs/validations/peaceful_days_articles.csv")
    args = p.parse_args(argv)

    key = os.getenv("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is not set (put it in .env).")

    # resolve the date range from the articles if not given
    if not (args.start and args.end):
        with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT min((published_utc AT TIME ZONE 'America/New_York')::date), "
                "       max((published_utc AT TIME ZONE 'America/New_York')::date) "
                "FROM public.articles WHERE category = ANY(%s)", (list(args.categories),))
            lo, hi = cur.fetchone()
        args.start = args.start or lo.isoformat()
        args.end = args.end or hi.isoformat()

    # fetch market data with enough lead for the trailing window (~2x cal days)
    lead = date.fromisoformat(args.start) - timedelta(days=args.lookback * 2 + 10)
    frm = lead.isoformat()
    print(f"fetching market data {frm} .. {args.end} "
          f"(SPY/{args.ndq_ticker}/VIX) …", file=sys.stderr)
    sp = fetch_daily_closes(args.sp_ticker, frm, args.end, key)
    nq = fetch_daily_closes(args.ndq_ticker, frm, args.end, key)
    vix = fetch_vix(frm, args.end)
    sp_mae = trailing_abs_move(sp, args.lookback)
    nq_mae = trailing_abs_move(nq, args.lookback)

    # classify every trading day (canonical calendar = SPY days)
    trading_days = sorted(sp)
    thr = args.max_move / 100.0
    peaceful: Dict[date, Tuple[float, float, float]] = {}
    for d in trading_days:
        if d not in vix or d not in sp_mae or d not in nq_mae:
            continue
        sp_ok, nq_ok = sp_mae[d] < thr, nq_mae[d] < thr
        move_ok = (sp_ok and nq_ok) if args.require == "both" else (sp_ok or nq_ok)
        if vix[d] < args.vix_max and move_ok:
            peaceful[d] = (vix[d], sp_mae[d], nq_mae[d])
    print(f"peaceful trading days: {len(peaceful)} / {len(trading_days)} in range "
          f"(VIX<{args.vix_max}, {args.require} index trailing-{args.lookback}d "
          f"mean|move|<{args.max_move}%)", file=sys.stderr)

    def classify_day(d: date) -> Optional[date]:
        i = bisect.bisect_right(trading_days, d) - 1   # latest trading day <= d
        return trading_days[i] if i >= 0 else None

    # filter articles
    articles = load_articles(args.start, args.end, args.categories)
    rows = []
    for aid, pub_et, primary, more, cat, title in articles:
        md = classify_day(pub_et.date())
        if md is None or md not in peaceful:
            continue
        v, spm, nqm = peaceful[md]
        rows.append({
            "article_id": aid,
            "published_et": pub_et.strftime("%Y-%m-%d %H:%M"),
            "market_date": md.isoformat(),
            "primary_ticker": primary,
            "more_tickers": "|".join(more),
            "category": cat,
            "vix": round(v, 2),
            "sp500_mae_10d_pct": round(spm * 100, 3),
            "nasdaq_mae_10d_pct": round(nqm * 100, 3),
            "title": title,
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [
            "article_id", "published_et", "market_date", "primary_ticker",
            "more_tickers", "category", "vix", "sp500_mae_10d_pct",
            "nasdaq_mae_10d_pct", "title"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} articles on peaceful days -> {args.out} "
          f"(of {len(articles)} {args.categories} articles in range)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
