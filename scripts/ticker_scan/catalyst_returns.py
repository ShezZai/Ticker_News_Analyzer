#!/usr/bin/env python3
"""Compute the buy-the-news return for every catalyst-category article in a range.

For each article whose ``category`` is a catalyst candidate (by default
"real news", "legal solicitation", "regulatory filing") and that has a
``primary_ticker``, this simulates:

    BUY  at the article's publication moment -- at the extended-hours price, so
         pre-market and after-hours entries are honored (entry = the open of the
         minute bar at/after the headline crossed).
    SELL at the next regular-session closing price:
           * entry before 16:00 ET (pre-market or regular)  -> SAME day's close
           * entry at/after 16:00 ET (after-hours)          -> NEXT trading day's close

After-hours entries are EXCLUDED by default (only pre-market/regular entries with
a same-day close are kept); pass --include-after-hours to keep them.

and writes the gain/loss percentage to a CSV alongside the article id, ticker,
category, and the entry/exit prices.

Prices come from the Massive API: 1-minute bars (extended hours) for the entry
price, and daily bars for the official regular-session close and the trading-day
calendar.

Usage:
    python catalyst_returns.py
    python catalyst_returns.py --start 2025-02-01 --end 2025-03-30
    python catalyst_returns.py --categories "real news,legal solicitation"
    python catalyst_returns.py --all-tickers   # a row per ticker each article names
    python catalyst_returns.py -o catalyst_returns.csv

Requires MASSIVE_API_KEY in env / .env, and NEWS_DB_DSN / DATABASE_URL for the
article metadata (default dbname=news).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as dtime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import psycopg
import requests
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"

MARKET_TZ = ZoneInfo("America/New_York")
AGGS_URL = "https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/{span}/{frm}/{to}"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
BAR_LIMIT = 50_000

REGULAR_CLOSE = dtime(16, 0)
PREMARKET_OPEN = dtime(4, 0)
REGULAR_OPEN = dtime(9, 30)

DEFAULT_START = "2025-02-01"
DEFAULT_END = "2025-03-30"
# "news, class action, etc" -> the categories that can actually move a stock.
DEFAULT_CATEGORIES = ["real news", "legal solicitation", "regulatory filing"]
# Buy can roll into the next session and an after-hours buy sells the next day,
# so fetch prices a little past the article window.
PRICE_TAIL_DAYS = 7


class CatalystError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Articles
# --------------------------------------------------------------------------- #
def load_articles(
    start: str, end: str, categories: Sequence[str]
) -> List[Tuple[int, str, List[str], str, datetime, str]]:
    """Return (id, primary, more_tickers, category, published_et, title) per article."""
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, primary_ticker, more_tickers, category, published_utc, title
            FROM public.articles
            WHERE category = ANY(%s)
              AND primary_ticker IS NOT NULL
              AND (published_utc AT TIME ZONE 'America/New_York')::date
                      BETWEEN %s AND %s
            ORDER BY published_utc
            """,
            (list(categories), start, end),
        )
        rows = cur.fetchall()
    out = []
    for aid, primary, more, category, pub_utc, title in rows:
        more_up = [t.strip().upper() for t in (more or []) if t and t.strip()]
        out.append((aid, primary.strip().upper(), more_up, category,
                    pub_utc.astimezone(MARKET_TZ), title))
    return out


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def _api_key(explicit: Optional[str]) -> str:
    key = explicit or os.getenv("MASSIVE_API_KEY")
    if not key:
        raise CatalystError("No API key. Pass --api-key or set MASSIVE_API_KEY in .env")
    return key


def _get(url: str, params: dict) -> dict:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        raise CatalystError(f"unexpected status {resp.status_code} for {url}")
    raise CatalystError(f"exhausted retries for {url}")


def _fetch(ticker: str, span: str, frm: str, to: str, api_key: str) -> List[dict]:
    url = AGGS_URL.format(ticker=ticker, span=span, frm=frm, to=to)
    params = {"adjusted": "true", "sort": "asc", "limit": BAR_LIMIT, "apiKey": api_key}
    results: List[dict] = []
    while True:
        payload = _get(url, params)
        results.extend(payload.get("results") or [])
        nxt = payload.get("next_url")
        if not nxt:
            break
        url, params = nxt, {"apiKey": api_key}
    return results


class TickerPrices:
    """Minute-bar entry prices + daily-close exits for one ticker."""

    def __init__(self, minute_bars: List[dict], daily_bars: List[dict]):
        # minute series: parallel arrays sorted by time, for bisect lookups
        self.times: List[datetime] = []
        self.opens: List[float] = []
        for b in minute_bars:
            self.times.append(datetime.fromtimestamp(b["t"] / 1000, MARKET_TZ))
            self.opens.append(b["o"])
        # daily closes keyed by ET trading date, plus the sorted trading-day list
        self.close: Dict[date, float] = {}
        for b in daily_bars:
            d = datetime.fromtimestamp(b["t"] / 1000, MARKET_TZ).date()
            self.close[d] = b["c"]
        self.trading_days: List[date] = sorted(self.close)

    def entry(self, ts: datetime) -> Optional[Tuple[datetime, float]]:
        """First tradeable (time, open) at/after the headline minute, or None."""
        floored = ts.replace(second=0, microsecond=0)
        i = bisect.bisect_left(self.times, floored)
        if i >= len(self.times):
            return None
        return self.times[i], self.opens[i]

    def next_trading_day(self, d: date) -> Optional[date]:
        i = bisect.bisect_right(self.trading_days, d)
        return self.trading_days[i] if i < len(self.trading_days) else None


def fetch_prices(ticker: str, frm: str, to: str, api_key: str) -> TickerPrices:
    minute = _fetch(ticker, "minute", frm, to, api_key)
    daily = _fetch(ticker, "day", frm, to, api_key)
    return TickerPrices(minute, daily)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def session_of(t: dtime) -> str:
    if PREMARKET_OPEN <= t < REGULAR_OPEN:
        return "pre-market"
    if REGULAR_OPEN <= t < REGULAR_CLOSE:
        return "regular"
    return "after-hours"


def simulate(article, prices: TickerPrices, include_after_hours: bool = False) -> Optional[dict]:
    """Buy at the headline, sell at the next regular close; return a CSV row."""
    aid, ticker, category, pub_et, title = article
    entry = prices.entry(pub_et)
    if entry is None:
        return None
    buy_dt, buy_price = entry
    if buy_price <= 0:
        return None

    # entry before the close -> same-day close; after-hours -> next trading day.
    if buy_dt.time() < REGULAR_CLOSE:
        sell_date = buy_dt.date()
        hold = "same-day-close"
    else:
        # after-hours entry: excluded by default (only same-day exits are kept).
        if not include_after_hours:
            return None
        sell_date = prices.next_trading_day(buy_dt.date())
        hold = "next-day-close"
    if sell_date is None or sell_date not in prices.close:
        return None
    sell_close = prices.close[sell_date]

    gain = (sell_close - buy_price) / buy_price * 100.0
    return {
        "article_id": aid,
        "ticker": ticker,
        "category": category,
        "published_et": pub_et.strftime("%Y-%m-%d %H:%M"),
        "entry_session": session_of(buy_dt.time()),
        "buy_et": buy_dt.strftime("%Y-%m-%d %H:%M"),
        "buy_price": round(buy_price, 4),
        "hold_until": hold,
        "sell_date": sell_date.isoformat(),
        "sell_close": round(sell_close, 4),
        "gain_pct": round(gain, 2),
        "title": title,
    }


def run(
    start: str, end: str, categories: Sequence[str], output: str,
    workers: int, api_key: str, include_after_hours: bool = False,
    all_tickers: bool = False,
) -> int:
    articles = load_articles(start, end, categories)
    if not articles:
        print("No catalyst articles found for the range/categories.")
        return 0

    # one spec per ticker to simulate: just the primary, or (with --all-tickers)
    # every ticker the article names -- primary + more_tickers -- all entered at
    # the SAME article moment. Each carries a ticker_role for the output.
    specs: List[Tuple] = []  # (aid, ticker, category, pub_et, title, role)
    for aid, primary, more, category, pub_et, title in articles:
        names = [primary] + (more if all_tickers else [])
        for tk in names:
            role = "primary" if tk == primary else "mentioned"
            specs.append((aid, tk, category, pub_et, title, role))

    tickers = sorted({s[1] for s in specs})
    print(
        f"{len(articles)} catalyst article(s) -> {len(specs)} (article,ticker) pair(s) "
        f"across {len(tickers)} ticker(s) [{', '.join(categories)}]"
        + ("  [all-tickers]" if all_tickers else "")
    )

    price_to = (date.fromisoformat(end) + timedelta(days=PRICE_TAIL_DAYS)).isoformat()
    prices: Dict[str, TickerPrices] = {}
    n_price_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_prices, tk, start, price_to, api_key): tk for tk in tickers}
        it = as_completed(futs)
        pbar = tqdm(it, total=len(futs), unit="ticker", desc="prices") if tqdm else it
        for fut in pbar:
            tk = futs[fut]
            try:
                prices[tk] = fut.result()
            except Exception as exc:  # noqa: BLE001
                n_price_fail += 1
                print(f"  prices {tk}: {exc}", file=sys.stderr)

    rows: List[dict] = []
    n_skip = 0
    for aid, tk, category, pub_et, title, role in specs:
        tp = prices.get(tk)
        row = simulate((aid, tk, category, pub_et, title), tp, include_after_hours) if tp else None
        if row is None:
            n_skip += 1
            continue
        row["ticker_role"] = role
        rows.append(row)

    rows.sort(key=lambda r: (r["published_et"], r["article_id"], r["ticker"]))
    fields = [
        "article_id", "ticker", "ticker_role", "category", "published_et",
        "entry_session", "buy_et", "buy_price", "hold_until", "sell_date",
        "sell_close", "gain_pct", "title",
    ]
    with open(output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} row(s) to {output} "
        f"({n_skip} (article,ticker) pair(s) skipped for missing prices, "
        f"{n_price_fail} ticker fetch failure(s))."
    )
    if rows:
        wins = sum(1 for r in rows if r["gain_pct"] > 0)
        avg = sum(r["gain_pct"] for r in rows) / len(rows)
        print(f"  avg gain {avg:+.2f}% | {wins}/{len(rows)} positive")
        by_cat: Dict[str, List[float]] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r["gain_pct"])
        for cat, gs in sorted(by_cat.items()):
            w = sum(1 for g in gs if g > 0)
            print(f"  {cat:<20} n={len(gs):<4} avg={sum(gs)/len(gs):+.2f}%  {w}/{len(gs)} up")
    return len(rows)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", default=DEFAULT_START, help=f"start date (default {DEFAULT_START})")
    p.add_argument("--end", default=DEFAULT_END, help=f"end date (default {DEFAULT_END})")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                   help="comma-separated catalyst categories "
                        f"(default: {', '.join(DEFAULT_CATEGORIES)})")
    p.add_argument("-o", "--output", default=None,
                   help="output CSV (default catalyst_returns_<start>_<end>.csv)")
    p.add_argument("--workers", type=int, default=8, help="concurrent price fetches (default 8)")
    p.add_argument("--include-after-hours", action="store_true",
                   help="also keep after-hours entries (sold at next day's close); "
                        "excluded by default")
    p.add_argument("--all-tickers", action="store_true",
                   help="emit a row for EVERY ticker each article names (primary + "
                        "more_tickers), each entered at the same article moment; "
                        "adds a ticker_role column (primary/mentioned)")
    p.add_argument("--api-key", help="override MASSIVE_API_KEY")
    args = p.parse_args(argv)

    try:
        key = _api_key(args.api_key)
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        output = args.output or f"catalyst_returns_{args.start}_{args.end}.csv"
        run(args.start, args.end, cats, output, args.workers, key,
            include_after_hours=args.include_after_hours, all_tickers=args.all_tickers)
    except CatalystError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except requests.HTTPError as exc:
        print(f"error: API request failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
