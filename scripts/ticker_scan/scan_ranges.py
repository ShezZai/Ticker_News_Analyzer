#!/usr/bin/env python3
"""Scan the universe for big intraday-range days and dump them to a CSV.

For every ticker this fetches hourly bars from the Massive API across a date
range -- INCLUDING pre-market and after-hours -- and, for each trading day,
computes the low-to-high range as a percentage:

    range_pct = (day_high - day_low) / day_low * 100

A day is flagged when ``range_pct`` meets the threshold (default 5%). Because
each hourly bar carries the true high/low of its hour, ``max(high)`` and
``min(low)`` over a day's bars are the exact daily extremes -- and since the
hourly series spans ~04:00-20:00 ET, those extremes include extended-hours
trading, as requested.

For every flagged day the NASDAQ index's own low-to-high range for that date is
added (``--index-ticker``, default ``I:COMP``, the NASDAQ Composite) so each row
carries a market-wide reference move.

Output CSV columns:
    date, ticker, low, high, range_pct, open, close, net_pct, direction,
    nasdaq_low, nasdaq_high, nasdaq_range_pct

``direction`` is "up" when the day's low printed before its high (the swing ran
up) and "down" otherwise; ``net_pct`` is the open->close change over the full
session. Rows are sorted by date then ticker.

Usage:
    python scan_ranges.py                       # all 118 tickers, >=5% range
    python scan_ranges.py --threshold 7         # >=7% range instead
    python scan_ranges.py --tickers NVDA,AMD    # just these
    python scan_ranges.py --start 2025-01-01 --end 2025-06-30
    python scan_ranges.py --bar minute          # exact (heavier) minute bars
    python scan_ranges.py -o big_days.csv

Requires MASSIVE_API_KEY in the environment / .env. The default ticker list is
read from public.ticker_data (NEWS_DB_DSN / DATABASE_URL, default dbname=news);
pass --tickers to skip the database entirely.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"

MARKET_TZ = ZoneInfo("America/New_York")
AGGS_URL = "https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/{span}/{frm}/{to}"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
BAR_LIMIT = 50_000  # Massive's per-request cap

DEFAULT_THRESHOLD = 5.0
DEFAULT_INDEX = "I:COMP"  # NASDAQ Composite
DEFAULT_NAS_THRESHOLD = 1.2
# The news dataset starts here; default the scan to the same window.
DEFAULT_START = "2024-11-01"


class ScanError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def load_universe() -> Tuple[List[str], Dict[str, str]]:
    """Return (tickers, segment_map) from public.ticker_data."""
    import psycopg

    try:
        with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT ticker, segment FROM public.ticker_data ORDER BY ticker")
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
        raise ScanError(
            f"Could not read the default ticker universe from the database "
            f"({exc}).\nPass --tickers AAPL,NVDA,... or set NEWS_DB_DSN."
        ) from exc
    tickers = [t.strip().upper() for t, _ in rows if t and t.strip()]
    if not tickers:
        raise ScanError("public.ticker_data is empty; populate it or pass --tickers.")
    segment_map = {t.strip().upper(): (seg or "") for t, seg in rows if t and t.strip()}
    return tickers, segment_map


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _api_key(explicit: Optional[str]) -> str:
    key = explicit or os.getenv("MASSIVE_API_KEY")
    if not key:
        raise ScanError("No API key. Pass --api-key or set MASSIVE_API_KEY in .env")
    return key


def _get(url: str, params: dict) -> dict:
    """GET with retry/backoff on rate limits and transient server errors."""
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, ...
            continue
        resp.raise_for_status()
        raise ScanError(f"unexpected status {resp.status_code} for {url}")
    raise ScanError(f"exhausted retries for {url}")


def fetch_bars(
    ticker: str, span: str, frm: str, to: str, api_key: str
) -> List[Tuple[datetime, float, float, float, float]]:
    """Return [(et_dt, open, high, low, close), ...] for `ticker` over [frm, to].

    `span` is "hour" or "minute"; both include extended-hours trading. Follows
    Massive's `next_url` pagination so ranges past the 50k-bar cap still work.
    """
    url = AGGS_URL.format(ticker=ticker, span=span, frm=frm, to=to)
    params = {"adjusted": "true", "sort": "asc", "limit": BAR_LIMIT, "apiKey": api_key}
    out: List[Tuple[datetime, float, float, float, float]] = []
    while True:
        payload = _get(url, params)
        for b in payload.get("results") or []:
            dt = datetime.fromtimestamp(b["t"] / 1000, MARKET_TZ)
            out.append((dt, b["o"], b["h"], b["l"], b["c"]))
        nxt = payload.get("next_url")
        if not nxt:
            break
        # next_url carries the cursor but not the key; re-attach it.
        url, params = nxt, {"apiKey": api_key}
    return out


# --------------------------------------------------------------------------- #
# Per-day reduction
# --------------------------------------------------------------------------- #
class DayRange:
    __slots__ = ("low", "high", "open", "close", "low_dt", "high_dt", "first_dt", "last_dt")

    def __init__(self, dt: datetime, o: float, h: float, l: float, c: float):
        self.low = l
        self.high = h
        self.open = o
        self.close = c
        self.low_dt = dt
        self.high_dt = dt
        self.first_dt = dt
        self.last_dt = dt

    def update(self, dt: datetime, o: float, h: float, l: float, c: float) -> None:
        if l < self.low:
            self.low, self.low_dt = l, dt
        if h > self.high:
            self.high, self.high_dt = h, dt
        if dt < self.first_dt:
            self.first_dt, self.open = dt, o
        if dt > self.last_dt:
            self.last_dt, self.close = dt, c

    def range_pct(self) -> Optional[float]:
        if self.low <= 0:
            return None
        return (self.high - self.low) / self.low * 100.0

    def net_pct(self) -> Optional[float]:
        if self.open <= 0:
            return None
        return (self.close - self.open) / self.open * 100.0

    def direction(self) -> str:
        # low printed before high -> the swing ran up; else down.
        return "up" if self.low_dt <= self.high_dt else "down"


def daily_ranges(
    bars: Sequence[Tuple[datetime, float, float, float, float]]
) -> Dict[date, DayRange]:
    """Collapse intraday bars into one DayRange per ET calendar date."""
    days: Dict[date, DayRange] = {}
    for dt, o, h, l, c in bars:
        d = dt.date()
        if d in days:
            days[d].update(dt, o, h, l, c)
        else:
            days[d] = DayRange(dt, o, h, l, c)
    return days


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
def scan_ticker(
    ticker: str, span: str, frm: str, to: str, threshold: float, api_key: str
) -> List[dict]:
    """Return one dict per flagged day for `ticker` (range_pct >= threshold)."""
    bars = fetch_bars(ticker, span, frm, to, api_key)
    flagged: List[dict] = []
    for d, dr in daily_ranges(bars).items():
        rng = dr.range_pct()
        if rng is None or rng < threshold:
            continue
        net = dr.net_pct()
        flagged.append(
            {
                "date": d.isoformat(),
                "ticker": ticker,
                "low": round(dr.low, 4),
                "high": round(dr.high, 4),
                "range_pct": round(rng, 2),
                "open": round(dr.open, 4),
                "close": round(dr.close, 4),
                "net_pct": round(net, 2) if net is not None else "",
                "direction": dr.direction(),
            }
        )
    return flagged


def index_ranges(
    index_ticker: str, dates: Sequence[date], api_key: str
) -> Dict[date, Tuple[float, float, float]]:
    """Daily (low, high, range_pct) for the index across the span of `dates`.

    One daily-bar request covers the whole window; only the dates actually
    flagged are returned.
    """
    if not dates:
        return {}
    frm, to = min(dates).isoformat(), max(dates).isoformat()
    url = AGGS_URL.format(ticker=index_ticker, span="day", frm=frm, to=to)
    params = {"adjusted": "true", "sort": "asc", "limit": BAR_LIMIT, "apiKey": api_key}
    wanted = set(dates)
    out: Dict[date, Tuple[float, float, float]] = {}
    try:
        payload = _get(url, params)
    except (ScanError, requests.HTTPError) as exc:
        print(f"  warning: could not fetch index {index_ticker}: {exc}", file=sys.stderr)
        return {}
    for b in payload.get("results") or []:
        d = datetime.fromtimestamp(b["t"] / 1000, MARKET_TZ).date()
        if d not in wanted:
            continue
        low, high = b["l"], b["h"]
        rng = (high - low) / low * 100.0 if low > 0 else None
        out[d] = (round(low, 4), round(high, 4), round(rng, 2) if rng is not None else "")
    return out


def run(
    tickers: Sequence[str],
    start: str,
    end: str,
    threshold: float,
    span: str,
    index_ticker: str,
    output: str,
    workers: int,
    api_key: str,
    nas_threshold: float = DEFAULT_NAS_THRESHOLD,
    segment_map: Optional[Dict[str, str]] = None,
) -> int:
    """Scan all tickers, attach the index reference, and write the CSV."""
    print(
        f"Scanning {len(tickers)} ticker(s) {start}..{end} for >= {threshold:g}% "
        f"low-to-high days ({span} bars, extended hours included) ..."
    )
    rows: List[dict] = []
    n_failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scan_ticker, tk, span, start, end, threshold, api_key): tk
            for tk in tickers
        }
        completed = as_completed(futures)
        pbar = tqdm(completed, total=len(futures), unit="ticker") if tqdm else completed
        for fut in pbar:
            tk = futures[fut]
            try:
                rows.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                print(f"  {tk}: {exc}", file=sys.stderr)
            if tqdm:
                pbar.set_postfix_str(f"{len(rows)} flagged | {n_failed} failed")

    # Attach the NASDAQ index reference for every flagged date.
    flagged_dates = sorted({date.fromisoformat(r["date"]) for r in rows})
    print(f"Fetching {index_ticker} reference for {len(flagged_dates)} date(s) ...")
    idx = index_ranges(index_ticker, flagged_dates, api_key)
    seg = segment_map or {}
    for r in rows:
        d = date.fromisoformat(r["date"])
        lo, hi, rng = idx.get(d, ("", "", ""))
        r["nasdaq_low"], r["nasdaq_high"], r["nasdaq_range_pct"] = lo, hi, rng
        r["segment"] = seg.get(r["ticker"], "")

    # Filter out days where the NASDAQ moved less than nas_threshold.
    before = len(rows)
    rows = [
        r for r in rows
        if isinstance(r["nasdaq_range_pct"], (int, float))
        and r["nasdaq_range_pct"] < nas_threshold
    ]
    if len(rows) < before:
        print(
            f"  nas_t filter (< {nas_threshold:g}%): dropped {before - len(rows)} row(s), "
            f"{len(rows)} remaining"
        )

    rows.sort(key=lambda r: (r["date"], r["ticker"]))

    fields = [
        "date", "ticker", "segment", "low", "high", "range_pct", "open", "close",
        "net_pct", "direction", "nasdaq_low", "nasdaq_high", "nasdaq_range_pct",
    ]
    with open(output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} flagged day(s) across "
        f"{len({r['ticker'] for r in rows})} ticker(s) to {output}"
        + (f"  ({n_failed} ticker(s) failed)" if n_failed else "")
    )
    return len(rows)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                   help=f"min low-to-high range in %% to flag a day (default {DEFAULT_THRESHOLD:g})")
    p.add_argument("--tickers", type=str, default=None,
                   help="comma-separated tickers (default: all 118 from ticker_data)")
    p.add_argument("--start", default=DEFAULT_START, help=f"start date YYYY-MM-DD (default {DEFAULT_START})")
    p.add_argument("--end", default=date.today().isoformat(),
                   help="end date YYYY-MM-DD (default today)")
    p.add_argument("--bar", choices=("hour", "minute"), default="hour",
                   help="intraday bar size; hour is exact for daily extremes and "
                        "much cheaper, minute is heavier (default hour)")
    p.add_argument("--index-ticker", default=DEFAULT_INDEX,
                   help=f"reference index symbol (default {DEFAULT_INDEX}, NASDAQ Composite)")
    p.add_argument("-o", "--output", default=None,
                   help="output CSV path (default ticker_range_scan_<start>_<end>.csv)")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent ticker fetches (default 8)")
    p.add_argument("--api-key", help="override MASSIVE_API_KEY")
    p.add_argument("--nas_t", type=float, default=DEFAULT_NAS_THRESHOLD,
                   help=f"max NASDAQ low-to-high %% on a flagged day; only keeps days "
                        f"where NASDAQ moved LESS than this (calm market days) "
                        f"(default {DEFAULT_NAS_THRESHOLD:g}; set 999 to disable)")
    args = p.parse_args(argv)

    try:
        key = _api_key(args.api_key)
        segment_map: Dict[str, str] = {}
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            # best-effort segment lookup even for explicit ticker lists
            try:
                _, segment_map = load_universe()
            except ScanError:
                pass
        else:
            tickers, segment_map = load_universe()
        output = args.output or f"ticker_range_scan_{args.start}_{args.end}.csv"
        run(
            tickers, args.start, args.end, args.threshold, args.bar,
            args.index_ticker, output, args.workers, key,
            nas_threshold=args.nas_t, segment_map=segment_map,
        )
    except ScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except requests.HTTPError as exc:
        print(f"error: API request failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
