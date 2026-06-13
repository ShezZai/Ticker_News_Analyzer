"""Temp one-off: fill price_at_publish_time + price_at_end_of_the_marketday.

Reads the 900-articles CSV, and for each row uses the Massive aggregates API
(reusing ticker_news.research.market_data.fetch_bars) to fill:

  - price_at_publish_time          : close of the 1-min bar covering published_et
                                     (extended hours included; nearest same-day
                                     bar if the exact minute has no trade)
  - price_at_end_of_the_marketday  : official regular-session daily close for the
                                     publish date (rolled forward to the next
                                     session if that date is a holiday/weekend)

published_et is parsed as US market time (America/New_York). Bars are cached per
(ticker, day) so duplicate ticker/day rows cost no extra API calls.

Run:  .venv\\Scripts\\python.exe enrich_prices.py
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ticker_news.research.market_data import MARKET_TZ, fetch_bars

SRC = r"C:\Agents\final-project\900_articles_with_ticker_and_prices.csv"
OUT = SRC  # overwrite in place (a .bak copy is written first)

# --- caches: one network call per (ticker, day) for each kind --------------
_minute_cache: dict[tuple[str, str], list[dict]] = {}
_daily_cache: dict[tuple[str, str], list[dict]] = {}


def parse_published(value: str) -> datetime:
    """DD/MM/YYYY HH:MM (or HH:MM:SS) -> tz-aware America/New_York datetime."""
    txt = value.strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            naive = datetime.strptime(txt, fmt)
            return naive.replace(tzinfo=MARKET_TZ)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized published_et: {value!r}")


def minute_bars(ticker: str, day: str) -> list[dict]:
    """Minute bars for [day-4, day] so overnight/weekend publishes can fall
    back to the most recent prior session's last trade."""
    key = (ticker, day)
    if key not in _minute_cache:
        start = (datetime.fromisoformat(day) - timedelta(days=4)).strftime("%Y-%m-%d")
        _minute_cache[key] = fetch_bars(
            ticker, span="minute", multiplier=1, frm=start, to=day
        )
    return _minute_cache[key]


def daily_bars(ticker: str, day: str) -> list[dict]:
    """Daily bars for [day, day+7] so a holiday/weekend can roll forward."""
    key = (ticker, day)
    if key not in _daily_cache:
        end = (datetime.fromisoformat(day) + timedelta(days=7)).strftime("%Y-%m-%d")
        _daily_cache[key] = fetch_bars(ticker, span="day", multiplier=1, frm=day, to=end)
    return _daily_cache[key]


def bar_dt(bar: dict) -> datetime:
    """Bar epoch-ms 't' -> market-time datetime."""
    return datetime.fromtimestamp(bar["t"] / 1000, tz=ZoneInfo("UTC")).astimezone(MARKET_TZ)


def publish_price(ticker: str, ts: datetime) -> float | None:
    """Close of the last minute bar starting at-or-before ts.

    For a timestamp inside a 1-min bar this is the covering bar; for
    after-hours/overnight/weekend timestamps it is the most recent prior
    trade (last known price as of publish). Looks back up to 4 days.
    """
    bars = minute_bars(ticker, ts.strftime("%Y-%m-%d"))
    best = None
    for bar in bars:  # sorted asc; keep the latest with start <= ts
        if bar_dt(bar) <= ts:
            best = bar
        else:
            break
    return float(best["c"]) if best is not None else None


def eod_close(ticker: str, ts: datetime) -> float | None:
    """Daily close for ts's date; roll forward to the next session if missing."""
    day = ts.strftime("%Y-%m-%d")
    bars = daily_bars(ticker, day)
    if not bars:
        return None
    target = ts.date()
    for bar in bars:  # bars are sorted asc; first bar with date >= target
        if bar_dt(bar).date() >= target:
            return float(bar["c"])
    return None


def main() -> int:
    with open(SRC, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for col in ("price_at_publish_time", "price_at_end_of_the_marketday"):
        if col not in fieldnames:
            raise SystemExit(f"expected column missing: {col}")

    # backup once before overwriting in place
    with open(SRC, encoding="utf-8-sig") as src, open(SRC + ".bak", "w", encoding="utf-8") as bak:
        bak.write(src.read())

    filled = blanks = 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        ticker = (row.get("primary_ticker") or "").strip().upper()
        raw_ts = (row.get("published_et") or "").strip()
        try:
            ts = parse_published(raw_ts)
            pub = publish_price(ticker, ts)
            close = eod_close(ticker, ts)
        except Exception as exc:  # keep going; log and leave blanks
            print(f"  [{i}/{total}] {ticker} {raw_ts!r}: {exc}", file=sys.stderr)
            pub = close = None

        row["price_at_publish_time"] = "" if pub is None else f"{pub:.4f}"
        row["price_at_end_of_the_marketday"] = "" if close is None else f"{close:.4f}"
        if pub is None or close is None:
            blanks += 1
        else:
            filled += 1
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} processed (filled={filled}, with-blanks={blanks})")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"done: {total} rows, {filled} fully filled, {blanks} had a blank. -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
