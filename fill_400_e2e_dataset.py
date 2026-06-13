"""Fill the 400-article E2E eval dataset CSV (langfuse_datasets/400-e2e.csv).

One reproducible pass over the tab-separated CSV that:

  1. From the shared `news` DB (DATABASE_URL), fills per article id:
       - main ticker          <- articles.primary_ticker
       - publishe_datetime     <- articles.published_utc (ISO-8601 UTC)

  2. From the Massive aggregates API (reusing
     ticker_news.research.market_data.fetch_bars) fills:
       - price_at_published        : last traded price as of the publish
                                     timestamp -- the close of the most recent
                                     1-min bar starting at/before published_et
                                     (extended hours included; looks back up to
                                     4 days for overnight/weekend publishes).
       - price_at_market_close     : the regular-session close that comes AFTER
                                     publication -- that day's official close if
                                     published before 16:00 ET on a trading day,
                                     otherwise the NEXT trading session's close
                                     (so after-hours / weekend news is not scored
                                     against a close that already happened).

  3. Derives the expected_output column (acceptable verdicts) from the move
     gain_pct = (close - published) / published * 100, with a +-0.3% deadband:
       - gain >= +0.3%   -> ["buy"]            (clearly up)
       - gain <= -0.3%   -> ["sell"]           (clearly down)
       - 0 < gain < 0.3% -> ["buy", "hold"]    (small up: buy or hold ok, sell not)
       - -0.3% < gain < 0-> ["sell", "hold"]   (small down: sell or hold ok, buy not)
       - gain == 0       -> ["hold"]
     i.e. inside the deadband both the gain-direction call AND hold are correct;
     only the opposite-direction call is wrong. Verdicts are lowercase to match
     the sentiment graph's buy/sell/hold Verdict.

Bars are cached per (ticker, publish-day) so duplicate ticker/day rows cost no
extra API calls. A .bak copy of the CSV is written before overwriting.

Run:  .venv\\Scripts\\python.exe fill_400_e2e_dataset.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import psycopg

from ticker_news.research.market_data import MARKET_TZ, REGULAR_CLOSE, fetch_bars
from ticker_news.shared.config import get_settings

CSV_PATH = r"C:\Agents\final-project\langfuse_datasets\400-e2e.csv"
DEADBAND_PCT = 0.3  # |move| below this -> hold is also acceptable

# Existing tab-separated columns (header keeps the original `publishe_datetime`
# spelling) plus the expected_output column we add.
TICKER_COL = "main ticker"
DATETIME_COL = "publishe_datetime"
PUB_PRICE_COL = "price_at_published"
CLOSE_PRICE_COL = "price_at_market_close"
EXPECTED_COL = "expected_output"

UTC = ZoneInfo("UTC")

# caches: one network call per (ticker, publish-day) per bar kind
_minute_cache: dict[tuple[str, str], list[dict]] = {}
_daily_cache: dict[tuple[str, str], list[dict]] = {}


def bar_dt(bar: dict) -> datetime:
    """Bar epoch-ms 't' -> America/New_York datetime."""
    return datetime.fromtimestamp(bar["t"] / 1000, tz=UTC).astimezone(MARKET_TZ)


def minute_bars(ticker: str, day: str) -> list[dict]:
    """1-min bars for [day-4, day] so overnight/weekend publishes can fall back
    to the most recent prior session's last trade."""
    key = (ticker, day)
    if key not in _minute_cache:
        start = (datetime.fromisoformat(day) - timedelta(days=4)).strftime("%Y-%m-%d")
        _minute_cache[key] = fetch_bars(
            ticker, span="minute", multiplier=1, frm=start, to=day
        )
    return _minute_cache[key]


def daily_bars(ticker: str, day: str) -> list[dict]:
    """Daily bars for [day-4, day+10] so we can pick the publish day's close or
    roll forward to the next session for after-hours / holiday publishes."""
    key = (ticker, day)
    if key not in _daily_cache:
        start = (datetime.fromisoformat(day) - timedelta(days=4)).strftime("%Y-%m-%d")
        end = (datetime.fromisoformat(day) + timedelta(days=10)).strftime("%Y-%m-%d")
        _daily_cache[key] = fetch_bars(ticker, span="day", multiplier=1, frm=start, to=end)
    return _daily_cache[key]


def price_at_published(ticker: str, ts: datetime) -> float | None:
    """Close of the most recent 1-min bar starting at/before `ts`.

    For a timestamp inside a trading minute this is the covering bar; for
    after-hours / overnight / weekend timestamps it is the most recent prior
    trade (the last known price as of publication). Looks back up to 4 days.
    """
    bars = minute_bars(ticker, ts.strftime("%Y-%m-%d"))
    best = None
    for bar in bars:  # sorted asc; keep the latest with start <= ts
        if bar_dt(bar) <= ts:
            best = bar
        else:
            break
    return float(best["c"]) if best is not None else None


def price_at_market_close(ticker: str, ts: datetime) -> float | None:
    """Regular-session close that comes AFTER publication.

    Published before 16:00 ET on a trading day -> that day's close. Published
    at/after 16:00 ET, or on a non-trading day (weekend/holiday) -> the next
    trading session's close. Mirrors ticker_scan.simulate's exit gating.
    """
    day = ts.strftime("%Y-%m-%d")
    closes = {bar_dt(b).date(): float(b["c"]) for b in daily_bars(ticker, day)}
    if not closes:
        return None
    pub_date = ts.date()
    if pub_date in closes and ts.timetz().replace(tzinfo=None) < REGULAR_CLOSE:
        return closes[pub_date]
    # after-hours on a trading day, or a non-trading publish date: next session
    later = sorted(d for d in closes if d > pub_date)
    return closes[later[0]] if later else None


def expected_output(pub: float | None, close: float | None) -> tuple[str, str]:
    """(expected_output JSON, gain_pct string) under the +-0.3% deadband rule."""
    if pub is None or close is None or pub <= 0:
        return "", ""
    gain = (close - pub) / pub * 100.0
    if gain >= DEADBAND_PCT:
        acceptable = ["buy"]
    elif gain <= -DEADBAND_PCT:
        acceptable = ["sell"]
    elif gain > 0:
        acceptable = ["buy", "hold"]
    elif gain < 0:
        acceptable = ["sell", "hold"]
    else:
        acceptable = ["hold"]
    return json.dumps(acceptable), f"{gain:+.3f}"


def load_db_fields(ids: list[int]) -> dict[int, tuple[str | None, datetime | None]]:
    """article id -> (primary_ticker, published_utc) for every requested id."""
    with psycopg.connect(get_settings().database_url) as conn:
        rows = conn.execute(
            "SELECT id, primary_ticker, published_utc "
            "FROM public.articles WHERE id = ANY(%s)",
            (ids,),
        ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def main() -> int:
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for col in (TICKER_COL, DATETIME_COL, PUB_PRICE_COL, CLOSE_PRICE_COL):
        if col not in fieldnames:
            raise SystemExit(f"expected column missing from CSV header: {col!r}")
    if EXPECTED_COL not in fieldnames:
        fieldnames.append(EXPECTED_COL)

    ids = [int(r["article_id"]) for r in rows]
    db = load_db_fields(ids)
    missing = sorted(set(ids) - set(db))
    if missing:
        raise SystemExit(f"article ids not found in DB: {missing}")

    # backup once before overwriting in place
    with open(CSV_PATH, encoding="utf-8-sig") as src, \
            open(CSV_PATH + ".bak", "w", encoding="utf-8", newline="") as bak:
        bak.write(src.read())

    total = len(rows)
    filled = blanks = 0
    for i, row in enumerate(rows, 1):
        aid = int(row["article_id"])
        ticker, published_utc = db[aid]
        ticker = (ticker or "").strip().upper()

        row[TICKER_COL] = ticker
        row[DATETIME_COL] = (
            published_utc.astimezone(timezone.utc).isoformat() if published_utc else ""
        )

        pub = close = None
        if ticker and published_utc is not None:
            ts_et = published_utc.astimezone(MARKET_TZ)
            try:
                pub = price_at_published(ticker, ts_et)
                close = price_at_market_close(ticker, ts_et)
            except Exception as exc:  # keep going; log and leave blanks
                print(f"  [{i}/{total}] {aid} {ticker}: {exc}", file=sys.stderr)

        row[PUB_PRICE_COL] = "" if pub is None else f"{pub:.4f}"
        row[CLOSE_PRICE_COL] = "" if close is None else f"{close:.4f}"
        expected, gain = expected_output(pub, close)
        row[EXPECTED_COL] = expected

        if pub is None or close is None:
            blanks += 1
        else:
            filled += 1
        if i % 25 == 0 or i == total:
            print(f"  {i}/{total} (filled={filled}, blanks={blanks})  "
                  f"last: {aid} {ticker} gain={gain or 'n/a'} -> {expected or '(blank)'}")

    # Write to a temp sibling, then atomically replace the target. If the
    # target is locked (e.g. open in Excel), keep the output as a .filled.csv
    # sibling so the (slow, paid) Massive recompute is never lost.
    tmp_path = CSV_PATH + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    out_path = CSV_PATH
    try:
        os.replace(tmp_path, CSV_PATH)
    except PermissionError:
        out_path = CSV_PATH.removesuffix(".csv") + ".filled.csv"
        os.replace(tmp_path, out_path)
        print(f"WARNING: {CSV_PATH} is locked (close Excel?); wrote {out_path} instead.")

    print(f"done: {total} rows, {filled} fully priced, {blanks} with a blank. -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
