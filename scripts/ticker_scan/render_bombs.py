#!/usr/bin/env python3
"""Render an intraday candle chart for every ticker/article hit in an articled CSV.

Reads a ``*_articled.csv`` produced by ``attach_articles.py`` and, for each
article attached to a (date, ticker) row, renders that ticker's intraday candle
chart for the trading day with the article's moment marked, via
``ticker_candles.make_chart``.

Marking rule:
  * same-day article      -> mark the article's own published_et timestamp.
  * prev-after-hours article -> the news broke after yesterday's close, so mark
    the pre-market start (04:00 ET) of the trading day instead.

Charts are written to ``pics_bombs/<ticker>_<article_id>_<date>.jpg`` where
``date`` is the trading day (the row's date).

Usage:
    python render_bombs.py INPUT_articled.csv
    python render_bombs.py INPUT_articled.csv --out-dir pics_bombs

Requires MASSIVE_API_KEY (used by ticker_candles) in env / .env.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import redirect_stdout

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# ticker_candles lives in scripts/checks_backtesting
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "checks_backtesting"))
import ticker_candles as tc  # noqa: E402

PREMARKET_START = "04:00"  # ET pre-market open


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", help="*_articled.csv from attach_articles.py")
    p.add_argument("--out-dir", default="pics_bombs", help="output folder (default pics_bombs)")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    csv.field_size_limit(10 ** 7)

    # Build the full job list first so we get an accurate progress bar.
    jobs: list[tuple[str, int, str, str, str]] = []  # ticker, aid, row_date, when, timestamp
    with open(args.input, newline="") as fh:
        for r in csv.DictReader(fh):
            arts = json.loads(r.get("articles") or "[]")
            ticker, row_date = r["ticker"].strip().upper(), r["date"]
            for a in arts:
                if a.get("when") == "prev-after-hours":
                    ts = f"{row_date} {PREMARKET_START}"
                else:
                    ts = a["published_et"]
                jobs.append((ticker, a["id"], row_date, a.get("when", ""), ts))

    if not jobs:
        print("No article attachments found in the CSV.")
        return 0

    print(f"Rendering {len(jobs)} chart(s) to {args.out_dir}/ ...")
    done = 0
    failed = 0
    iterator = tqdm(jobs, unit="chart") if tqdm else jobs
    for ticker, aid, row_date, _when, ts in iterator:
        out = os.path.join(args.out_dir, f"{ticker}_{aid}_{row_date}.jpg")
        try:
            with open(os.devnull, "w") as null, redirect_stdout(null):
                tc.make_chart(ticker, ts, output=out)
            done += 1
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failed += 1
            print(f"  {ticker} {aid} {row_date} ({ts}): {exc}", file=sys.stderr)

    print(f"Done. Wrote {done} chart(s); {failed} failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
