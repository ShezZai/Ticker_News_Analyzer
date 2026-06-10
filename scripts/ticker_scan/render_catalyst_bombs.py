#!/usr/bin/env python3
"""Render entry-point candle charts for the big movers in a catalyst_returns CSV.

Reads a CSV produced by ``catalyst_returns.py`` and, for every row whose
``gain_pct`` magnitude exceeds the threshold (default 3% up or down), renders the
ticker's intraday candle chart for the entry day with the buy moment (``buy_et``)
marked, via ``ticker_candles.make_chart``.

Charts are written to ``pics_bombs/catalysts/<ticker>_<article_id>_<buy_date>.jpg``.

Usage:
    python render_catalyst_bombs.py
    python render_catalyst_bombs.py catalyst_returns_2025-02-01_2025-03-30.csv
    python render_catalyst_bombs.py INPUT.csv --threshold 5 --out-dir pics_bombs/catalysts

Requires MASSIVE_API_KEY (used by ticker_candles) in env / .env.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import redirect_stdout

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "checks_backtesting"))
import ticker_candles as tc  # noqa: E402

DEFAULT_INPUT = "catalyst_returns_2025-02-01_2025-03-30.csv"
DEFAULT_THRESHOLD = 3.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                   help=f"catalyst_returns CSV (default {DEFAULT_INPUT})")
    p.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                   help=f"min |gain_pct| to render, up or down (default {DEFAULT_THRESHOLD:g})")
    p.add_argument("--out-dir", default="pics_bombs/catalysts",
                   help="output folder (default pics_bombs/catalysts)")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    csv.field_size_limit(10 ** 7)

    jobs: list[tuple[str, str, str, float]] = []  # ticker, aid, buy_et, gain
    with open(args.input, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                gain = float(r["gain_pct"])
            except (KeyError, ValueError):
                continue
            if abs(gain) > args.threshold:
                jobs.append((r["ticker"].strip().upper(), r["article_id"], r["buy_et"], gain))

    if not jobs:
        print(f"No rows with |gain_pct| > {args.threshold:g} in {args.input}.")
        return 0

    print(f"Rendering {len(jobs)} mover(s) (|gain| > {args.threshold:g}%) to {args.out_dir}/ ...")
    done = failed = 0
    iterator = tqdm(jobs, unit="chart") if tqdm else jobs
    for ticker, aid, buy_et, _gain in iterator:
        buy_date = buy_et.split(" ")[0]
        out = os.path.join(args.out_dir, f"{ticker}_{aid}_{buy_date}.jpg")
        try:
            with open(os.devnull, "w") as null, redirect_stdout(null):
                tc.make_chart(ticker, buy_et, output=out)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {ticker} {aid} {buy_et}: {exc}", file=sys.stderr)

    print(f"Done. Wrote {done} chart(s); {failed} failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
