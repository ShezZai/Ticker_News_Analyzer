#!/usr/bin/env python3
"""For each article behind a pics folder, render a chart for EVERY ticker it names.

Given a folder of ``<ticker>_<article_id>_<date>.jpg`` charts (e.g. a catalyst
bucket) and the catalyst_returns CSV they came from, this looks up each article's
full ticker set -- ``primary_ticker`` plus ``more_tickers`` -- and renders that
ticker's intraday candle chart at the SAME entry moment (``buy_et``), so you can
see how every named stock reacted, not just the primary one.

Charts are written next to the originals as ``<ticker>_<article_id>_<date>.jpg``;
files that already exist (the primary-ticker charts) are skipped.

Usage:
    python render_all_tickers.py
    python render_all_tickers.py --pics-dir pics_bombs/catalysts/other_related
    python render_all_tickers.py --csv catalyst_returns_2025-02-01_2025-03-30.csv

Requires MASSIVE_API_KEY (ticker_candles) and NEWS_DB_DSN / DATABASE_URL.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from contextlib import redirect_stdout

import psycopg

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "checks_backtesting"))
import ticker_candles as tc  # noqa: E402  (also calls load_dotenv at import)

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"
DEFAULT_PICS = "pics_bombs/catalysts/other_related"
DEFAULT_CSV = "catalyst_returns_2025-02-01_2025-03-30.csv"
_FNAME = re.compile(r"^[A-Z0-9.]+_(\d+)_\d{4}-\d{2}-\d{2}\.jpg$")


def article_tickers(aids: list[int]) -> dict[int, list[str]]:
    """Map article id -> [primary_ticker, *more_tickers] (deduped, order-preserving)."""
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, primary_ticker, more_tickers FROM public.articles "
            "WHERE id = ANY(%s)",
            (aids,),
        )
        rows = cur.fetchall()
    out: dict[int, list[str]] = {}
    for aid, primary, more in rows:
        seen, tickers = set(), []
        for t in [primary, *(more or [])]:
            if t and t.upper() not in seen:
                seen.add(t.upper())
                tickers.append(t.upper())
        out[aid] = tickers
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pics-dir", default=DEFAULT_PICS, help=f"folder of charts (default {DEFAULT_PICS})")
    p.add_argument("--csv", default=DEFAULT_CSV, help=f"catalyst_returns CSV (default {DEFAULT_CSV})")
    p.add_argument("--out-dir", default=None, help="output folder (default: --pics-dir)")
    args = p.parse_args(argv)
    out_dir = args.out_dir or args.pics_dir
    os.makedirs(out_dir, exist_ok=True)
    csv.field_size_limit(10 ** 7)

    # aid -> buy_et, from the CSV (the shared entry moment for all of an article's tickers)
    buy_et: dict[int, str] = {}
    with open(args.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            buy_et[int(r["article_id"])] = r["buy_et"]

    # article ids present in the pics folder
    aids = sorted({int(m.group(1)) for f in os.listdir(args.pics_dir)
                   if (m := _FNAME.match(f))})
    if not aids:
        print(f"No <ticker>_<id>_<date>.jpg charts found in {args.pics_dir}.")
        return 0
    tickers_by_aid = article_tickers(aids)

    # build the render jobs: every ticker of every article, minus charts already present
    jobs: list[tuple[str, int, str, str]] = []  # ticker, aid, buy_et, out_path
    skipped_existing = no_buyet = 0
    for aid in aids:
        ts = buy_et.get(aid)
        if not ts:
            no_buyet += 1
            continue
        bdate = ts.split(" ")[0]
        for tk in tickers_by_aid.get(aid, []):
            out = os.path.join(out_dir, f"{tk}_{aid}_{bdate}.jpg")
            if os.path.exists(out):
                skipped_existing += 1
                continue
            jobs.append((tk, aid, ts, out))

    print(f"{len(aids)} article(s); {len(jobs)} new ticker-chart(s) to render "
          f"({skipped_existing} already present"
          f"{f', {no_buyet} missing buy_et' if no_buyet else ''}).")
    if not jobs:
        return 0

    done = failed = 0
    it = tqdm(jobs, unit="chart") if tqdm else jobs
    for tk, aid, ts, out in it:
        try:
            with open(os.devnull, "w") as null, redirect_stdout(null):
                tc.make_chart(tk, ts, output=out)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {tk} {aid} {ts}: {exc}", file=sys.stderr)

    print(f"Done. Wrote {done} chart(s); {failed} failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
