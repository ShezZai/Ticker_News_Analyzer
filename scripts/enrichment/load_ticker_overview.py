"""Create and populate ``public.ticker_overview`` with Yahoo Finance company
descriptions.

For every ticker in ``public.ticker_data`` (or an explicit ``--tickers`` list)
this fetches the company profile description — the "Description" block on
https://finance.yahoo.com/quote/<TICKER>/profile/ — via yfinance
(``Ticker.info["longBusinessSummary"]``) and upserts it.

Table:
    public.ticker_overview
        ticker       text primary key   -- e.g. NVDA
        description  text               -- Yahoo profile business summary
        scraped_at   timestamptz        -- when the description was fetched

Resumable: each row is committed individually, and tickers that already have a
row are skipped unless ``--refresh`` is passed.

Usage:
    python load_ticker_overview.py
    python load_ticker_overview.py --tickers NVDA,AMD
    python load_ticker_overview.py --refresh --delay 1.0

Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import psycopg
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"
DEFAULT_DELAY = 0.5  # seconds between Yahoo requests


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the ticker_overview table if it doesn't already exist."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS public.ticker_overview ("
            "  ticker       text PRIMARY KEY, "
            "  description  text, "
            "  scraped_at   timestamptz NOT NULL DEFAULT now()"
            ")"
        )
    conn.commit()


def fetch_description(ticker: str) -> Optional[str]:
    """Return the Yahoo profile business summary for *ticker*, or None."""
    info = yf.Ticker(ticker).info or {}
    description = (info.get("longBusinessSummary") or "").strip()
    return description or None


def list_tickers(conn: psycopg.Connection) -> List[str]:
    """All tickers from public.ticker_data."""
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM public.ticker_data ORDER BY ticker")
        return [row[0] for row in cur.fetchall()]


def existing_tickers(conn: psycopg.Connection) -> set:
    """Tickers that already have an overview row."""
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM public.ticker_overview")
        return {row[0] for row in cur.fetchall()}


def select_pending(tickers: List[str], done: set, refresh: bool) -> List[str]:
    """Tickers still to fetch: everything when *refresh*, else the new ones."""
    if refresh:
        return list(tickers)
    return [t for t in tickers if t not in done]


def upsert(conn: psycopg.Connection, ticker: str, description: str) -> None:
    """Insert or replace one overview row; commits immediately."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.ticker_overview (ticker, description, scraped_at) "
            "VALUES (%s, %s, now()) "
            "ON CONFLICT (ticker) DO UPDATE SET "
            "  description = EXCLUDED.description, "
            "  scraped_at = EXCLUDED.scraped_at",
            (ticker, description),
        )
    conn.commit()


def load(tickers: Optional[List[str]], refresh: bool, delay: float) -> None:
    """Fetch + upsert descriptions for every pending ticker."""
    conn = psycopg.connect(DB_DSN)
    try:
        ensure_schema(conn)
        if tickers is None:
            tickers = list_tickers(conn)
            if not tickers:
                print("public.ticker_data is empty - load it first or pass --tickers")
                return
        pending = select_pending(tickers, existing_tickers(conn), refresh)
        skipped = len(tickers) - len(pending)
        print(f"{len(tickers)} ticker(s), {skipped} already loaded, fetching {len(pending)}")

        loaded, empty, failed = 0, [], []
        for i, ticker in enumerate(pending):
            if i:
                time.sleep(delay)
            try:
                description = fetch_description(ticker)
            except Exception as exc:  # noqa: BLE001 - keep going on any Yahoo hiccup
                failed.append(ticker)
                print(f"  {ticker}: FAILED ({exc})")
                continue
            if description is None:
                empty.append(ticker)
                print(f"  {ticker}: no description on Yahoo, skipped")
                continue
            upsert(conn, ticker, description)
            loaded += 1
            print(f"  {ticker}: ok ({len(description)} chars)")

        print(
            f"Done: {loaded} loaded, {skipped} skipped (already present), "
            f"{len(empty)} without description, {len(failed)} failed"
        )
        if empty:
            print(f"  no description: {', '.join(empty)}")
        if failed:
            print(f"  failed: {', '.join(failed)}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Yahoo Finance company descriptions into public.ticker_overview."
    )
    parser.add_argument(
        "--tickers",
        help="comma-separated tickers to load (default: all of public.ticker_data)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-fetch tickers that already have an overview row",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"seconds to wait between Yahoo requests (default {DEFAULT_DELAY})",
    )
    args = parser.parse_args()

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    load(tickers, refresh=args.refresh, delay=args.delay)


if __name__ == "__main__":
    main()
