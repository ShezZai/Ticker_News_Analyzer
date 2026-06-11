"""Reference-data loaders: ticker universe and Yahoo Finance company overviews.

Merges the legacy ``load_ticker_data.py`` (universe) and
``load_ticker_overview.py`` (Yahoo descriptions) scripts into a
single module with renamed entry points:

    ensure_universe_schema / load_universe   — from load_ticker_data.py
    ensure_overview_schema / load_overviews  — from load_ticker_overview.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Optional, Tuple

import psycopg

# repo root: src/ticker_news/enrichment/ -> parents[0]=enrichment, [1]=ticker_news,
# [2]=src, [3]=repo root
DEFAULT_CSV: Path = (
    Path(__file__).resolve().parents[3]
    / "ai_compute_us_market_universe_consolidated_segments_min5.csv"
)

DEFAULT_DELAY = 0.5  # seconds between Yahoo requests

# ---------------------------------------------------------------------------
# Universe (ticker_data) half — ported from load_ticker_data.py
# ---------------------------------------------------------------------------


def ensure_universe_schema(conn: psycopg.Connection) -> None:
    """Create the ticker_data table if it doesn't already exist."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS public.ticker_data ("
            "  ticker        text PRIMARY KEY, "
            "  company_name  text, "
            "  segment       text"
            ")"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS ticker_data_segment_idx "
            "ON public.ticker_data (segment);"
        )
    conn.commit()


def read_rows(csv_path: Path) -> List[Tuple[str, str, str]]:
    """Pull (ticker, company_name, primary_ai_segment) from the universe CSV."""
    rows: List[Tuple[str, str, str]] = []
    # utf-8-sig strips the BOM present at the start of the file.
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            company_name = (row.get("company_name") or "").strip() or None
            segment = (row.get("primary_ai_segment") or "").strip() or None
            rows.append((ticker, company_name, segment))
    return rows


def load_universe(csv_path: Path) -> int:
    """Create the table and upsert every ticker from the CSV."""
    from ticker_news.shared.db import connect

    rows = read_rows(csv_path)
    print(f"Read {len(rows)} ticker(s) from {csv_path.name}")

    conn = connect()
    try:
        ensure_universe_schema(conn)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO public.ticker_data (ticker, company_name, segment) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (ticker) DO UPDATE SET "
                "  company_name = EXCLUDED.company_name, "
                "  segment = EXCLUDED.segment",
                rows,
            )
        conn.commit()
        print(f"Upserted {len(rows)} row(s) into public.ticker_data")
        return len(rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Overview (ticker_overview) half — ported from load_ticker_overview.py
# ---------------------------------------------------------------------------


def ensure_overview_schema(conn: psycopg.Connection) -> None:
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
    import yfinance as yf

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


def load_overviews(
    tickers: Optional[List[str]] = None,
    refresh: bool = False,
    delay: float = DEFAULT_DELAY,
) -> None:
    """Fetch + upsert descriptions for every pending ticker."""
    from ticker_news.shared.db import connect

    conn = connect()
    try:
        ensure_overview_schema(conn)
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
