"""Create and populate ``public.ticker_data`` from the market-universe CSV.

Reads ai_compute_us_market_universe_consolidated_segments_min5.csv and loads a
ticker -> AI-segment lookup table into Postgres. This is the table-backed
equivalent of the hard-coded ``TICKER_SEGMENTS`` dict in ``tag_segments.py``,
with the company name carried alongside each ticker.

Table:
    public.ticker_data
        ticker        text primary key  -- e.g. NVDA (the table's index)
        company_name  text              -- e.g. NVIDIA Corporation
        segment       text              -- primary_ai_segment from the CSV

Re-runnable: rows are upserted on ``ticker``.

Usage:
    python load_ticker_segments.py
    python load_ticker_segments.py --csv path/to/universe.csv

Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"
DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "ai_compute_us_market_universe_consolidated_segments_min5.csv"
)


def ensure_schema(conn: psycopg.Connection) -> None:
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


def load(csv_path: Path) -> int:
    """Create the table and upsert every ticker from the CSV."""
    rows = read_rows(csv_path)
    print(f"Read {len(rows)} ticker(s) from {csv_path.name}")

    conn = psycopg.connect(DB_DSN)
    try:
        ensure_schema(conn)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load ticker -> segment + company_name into Postgres."
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help="path to the market-universe CSV (defaults to the repo copy)",
    )
    args = parser.parse_args()
    load(args.csv)


if __name__ == "__main__":
    main()
