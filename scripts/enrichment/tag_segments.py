"""Tag each article with its primary ticker/segment and any secondary ones.

The ticker -> AI-segment lookup is hard-coded below in ``TICKER_SEGMENTS``
(derived from ai_compute_us_market_universe_consolidated_segments_min5.csv's
``primary_ai_segment`` column). From each row's ``tickers`` array it derives
four columns on ``public.articles``:

    primary_ticker   text     -- the first ticker in `tickers`
    primary_segment  text     -- that ticker's AI segment (NULL if unknown)
    more_tickers     text[]   -- the remaining tickers (NULL if only one)
    more_segments    text[]   -- distinct additional segments among `more_tickers`
                                 (excludes primary_segment; NULL if none)

Re-runnable: by default every row is recomputed; pass --only-missing to fill
just the rows that don't have a primary_ticker yet.

Usage:
    python tag_segments.py
    python tag_segments.py --only-missing
    python tag_segments.py --no-index

Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import psycopg
from dotenv import load_dotenv

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"
BATCH_SIZE = 1000

# Hard-coded ticker -> primary AI segment, from the market-universe CSV.
TICKER_SEGMENTS: Dict[str, str] = {
    "NVDA":  "AI Compute Silicon & Semiconductor IP",
    "AMD":   "AI Compute Silicon & Semiconductor IP",
    "INTC":  "AI Compute Silicon & Semiconductor IP",
    "AVGO":  "AI Compute Silicon & Semiconductor IP",
    "MRVL":  "AI Compute Silicon & Semiconductor IP",
    "QCOM":  "AI Compute Silicon & Semiconductor IP",
    "ARM":   "AI Compute Silicon & Semiconductor IP",
    "CBRS":  "AI Compute Silicon & Semiconductor IP",
    "ALAB":  "Networking & Interconnect",
    "CRDO":  "Networking & Interconnect",
    "RMBS":  "Memory & Storage",
    "LSCC":  "AI Compute Silicon & Semiconductor IP",
    "AMBA":  "AI Compute Silicon & Semiconductor IP",
    "TXN":   "Power & Analog Semiconductors",
    "ADI":   "Power & Analog Semiconductors",
    "MCHP":  "Power & Analog Semiconductors",
    "MPWR":  "Power & Analog Semiconductors",
    "ON":    "Power & Analog Semiconductors",
    "WOLF":  "Power & Analog Semiconductors",
    "NVTS":  "Power & Analog Semiconductors",
    "POWI":  "Power & Analog Semiconductors",
    "CEVA":  "AI Compute Silicon & Semiconductor IP",
    "MU":    "Memory & Storage",
    "SNDK":  "Memory & Storage",
    "WDC":   "Memory & Storage",
    "STX":   "Memory & Storage",
    "PSTG":  "Memory & Storage",
    "NTAP":  "Memory & Storage",
    "SIMO":  "Memory & Storage",
    "MRAM":  "Memory & Storage",
    "TSM":   "Chip Manufacturing, Packaging & Materials",
    "GFS":   "Chip Manufacturing, Packaging & Materials",
    "UMC":   "Chip Manufacturing, Packaging & Materials",
    "STM":   "Power & Analog Semiconductors",
    "ASX":   "Chip Manufacturing, Packaging & Materials",
    "AMKR":  "Chip Manufacturing, Packaging & Materials",
    "ASML":  "Semiconductor Equipment",
    "AMAT":  "Semiconductor Equipment",
    "LRCX":  "Semiconductor Equipment",
    "KLAC":  "Semiconductor Equipment",
    "TER":   "Semiconductor Equipment",
    "AEHR":  "Semiconductor Equipment",
    "COHU":  "Semiconductor Equipment",
    "ACLS":  "Semiconductor Equipment",
    "UCTT":  "Semiconductor Equipment",
    "ICHR":  "Semiconductor Equipment",
    "VECO":  "Semiconductor Equipment",
    "ONTO":  "Semiconductor Equipment",
    "FORM":  "Semiconductor Equipment",
    "PLAB":  "Semiconductor Equipment",
    "MKSI":  "Semiconductor Equipment",
    "CAMT":  "Semiconductor Equipment",
    "ENTG":  "Chip Manufacturing, Packaging & Materials",
    "AXTI":  "Chip Manufacturing, Packaging & Materials",
    "IPGP":  "Chip Manufacturing, Packaging & Materials",
    "ANET":  "Networking & Interconnect",
    "CSCO":  "Networking & Interconnect",
    "CIEN":  "Networking & Interconnect",
    "AAOI":  "Networking & Interconnect",
    "COHR":  "Networking & Interconnect",
    "LITE":  "Networking & Interconnect",
    "FN":    "Networking & Interconnect",
    "FLEX":  "AI Servers & Full Systems",
    "CLS":   "AI Servers & Full Systems",
    "SANM":  "AI Servers & Full Systems",
    "JBL":   "AI Servers & Full Systems",
    "DELL":  "AI Servers & Full Systems",
    "SMCI":  "AI Servers & Full Systems",
    "HPE":   "AI Servers & Full Systems",
    "IBM":   "AI Servers & Full Systems",
    "HPQ":   "AI Servers & Full Systems",
    "LNVGY": "AI Servers & Full Systems",
    "MSFT":  "Cloud / AI Infrastructure Operators",
    "GOOGL": "Cloud / AI Infrastructure Operators",
    "GOOG":  "Cloud / AI Infrastructure Operators",
    "AMZN":  "Cloud / AI Infrastructure Operators",
    "META":  "AI Software Infrastructure & Foundation Models",
    "ORCL":  "Cloud / AI Infrastructure Operators",
    "SNOW":  "AI Software Infrastructure & Foundation Models",
    "PLTR":  "AI Applications",
    "DDOG":  "AI Software Infrastructure & Foundation Models",
    "MDB":   "AI Software Infrastructure & Foundation Models",
    "ESTC":  "AI Software Infrastructure & Foundation Models",
    "GTLB":  "AI Software Infrastructure & Foundation Models",
    "PATH":  "AI Applications",
    "AI":    "AI Applications",
    "SOUN":  "AI Applications",
    "BBAI":  "AI Applications",
    "S":     "AI Applications",
    "CRWD":  "AI Applications",
    "PANW":  "AI Applications",
    "NOW":   "AI Applications",
    "ADBE":  "AI Applications",
    "SNPS":  "AI Software Infrastructure & Foundation Models",
    "CDNS":  "AI Software Infrastructure & Foundation Models",
    "ANSS":  "AI Software Infrastructure & Foundation Models",
    "VRT":   "Power, Cooling & Data-Center Infrastructure",
    "ETN":   "Power, Cooling & Data-Center Infrastructure",
    "GEV":   "Power, Cooling & Data-Center Infrastructure",
    "PWR":   "Power, Cooling & Data-Center Infrastructure",
    "HUBB":  "Power, Cooling & Data-Center Infrastructure",
    "POWL":  "Power, Cooling & Data-Center Infrastructure",
    "NVT":   "Power, Cooling & Data-Center Infrastructure",
    "CARR":  "Power, Cooling & Data-Center Infrastructure",
    "TT":    "Power, Cooling & Data-Center Infrastructure",
    "EQIX":  "Cloud / AI Infrastructure Operators",
    "DLR":   "Cloud / AI Infrastructure Operators",
    "CORZ":  "Cloud / AI Infrastructure Operators",
    "IREN":  "Cloud / AI Infrastructure Operators",
    "APLD":  "Cloud / AI Infrastructure Operators",
    "NBIS":  "Cloud / AI Infrastructure Operators",
    "KEYS":  "Semiconductor Equipment",
    "GLW":   "Networking & Interconnect",
    "APH":   "Networking & Interconnect",
    "TEL":   "Networking & Interconnect",
    "BELFA": "Networking & Interconnect",
    "BELFB": "Networking & Interconnect",
    "NOK":   "Networking & Interconnect",
}


def ensure_schema(conn: psycopg.Connection) -> None:
    """Add the segment columns if they don't already exist."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE public.articles "
            "ADD COLUMN IF NOT EXISTS primary_ticker  text, "
            "ADD COLUMN IF NOT EXISTS primary_segment text, "
            "ADD COLUMN IF NOT EXISTS more_tickers    text[], "
            "ADD COLUMN IF NOT EXISTS more_segments   text[]"
        )
    conn.commit()


def compute_row(
    tickers: Optional[Sequence[str]], seg_map: Dict[str, str]
) -> Tuple[Optional[str], Optional[str], Optional[List[str]], Optional[List[str]]]:
    """Derive (primary_ticker, primary_segment, more_tickers, more_segments)."""
    clean = [t.strip().upper() for t in (tickers or []) if t and t.strip()]
    if not clean:
        return None, None, None, None

    primary_ticker = clean[0]
    primary_segment = seg_map.get(primary_ticker)
    more_tickers = clean[1:]

    # distinct segments contributed by the other tickers, beyond the primary one
    more_segments: List[str] = []
    seen = {primary_segment} if primary_segment else set()
    for t in more_tickers:
        seg = seg_map.get(t)
        if seg and seg not in seen:
            seen.add(seg)
            more_segments.append(seg)

    return (
        primary_ticker,
        primary_segment,
        more_tickers or None,
        more_segments or None,
    )


def create_indexes(conn: psycopg.Connection) -> None:
    """Build helpful indexes for filtering by ticker/segment (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_primary_ticker_idx "
            "ON public.articles (primary_ticker);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_primary_segment_idx "
            "ON public.articles (primary_segment);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_more_segments_idx "
            "ON public.articles USING gin (more_segments);"
        )
    conn.commit()


def tag_all(
    only_missing: bool = False,
    build_index: bool = True,
) -> int:
    """Populate the segment columns for every (or only untagged) article."""
    seg_map = TICKER_SEGMENTS
    print(f"Using {len(seg_map)} hard-coded ticker->segment mappings")

    conn = psycopg.connect(DB_DSN)
    try:
        ensure_schema(conn)

        where = "WHERE primary_ticker IS NULL" if only_missing else ""
        with conn.cursor() as cur:
            cur.execute(f"SELECT id, tickers FROM public.articles {where} ORDER BY id")
            rows = cur.fetchall()

        if not rows:
            print("Nothing to tag.")
            return 0

        print(f"Tagging {len(rows)} article(s) ...")
        unmatched: set[str] = set()  # primary tickers not present in the mapping
        with_primary_seg = 0
        with_more = 0

        updates: List[tuple] = []
        for rid, tickers in rows:
            pt, ps, mt, ms = compute_row(tickers, seg_map)
            if pt and ps is None:
                unmatched.add(pt)
            if ps:
                with_primary_seg += 1
            if ms:
                with_more += 1
            updates.append((pt, ps, mt, ms, rid))

        batches = range(0, len(updates), BATCH_SIZE)
        iterator = tqdm(batches, unit="batch") if tqdm else batches
        for start in iterator:
            chunk = updates[start : start + BATCH_SIZE]
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE public.articles SET primary_ticker = %s, "
                    "primary_segment = %s, more_tickers = %s, more_segments = %s "
                    "WHERE id = %s",
                    chunk,
                )
            conn.commit()

        print(
            f"Done. {with_primary_seg} row(s) got a primary_segment, "
            f"{with_more} row(s) got more_segments."
        )
        if unmatched:
            sample = ", ".join(sorted(unmatched)[:15])
            print(
                f"{len(unmatched)} primary ticker(s) had no segment in the mapping "
                f"(e.g. {sample})"
            )

        if build_index:
            print("Building segment/ticker indexes ...")
            create_indexes(conn)
        return len(updates)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag articles with primary/secondary tickers and AI segments."
    )
    parser.add_argument(
        "--only-missing", action="store_true",
        help="only tag rows that don't have a primary_ticker yet",
    )
    parser.add_argument(
        "--no-index", dest="build_index", action="store_false",
        help="skip building the ticker/segment indexes",
    )
    args = parser.parse_args()

    tag_all(
        only_missing=args.only_missing,
        build_index=args.build_index,
    )


if __name__ == "__main__":
    main()
