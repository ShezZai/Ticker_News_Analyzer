"""Fetch news+sentiment for the AI-compute universe (118 tickers) since 2024-11-01."""

import csv
from datetime import date

from ticker_news import fetch_news_csv

UNIVERSE_CSV = "ai_compute_us_market_universe_consolidated_segments_min5.csv"


def load_tickers(path: str) -> list[str]:
    # utf-8-sig strips the BOM so the first column is "ticker", not "﻿ticker".
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [row["ticker"].strip() for row in csv.DictReader(fh) if row.get("ticker")]


if __name__ == "__main__":
    tickers = load_tickers(UNIVERSE_CSV)
    print(f"Loaded {len(tickers)} tickers")

    fetch_news_csv(
        ticker_list=tickers,
        start_date="2024-11-01",
        end_date=date.today().isoformat(),   # through today
        output_path="ai_compute_news_since_2024-11-01.csv",
    )
