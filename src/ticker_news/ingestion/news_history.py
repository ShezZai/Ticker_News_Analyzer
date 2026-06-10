"""Historical news + provider sentiment fetch from the Massive.com REST API.

Port of the legacy scripts/data_getting_parsing/ticker_news.py range-fetch
half. Produces a CSV with one row per (ticker, article) pair:

    ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name

sentiment/sentiment_reasoning come from the article's `insights` entry whose
ticker matches the row's ticker. Retry/backoff plumbing is shared with the
live poller via massive_rest._request.
"""

from __future__ import annotations

import csv
from typing import Dict, Iterable, List, Optional

import requests

from ticker_news.ingestion.massive_rest import (
    BASE_URL,
    PAGE_LIMIT,
    MassiveAPIError,
    _request,
)
from ticker_news.shared.config import get_settings

CSV_HEADER = [
    "ticker",
    "article_url",
    "published_utc",
    "sentiment",
    "sentiment_reasoning",
    "publisher_name",
]


def fetch_range(
    ticker: str, start_iso: str, end_iso: str, *, key: str | None = None
) -> list[dict]:
    """All articles for `ticker` in [start, end] via /v2/reference/news pagination.

    Follows next_url, re-attaching the apiKey (next_url carries the cursor +
    filters but not the key).
    """
    key = key or get_settings().massive_api_key
    if not key:
        raise MassiveAPIError("MASSIVE_API_KEY is not set (put it in .env).")
    out: list[dict] = []
    params: Optional[dict] = {
        "ticker": ticker,
        "published_utc.gte": start_iso,
        "published_utc.lte": end_iso,
        "order": "asc",
        "sort": "published_utc",
        "limit": PAGE_LIMIT,
        "apiKey": key,
    }
    url = BASE_URL
    with requests.Session() as session:
        while url:
            payload = _request(session, url, params)
            out.extend(payload.get("results", []) or [])
            url = payload.get("next_url")
            params = {"apiKey": key} if url else None
    return out


def _sentiment_for(article: dict, ticker: str) -> tuple[str, str]:
    """Return (sentiment, reasoning) for `ticker` from the article's insights."""
    for insight in article.get("insights") or []:
        if (insight.get("ticker") or "").upper() == ticker.upper():
            return insight.get("sentiment", ""), insight.get("sentiment_reasoning", "")
    return "", ""


def fetch_news_rows(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    *,
    key: str | None = None,
) -> List[Dict[str, str]]:
    """One CSV-shaped dict per (ticker, article) pair; dedupes pairs."""
    rows: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()  # (ticker, article_url) dedupe
    for ticker in [t.strip().upper() for t in tickers if t and t.strip()]:
        for article in fetch_range(ticker, start_date, end_date, key=key):
            url = article.get("article_url", "")
            pair = (ticker, url)
            if pair in seen:
                continue
            seen.add(pair)
            sentiment, reasoning = _sentiment_for(article, ticker)
            rows.append(
                {
                    "ticker": ticker,
                    "article_url": url,
                    "published_utc": article.get("published_utc", ""),
                    "sentiment": sentiment,
                    "sentiment_reasoning": reasoning,
                    "publisher_name": (article.get("publisher") or {}).get("name", ""),
                }
            )
    return rows


def fetch_news_csv(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    *,
    output_path: str = "news.csv",
    key: str | None = None,
) -> str:
    """fetch_news_rows + csv.DictWriter; returns the written path."""
    rows = fetch_news_rows(tickers, start_date, end_date, key=key)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
