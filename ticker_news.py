"""Fetch stock news with per-ticker sentiment from the Massive.com REST API.

Public entry point:

    fetch_news_csv(ticker_list, start_date, end_date, output_path="news.csv")

Produces a CSV with the header:

    ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name

Each row is one (ticker, article) pair. The sentiment/sentiment_reasoning are
taken from the article's `insights` entry that matches that ticker.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, Iterable, Iterator, List, Optional

import requests
from dotenv import load_dotenv

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

BASE_URL = "https://api.massive.com/v2/reference/news"
PAGE_LIMIT = 1000          # max page size allowed by the API
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0        # seconds, exponential
REQUEST_TIMEOUT = 30       # seconds

CSV_HEADER = [
    "ticker",
    "article_url",
    "published_utc",
    "sentiment",
    "sentiment_reasoning",
    "publisher_name",
]


class MassiveAPIError(RuntimeError):
    """Raised when the Massive API returns a non-recoverable error."""


def _get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("MASSIVE_API_KEY")
    if not key:
        raise MassiveAPIError(
            "No API key found. Pass api_key= or set MASSIVE_API_KEY in your .env"
        )
    return key


def _request(session: requests.Session, url: str, params: Optional[dict]) -> dict:
    """GET with retries on transient errors (429 / 5xx / network)."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"transient {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise MassiveAPIError(f"Request to {url} failed: {last_exc}")


def _iter_articles(
    session: requests.Session,
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> Iterator[dict]:
    """Yield every news article for `ticker` in the date range, across all pages."""
    params: Optional[dict] = {
        "ticker": ticker,
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "order": "asc",
        "sort": "published_utc",
        "limit": PAGE_LIMIT,
        "apiKey": api_key,
    }
    url = BASE_URL
    while url:
        payload = _request(session, url, params)
        yield from payload.get("results", []) or []
        url = payload.get("next_url")
        # next_url already carries the cursor + filters, but not the apiKey.
        params = {"apiKey": api_key} if url else None


def _sentiment_for(article: dict, ticker: str) -> tuple[str, str]:
    """Return (sentiment, reasoning) for `ticker` from the article's insights."""
    for insight in article.get("insights") or []:
        if (insight.get("ticker") or "").upper() == ticker.upper():
            return insight.get("sentiment", ""), insight.get("sentiment_reasoning", "")
    return "", ""


def fetch_news_rows(
    ticker_list: Iterable[str],
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Fetch news rows for each ticker. Returns a list of dicts keyed by CSV_HEADER.

    Args:
        ticker_list: iterable of ticker symbols, e.g. ["AAPL", "MSFT"].
        start_date:  inclusive lower bound, "YYYY-MM-DD" (or RFC3339 timestamp).
        end_date:    inclusive upper bound, "YYYY-MM-DD" (or RFC3339 timestamp).
        api_key:     overrides MASSIVE_API_KEY from the environment.
    """
    key = _get_api_key(api_key)
    tickers = [t.strip().upper() for t in ticker_list if t and t.strip()]
    rows: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()  # (ticker, article_url) dedupe

    iterator = tqdm(tickers, desc="tickers") if tqdm else tickers
    with requests.Session() as session:
        for ticker in iterator:
            for article in _iter_articles(session, ticker, start_date, end_date, key):
                url = article.get("article_url", "")
                key_pair = (ticker, url)
                if key_pair in seen:
                    continue
                seen.add(key_pair)
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
    ticker_list: Iterable[str],
    start_date: str,
    end_date: str,
    output_path: str = "news.csv",
    api_key: Optional[str] = None,
) -> str:
    """Fetch news for the given tickers/date range and write a CSV.

    Returns the path to the written CSV.
    """
    rows = fetch_news_rows(ticker_list, start_date, end_date, api_key=api_key)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows for "
          f"{len(set(r['ticker'] for r in rows))} ticker(s) -> {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch stock news sentiment to CSV.")
    parser.add_argument("tickers", help="Comma-separated tickers, e.g. AAPL,MSFT")
    parser.add_argument("start_date", help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("end_date", help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("-o", "--output", default="news.csv", help="Output CSV path")
    args = parser.parse_args()

    fetch_news_csv(
        args.tickers.split(","),
        args.start_date,
        args.end_date,
        output_path=args.output,
    )
