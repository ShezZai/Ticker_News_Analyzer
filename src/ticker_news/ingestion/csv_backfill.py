"""CSV backfill source — wraps the legacy news-CSV flow as a NewsFeedSource."""

from __future__ import annotations

from typing import AsyncIterator

from ticker_news.ingestion.feed import FeedItem
from ticker_news.scraping.csv_source import read_jobs


class CsvBackfillSource:
    def __init__(self, csv_path: str, *, limit: int | None = None):
        self.csv_path = csv_path
        self.limit = limit

    async def stream(self) -> AsyncIterator[FeedItem]:
        produced = 0
        # Blocking file I/O is intentional: backfill reads one bounded CSV at
        # startup. A high-throughput source should use asyncio.to_thread.
        for job in read_jobs(self.csv_path):
            if self.limit is not None and produced >= self.limit:
                return
            produced += 1
            yield FeedItem(
                url=job.url,
                tickers=job.tickers,
                published_utc=job.published_utc,
                publisher=job.publisher,
            )
