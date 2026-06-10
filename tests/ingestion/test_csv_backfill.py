import pytest

from ticker_news.ingestion.csv_backfill import CsvBackfillSource
from ticker_news.ingestion.feed import FeedItem

CSV = """\
tickers,article_url,published_utc,publisher_name
"NVDA,AMD",https://example.com/a,2026-01-02T03:04:05Z,Benzinga
NVDA,https://example.com/b,,
,,2026-01-01T00:00:00Z,NoUrl
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "news.csv"
    p.write_text(CSV, encoding="utf-8")
    return str(p)


async def _collect(source):
    return [item async for item in source.stream()]


async def test_yields_feed_items_with_parsed_fields(csv_path):
    items = await _collect(CsvBackfillSource(csv_path))
    assert len(items) == 2  # the url-less row is dropped
    first = items[0]
    assert isinstance(first, FeedItem)
    assert first.url == "https://example.com/a"
    assert first.tickers == ["NVDA", "AMD"]
    assert first.published_utc is not None and first.published_utc.year == 2026
    assert first.publisher == "Benzinga"


async def test_missing_optional_fields_are_none(csv_path):
    items = await _collect(CsvBackfillSource(csv_path))
    second = items[1]
    assert second.url == "https://example.com/b"
    assert second.published_utc is None
    assert second.publisher is None


async def test_limit_caps_rows(csv_path):
    items = await _collect(CsvBackfillSource(csv_path, limit=1))
    assert len(items) == 1
