from datetime import datetime, timedelta, timezone

from ticker_news.ingestion.massive_rest import MassiveRestSource

ARTICLE_A = {
    "article_url": "https://example.com/a",
    "published_utc": "2026-06-09T10:00:00Z",
    "publisher": {"name": "Benzinga"},
    "insights": [{"ticker": "NVDA", "sentiment": "positive", "sentiment_reasoning": "beat"}],
}
ARTICLE_A_AMD = {**ARTICLE_A, "insights": [{"ticker": "AMD", "sentiment": "neutral", "sentiment_reasoning": ""}]}
ARTICLE_B = {
    "article_url": "https://example.com/b",
    "published_utc": "2026-06-09T11:00:00Z",
    "publisher": {"name": "Reuters"},
    "insights": [],
}


def _source(pages, tickers=("NVDA", "AMD")):
    calls = []

    def fake_fetch(ticker, since_iso):
        calls.append((ticker, since_iso))
        return pages.get(ticker, [])

    src = MassiveRestSource(
        list(tickers), poll_interval_s=0.01, lookback=timedelta(days=1),
        fetch_articles=fake_fetch, max_polls=1,
    )
    return src, calls


async def _collect(src):
    return [item async for item in src.stream()]


async def test_groups_same_url_across_tickers_merging_tickers():
    src, _ = _source({"NVDA": [ARTICLE_A], "AMD": [ARTICLE_A_AMD]})
    items = await _collect(src)
    assert len(items) == 1
    assert items[0].url == "https://example.com/a"
    assert sorted(items[0].tickers) == ["AMD", "NVDA"]


async def test_sentiment_lands_in_source_meta_per_ticker():
    src, _ = _source({"NVDA": [ARTICLE_A], "AMD": []})
    items = await _collect(src)
    meta = items[0].source_meta
    assert meta["sentiments"]["NVDA"]["sentiment"] == "positive"


async def test_cursor_advances_past_seen_articles():
    src, calls = _source({"NVDA": [ARTICLE_A, ARTICLE_B], "AMD": []})
    await _collect(src)
    # after one poll, the per-ticker cursor moved to the newest published_utc
    assert src.cursor("NVDA") == datetime(2026, 6, 9, 11, 0, tzinfo=timezone.utc)


async def test_second_poll_does_not_reyield_seen_urls():
    pages = {"NVDA": [ARTICLE_A], "AMD": []}
    calls = []

    def fake_fetch(ticker, since_iso):
        calls.append((ticker, since_iso))
        return pages.get(ticker, [])

    src = MassiveRestSource(
        ["NVDA", "AMD"], poll_interval_s=0.01, lookback=timedelta(days=1),
        fetch_articles=fake_fetch, max_polls=2,
    )
    items = await _collect(src)
    assert len(items) == 1  # second poll returns the same article; not re-yielded
