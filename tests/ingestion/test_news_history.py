import pytest

from ticker_news.ingestion import news_history as nh
from ticker_news.ingestion.massive_rest import BASE_URL, MassiveAPIError


def _articles(ticker):
    return [{
        "article_url": f"https://x.com/{ticker.lower()}",
        "published_utc": "2025-01-02T03:04:05Z",
        "publisher": {"name": "Pub"},
        "insights": [
            {"ticker": ticker, "sentiment": "positive", "sentiment_reasoning": "why"},
            {"ticker": "OTHER", "sentiment": "neutral", "sentiment_reasoning": ""},
        ],
    }]


def test_fetch_news_rows_extracts_per_ticker_sentiment(monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles(t))
    rows = nh.fetch_news_rows(["NVDA"], "2025-01-01", "2025-01-31")
    assert rows == [{
        "ticker": "NVDA",
        "article_url": "https://x.com/nvda",
        "published_utc": "2025-01-02T03:04:05Z",
        "sentiment": "positive",
        "sentiment_reasoning": "why",
        "publisher_name": "Pub",
    }]


def test_fetch_news_rows_dedupes_ticker_url_pairs(monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles("NVDA") * 2)
    rows = nh.fetch_news_rows(["NVDA"], "2025-01-01", "2025-01-31")
    assert len(rows) == 1


def test_write_news_csv_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles(t))
    out = tmp_path / "news.csv"
    nh.fetch_news_csv(["NVDA"], "2025-01-01", "2025-01-31", output_path=str(out))
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == (
        "ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name"
    )
    assert "NVDA,https://x.com/nvda" in text


def test_fetch_range_paginates_reattaching_api_key(monkeypatch):
    calls = []

    def fake_request(session, url, params):
        calls.append((url, params))
        if url == BASE_URL:
            return {"results": [{"article_url": "https://x.com/1"}],
                    "next_url": "https://api.massive.com/page2"}
        return {"results": [{"article_url": "https://x.com/2"}]}

    monkeypatch.setattr(nh, "_request", fake_request)
    articles = nh.fetch_range("NVDA", "2025-01-01", "2025-01-31", key="k")
    assert [a["article_url"] for a in articles] == ["https://x.com/1", "https://x.com/2"]
    first_url, first_params = calls[0]
    assert first_url == BASE_URL
    assert first_params["ticker"] == "NVDA"
    assert first_params["published_utc.gte"] == "2025-01-01"
    assert first_params["published_utc.lte"] == "2025-01-31"
    assert first_params["order"] == "asc"
    assert first_params["sort"] == "published_utc"
    assert first_params["apiKey"] == "k"
    # next_url carries the cursor + filters but not the apiKey: re-attach it.
    assert calls[1] == ("https://api.massive.com/page2", {"apiKey": "k"})


def test_fetch_range_requires_api_key(monkeypatch):
    monkeypatch.setattr(nh, "get_settings", lambda: type("S", (), {"massive_api_key": None})())
    with pytest.raises(MassiveAPIError, match="MASSIVE_API_KEY"):
        nh.fetch_range("NVDA", "2025-01-01", "2025-01-31")
