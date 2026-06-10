"""Offline tests for ticker_news.enrichment.reference_data."""

import pytest

import ticker_news.enrichment.reference_data as rd


class FakeYfTicker:
    def __init__(self, info):
        self.info = info


def test_fetch_description_returns_summary(monkeypatch):
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda t: FakeYfTicker({"longBusinessSummary": "NVIDIA makes GPUs."}),
    )
    assert rd.fetch_description("NVDA") == "NVIDIA makes GPUs."


def test_fetch_description_missing_key(monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", lambda t: FakeYfTicker({"sector": "Tech"}))
    assert rd.fetch_description("NVDA") is None


def test_fetch_description_blank_summary(monkeypatch):
    monkeypatch.setattr(
        "yfinance.Ticker", lambda t: FakeYfTicker({"longBusinessSummary": "   "})
    )
    assert rd.fetch_description("NVDA") is None


def test_fetch_description_none_info(monkeypatch):
    monkeypatch.setattr("yfinance.Ticker", lambda t: FakeYfTicker(None))
    assert rd.fetch_description("NVDA") is None


def test_select_pending_skips_existing():
    assert rd.select_pending(["A", "B", "C"], {"B"}, refresh=False) == ["A", "C"]


def test_select_pending_refresh_keeps_all():
    assert rd.select_pending(["A", "B"], {"A", "B"}, refresh=True) == ["A", "B"]


def test_select_pending_all_new():
    assert rd.select_pending(["A", "B"], set(), refresh=False) == ["A", "B"]
