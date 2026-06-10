"""Offline tests for scripts/enrichment/load_ticker_overview.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "enrichment"))

import load_ticker_overview as lto


class FakeYfTicker:
    def __init__(self, info):
        self.info = info


def test_fetch_description_returns_summary(monkeypatch):
    monkeypatch.setattr(
        lto.yf, "Ticker",
        lambda t: FakeYfTicker({"longBusinessSummary": "NVIDIA makes GPUs."}),
    )
    assert lto.fetch_description("NVDA") == "NVIDIA makes GPUs."


def test_fetch_description_missing_key(monkeypatch):
    monkeypatch.setattr(lto.yf, "Ticker", lambda t: FakeYfTicker({"sector": "Tech"}))
    assert lto.fetch_description("NVDA") is None


def test_fetch_description_blank_summary(monkeypatch):
    monkeypatch.setattr(
        lto.yf, "Ticker", lambda t: FakeYfTicker({"longBusinessSummary": "   "})
    )
    assert lto.fetch_description("NVDA") is None


def test_fetch_description_none_info(monkeypatch):
    monkeypatch.setattr(lto.yf, "Ticker", lambda t: FakeYfTicker(None))
    assert lto.fetch_description("NVDA") is None


def test_select_pending_skips_existing():
    assert lto.select_pending(["A", "B", "C"], {"B"}, refresh=False) == ["A", "C"]


def test_select_pending_refresh_keeps_all():
    assert lto.select_pending(["A", "B"], {"A", "B"}, refresh=True) == ["A", "B"]


def test_select_pending_all_new():
    assert lto.select_pending(["A", "B"], set(), refresh=False) == ["A", "B"]
