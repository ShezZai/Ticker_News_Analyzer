from datetime import time as dtime

import pytest
import requests

from ticker_news.research import market_data as md


def _resp(status, payload=None):
    class R:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise requests.HTTPError(f"{status}", response=self)

        def json(self):
            return payload or {}

    return R()


def test_session_of_boundaries():
    assert md.session_of(dtime(3, 59)) == "closed"
    assert md.session_of(dtime(4, 0)) == "premarket"
    assert md.session_of(dtime(9, 29)) == "premarket"
    assert md.session_of(dtime(9, 30)) == "regular"
    assert md.session_of(dtime(15, 59)) == "regular"
    assert md.session_of(dtime(16, 0)) == "after_hours"
    assert md.session_of(dtime(19, 59)) == "after_hours"
    assert md.session_of(dtime(20, 0)) == "closed"


def test_api_key_explicit_wins(monkeypatch):
    assert md.api_key("abc") == "abc"


def test_api_key_missing_raises(monkeypatch):
    monkeypatch.setattr(md, "_settings_key", lambda: "")
    with pytest.raises(RuntimeError):
        md.api_key(None)


def test_get_json_does_not_retry_client_errors(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(url)
        return _resp(404)

    monkeypatch.setattr(md.requests, "get", fake_get)
    monkeypatch.setattr(md._time, "sleep",
                        lambda s: pytest.fail("sleep called for non-transient error"))
    with pytest.raises(RuntimeError):
        md.get_json("https://api.massive.com/x", {})
    assert len(calls) == 1


def test_get_json_retries_transient_then_succeeds(monkeypatch):
    responses = [_resp(503), _resp(200, {"results": [1]})]
    calls = []
    sleeps = []

    def fake_get(url, params, timeout):
        calls.append(url)
        return responses[len(calls) - 1]

    monkeypatch.setattr(md.requests, "get", fake_get)
    monkeypatch.setattr(md._time, "sleep", lambda s: sleeps.append(s))
    assert md.get_json("https://api.massive.com/x", {}) == {"results": [1]}
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_fetch_bars_paginates(monkeypatch):
    pages = [
        {"results": [{"t": 1}], "next_url": "https://api.massive.com/page2"},
        {"results": [{"t": 2}], "next_url": None},
    ]
    calls = []

    def fake_get_json(url, params):
        calls.append((url, dict(params)))
        return pages[len(calls) - 1]

    monkeypatch.setattr(md, "get_json", fake_get_json)
    bars = md.fetch_bars("NVDA", span="minute", frm="2025-01-02", to="2025-01-02",
                         key="k")
    assert [b["t"] for b in bars] == [1, 2]
    assert calls[0][1]["apiKey"] == "k"          # key on first call
    assert calls[1][1] == {"apiKey": "k"}        # next_url keeps only the key
