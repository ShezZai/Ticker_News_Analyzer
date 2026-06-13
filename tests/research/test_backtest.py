"""Offline tests for the analyst-panel verdict backtest (pure summarize +
evaluate with stubbed price fetching/simulation)."""

import csv
from datetime import datetime

import pytest

from ticker_news.research import backtest as bt
from ticker_news.research.backtest import summarize
from ticker_news.research.market_data import MARKET_TZ


def _r(action, conf, signed):
    return {"action": action, "confidence": conf, "signed_return_pct": signed}


def _verdict(aid=1, ticker="AAA", action="buy", conf=0.9, hour=10, minute=0):
    return {
        "article_id": aid,
        "url": f"https://x.test/{aid}",
        "ticker": ticker,
        "action": action,
        "confidence": conf,
        "published_utc": datetime(2025, 1, 6, hour, minute, tzinfo=MARKET_TZ),
        "title": f"headline {aid}",
    }


def _sim_row(gain):
    """A row in the shape ticker_scan.simulate returns."""
    return {
        "article_id": 0, "ticker": "AAA", "category": "",
        "published_et": "2025-01-06 10:00", "entry_session": "regular",
        "buy_et": "2025-01-06 10:00", "buy_price": 100.0,
        "hold_until": "same-day-close", "sell_date": "2025-01-06",
        "sell_close": 102.0, "gain_pct": gain, "title": "",
    }


# --------------------------------------------------------------------------- #
# summarize (pure)
# --------------------------------------------------------------------------- #
def test_summarize_win_rate_and_buckets():
    s = summarize([_r("buy", 0.9, 2.0), _r("buy", 0.9, -1.0), _r("sell", 0.6, 3.0)])
    assert s["overall"]["n"] == 3
    assert s["overall"]["win_rate"] == pytest.approx(2 / 3)
    assert s["by_action"]["buy"]["avg_return_pct"] == pytest.approx(0.5)
    assert s["by_confidence"][">=0.85"]["n"] == 2
    assert s["by_confidence"]["<0.7"]["n"] == 1


def test_summarize_bucket_boundaries():
    s = summarize([_r("buy", 0.7, 1.0), _r("buy", 0.85, 1.0), _r("buy", 0.6999, 1.0)])
    assert s["by_confidence"]["0.7-0.85"]["n"] == 1
    assert s["by_confidence"][">=0.85"]["n"] == 1
    assert s["by_confidence"]["<0.7"]["n"] == 1


def test_summarize_empty_input():
    s = summarize([])
    assert s["total"] == 0
    assert s["evaluated"] == 0
    assert s["skipped"] == {}
    assert s["overall"] == {"n": 0, "win_rate": None, "avg_return_pct": None}
    assert s["by_action"] == {}
    assert set(s["by_confidence"]) == {"<0.7", "0.7-0.85", ">=0.85"}
    assert all(b["n"] == 0 for b in s["by_confidence"].values())


def test_summarize_all_skipped_accounts_for_every_row():
    rows = [
        {**_r("buy", 0.9, None), "skip_reason": "skipped_after_hours"},
        {**_r("hold", 0.8, None), "skip_reason": "skipped_hold"},
        {**_r("hold", 0.8, None), "skip_reason": "skipped_hold"},
    ]
    s = summarize(rows)
    assert s["total"] == 3
    assert s["evaluated"] == 0
    assert s["skipped"] == {"skipped_after_hours": 1, "skipped_hold": 2}
    assert s["overall"]["n"] == 0 and s["overall"]["win_rate"] is None


def test_summarize_holds_count_but_do_not_score():
    # An evaluated hold (include_hold run) has signed_return_pct=None: it counts
    # toward n but is excluded from win_rate / avg_return_pct.
    s = summarize([_r("buy", 0.9, 2.0), _r("hold", 0.9, None)])
    assert s["overall"]["n"] == 2
    assert s["overall"]["win_rate"] == pytest.approx(1.0)
    assert s["overall"]["avg_return_pct"] == pytest.approx(2.0)
    assert s["by_action"]["hold"] == {"n": 1, "win_rate": None, "avg_return_pct": None}


# --------------------------------------------------------------------------- #
# evaluate with stubbed fetch_prices / simulate
# --------------------------------------------------------------------------- #
def test_evaluate_signs_returns_and_fetches_once_per_ticker(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bt, "fetch_prices",
        lambda tk, frm, to, key=None: calls.append((tk, frm, to, key)) or "PRICES",
    )
    monkeypatch.setattr(
        bt, "simulate",
        lambda art, prices, include_after_hours=False: _sim_row(2.0),
    )
    res = bt.evaluate(
        [_verdict(1, action="buy"), _verdict(2, action="sell")], key="k"
    )
    # two verdicts, same ticker, same window -> ONE price fetch
    assert calls == [("AAA", "2025-01-06", "2025-01-13", "k")]  # +PRICE_TAIL_DAYS
    assert [r["signed_return_pct"] for r in res] == [2.0, -2.0]
    assert all(r["skip_reason"] is None for r in res)
    assert res[0]["gain_pct"] == 2.0 and res[1]["gain_pct"] == 2.0
    assert res[0]["article_id"] == 1 and res[0]["action"] == "buy"
    assert res[0]["entry_session"] == "regular"


def test_evaluate_hold_skipped_by_default(monkeypatch):
    def boom(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("fetch_prices must not be called for hold-only input")

    monkeypatch.setattr(bt, "fetch_prices", boom)
    res = bt.evaluate([_verdict(action="hold")])
    assert len(res) == 1
    assert res[0]["skip_reason"] == "skipped_hold"
    assert res[0]["signed_return_pct"] is None


def test_evaluate_include_hold_tracks_unsigned(monkeypatch):
    monkeypatch.setattr(bt, "fetch_prices", lambda tk, frm, to, key=None: "PRICES")
    monkeypatch.setattr(
        bt, "simulate",
        lambda art, prices, include_after_hours=False: _sim_row(3.0),
    )
    res = bt.evaluate([_verdict(action="hold")], include_hold=True, key="k")
    assert res[0]["skip_reason"] is None
    assert res[0]["gain_pct"] == 3.0          # raw buy-the-news return is kept
    assert res[0]["signed_return_pct"] is None  # but a hold has no direction


def test_evaluate_simulate_none_skip_reasons(monkeypatch):
    monkeypatch.setattr(bt, "fetch_prices", lambda tk, frm, to, key=None: "PRICES")
    monkeypatch.setattr(
        bt, "simulate", lambda art, prices, include_after_hours=False: None
    )
    res = bt.evaluate(
        [_verdict(1, hour=16, minute=30), _verdict(2, hour=10)], key="k"
    )
    assert res[0]["skip_reason"] == "skipped_after_hours"   # published 16:30 ET
    assert res[1]["skip_reason"] == "skipped_no_prices"     # missing bars
    assert all(r["signed_return_pct"] is None for r in res)


def test_evaluate_fetch_failure_lands_in_skip_bucket(monkeypatch, capsys):
    def boom(tk, frm, to, key=None):
        raise RuntimeError("Massive request failed")

    monkeypatch.setattr(bt, "fetch_prices", boom)
    res = bt.evaluate([_verdict()], key="k")
    assert res[0]["skip_reason"] == "skipped_no_prices"
    assert "AAA" in capsys.readouterr().err


def test_evaluate_empty_input_needs_no_api_key():
    assert bt.evaluate([]) == []


# --------------------------------------------------------------------------- #
# CSV writing
# --------------------------------------------------------------------------- #
def test_write_csv_row_fields_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "fetch_prices", lambda tk, frm, to, key=None: "PRICES")
    monkeypatch.setattr(
        bt, "simulate",
        lambda art, prices, include_after_hours=False: _sim_row(2.5),
    )
    rows = bt.evaluate(
        [_verdict(1, action="sell"), _verdict(2, action="hold")], key="k"
    )
    out = tmp_path / "bt.csv"
    bt.write_csv(rows, str(out))

    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        got = list(reader)
        assert reader.fieldnames == bt.BACKTEST_FIELDS

    ok = got[0]
    assert ok["article_id"] == "1" and ok["ticker"] == "AAA"
    assert ok["action"] == "sell" and ok["confidence"] == "0.9"
    assert ok["published_et"] == "2025-01-06 10:00"
    assert ok["buy_price"] == "100.0" and ok["sell_close"] == "102.0"
    assert ok["gain_pct"] == "2.5" and ok["signed_return_pct"] == "-2.5"
    assert ok["skip_reason"] == ""
    assert ok["url"] == "https://x.test/1" and ok["title"] == "headline 1"

    skipped = got[1]
    assert skipped["skip_reason"] == "skipped_hold"
    assert skipped["gain_pct"] == "" and skipped["signed_return_pct"] == ""


# --------------------------------------------------------------------------- #
# load_verdicts row shaping
# --------------------------------------------------------------------------- #
def test_load_verdicts_rounds_float32_confidence():
    """article_sentiment.confidence is a Postgres real: 0.7 arrives as
    0.699999988 and must still land in the 0.7-0.85 bucket."""
    import struct

    f32_07 = struct.unpack("f", struct.pack("f", 0.7))[0]
    row = (1, "https://x.test/1", "nvda", "BUY", f32_07,
           datetime(2025, 1, 6, 15, 0, tzinfo=MARKET_TZ), "t")

    class _Cur:
        def execute(self, sql, params):
            pass

        def fetchall(self):
            return [row]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

    verdicts = bt.load_verdicts(_Conn(), "2025-01-01", "2025-01-31")
    assert verdicts[0]["confidence"] == 0.7
    s = summarize([{**verdicts[0], "signed_return_pct": 1.0, "skip_reason": ""}])
    assert s["by_confidence"]["0.7-0.85"]["n"] == 1


# --------------------------------------------------------------------------- #
# run_backtest orchestration (all stubbed)
# --------------------------------------------------------------------------- #
def test_run_backtest_writes_default_csv_and_returns_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr("ticker_news.shared.db.connect", lambda **kw: _FakeConn())
    monkeypatch.setattr(
        bt, "load_verdicts", lambda conn, start, end: [_verdict(1), _verdict(2, action="hold")]
    )
    monkeypatch.setattr(bt, "fetch_prices", lambda tk, frm, to, key=None: "PRICES")
    monkeypatch.setattr(
        bt, "simulate",
        lambda art, prices, include_after_hours=False: _sim_row(1.5),
    )
    summary = bt.run_backtest(start="2025-01-01", end="2025-01-31", key="k")

    assert (tmp_path / "backtest_verdicts_2025-01-01_2025-01-31.csv").exists()
    assert summary["total"] == 2
    assert summary["evaluated"] == 1
    assert summary["skipped"] == {"skipped_hold": 1}
    assert summary["overall"]["n"] == 1
    assert summary["overall"]["avg_return_pct"] == pytest.approx(1.5)
    out = capsys.readouterr().out
    assert "overall" in out and "skipped_hold" in out
