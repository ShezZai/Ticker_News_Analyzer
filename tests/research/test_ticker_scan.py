"""Offline tests for the ticker-scan suite (pure logic + stubbed orchestration)."""

import csv
import json
from datetime import date, datetime

from ticker_news.research import ticker_scan as ts
from ticker_news.research.market_data import MARKET_TZ


def _ms(y, mo, d, h, mi=0):
    """Epoch milliseconds for an ET wall-clock instant."""
    return int(datetime(y, mo, d, h, mi, tzinfo=MARKET_TZ).timestamp() * 1000)


def _bar(t, o, h, l, c):  # noqa: E741 - mirrors Massive's field names
    return {"t": t, "o": o, "h": h, "l": l, "c": c}


# --------------------------------------------------------------------------- #
# daily_ranges
# --------------------------------------------------------------------------- #
def test_daily_ranges_collapses_per_et_date():
    bars = [
        # Mon 2025-01-06: low (99) prints before high (106) -> "up";
        # the 23:30 ET bar is 04:30 UTC on Jan 7 — it must still land on Jan 6.
        _bar(_ms(2025, 1, 6, 9, 0), 100.0, 101.0, 99.0, 100.5),
        _bar(_ms(2025, 1, 6, 10, 0), 100.5, 106.0, 100.0, 105.0),
        _bar(_ms(2025, 1, 6, 23, 30), 105.0, 105.5, 104.0, 104.5),
        # Tue 2025-01-07: high (112) prints before low (103) -> "down".
        _bar(_ms(2025, 1, 7, 9, 0), 110.0, 112.0, 108.0, 109.0),
        _bar(_ms(2025, 1, 7, 10, 0), 109.0, 110.0, 103.0, 104.0),
    ]
    days = ts.daily_ranges(bars)
    assert set(days) == {date(2025, 1, 6), date(2025, 1, 7)}

    d1 = days[date(2025, 1, 6)]
    assert d1.low == 99.0 and d1.high == 106.0
    assert d1.open == 100.0           # first bar of the ET date
    assert d1.close == 104.5          # last bar — the 23:30 ET one
    assert d1.range_pct() == (106.0 - 99.0) / 99.0 * 100.0
    assert d1.net_pct() == (104.5 - 100.0) / 100.0 * 100.0
    assert d1.direction() == "up"

    d2 = days[date(2025, 1, 7)]
    assert d2.low == 103.0 and d2.high == 112.0
    assert d2.direction() == "down"


def test_daily_ranges_single_bar_is_up():
    days = ts.daily_ranges([_bar(_ms(2025, 1, 6, 9, 0), 10.0, 11.0, 9.0, 10.5)])
    assert days[date(2025, 1, 6)].direction() == "up"


# --------------------------------------------------------------------------- #
# TickerPrices / simulate
# --------------------------------------------------------------------------- #
def _prices():
    minute = [
        _bar(_ms(2025, 1, 6, 8, 30), 100.0, 0, 0, 0),
        _bar(_ms(2025, 1, 6, 8, 31), 100.5, 0, 0, 0),
        _bar(_ms(2025, 1, 6, 9, 30), 101.0, 0, 0, 0),
        _bar(_ms(2025, 1, 6, 10, 0), 102.0, 0, 0, 0),
        _bar(_ms(2025, 1, 6, 17, 0), 104.0, 0, 0, 0),
        _bar(_ms(2025, 1, 7, 4, 0), 106.0, 0, 0, 0),
    ]
    daily = [
        _bar(_ms(2025, 1, 6, 12, 0), 0, 0, 0, 105.5),
        _bar(_ms(2025, 1, 7, 12, 0), 0, 0, 0, 110.0),
    ]
    return ts.TickerPrices(minute, daily)


def _article(pub_et, aid=1, ticker="TEST"):
    return {"id": aid, "ticker": ticker, "category": "real news",
            "published_et": pub_et, "title": "headline"}


def test_simulate_premarket_entry_same_day_close():
    pub = datetime(2025, 1, 6, 8, 30, 30, tzinfo=MARKET_TZ)  # seconds get floored
    row = ts.simulate(_article(pub), _prices())
    assert row is not None
    assert row["buy_et"] == "2025-01-06 08:30"
    assert row["buy_price"] == 100.0
    assert row["entry_session"] == "premarket"
    assert row["hold_until"] == "same-day-close"
    assert row["sell_date"] == "2025-01-06"
    assert row["sell_close"] == 105.5
    assert row["gain_pct"] == round((105.5 - 100.0) / 100.0 * 100.0, 2)  # 5.5


def test_simulate_regular_entry_same_day_close():
    pub = datetime(2025, 1, 6, 9, 45, tzinfo=MARKET_TZ)
    row = ts.simulate(_article(pub), _prices())
    assert row["buy_et"] == "2025-01-06 10:00"   # first minute bar >= ts
    assert row["buy_price"] == 102.0
    assert row["entry_session"] == "regular"
    assert row["sell_date"] == "2025-01-06"
    assert row["gain_pct"] == round((105.5 - 102.0) / 102.0 * 100.0, 2)  # 3.43


def test_simulate_after_hours_excluded_by_default():
    pub = datetime(2025, 1, 6, 16, 30, tzinfo=MARKET_TZ)
    assert ts.simulate(_article(pub), _prices()) is None


def test_simulate_after_hours_included_sells_next_day():
    pub = datetime(2025, 1, 6, 16, 30, tzinfo=MARKET_TZ)
    row = ts.simulate(_article(pub), _prices(), include_after_hours=True)
    assert row["buy_et"] == "2025-01-06 17:00"
    assert row["buy_price"] == 104.0
    assert row["entry_session"] == "after_hours"   # new label, not "after-hours"
    assert row["hold_until"] == "next-day-close"
    assert row["sell_date"] == "2025-01-07"
    assert row["sell_close"] == 110.0
    assert row["gain_pct"] == round((110.0 - 104.0) / 104.0 * 100.0, 2)  # 5.77


def test_simulate_closed_publication_enters_next_premarket():
    # Published 21:00 ET ("closed"): no bars left that evening, so the entry is
    # the next session's first minute bar — gated by THAT bar's time, hence a
    # plain same-day exit that survives the default after-hours exclusion.
    pub = datetime(2025, 1, 6, 21, 0, tzinfo=MARKET_TZ)
    row = ts.simulate(_article(pub), _prices())
    assert row["buy_et"] == "2025-01-07 04:00"
    assert row["buy_price"] == 106.0
    assert row["entry_session"] == "premarket"
    assert row["hold_until"] == "same-day-close"
    assert row["sell_date"] == "2025-01-07"
    assert row["gain_pct"] == round((110.0 - 106.0) / 106.0 * 100.0, 2)  # 3.77


def test_simulate_no_bars_after_publication_returns_none():
    pub = datetime(2025, 1, 8, 9, 0, tzinfo=MARKET_TZ)
    assert ts.simulate(_article(pub), _prices(), include_after_hours=True) is None


def test_next_trading_day_skips_weekend():
    daily = [
        _bar(_ms(2025, 1, 10, 12, 0), 0, 0, 0, 111.0),  # Friday
        _bar(_ms(2025, 1, 13, 12, 0), 0, 0, 0, 112.0),  # Monday
    ]
    tp = ts.TickerPrices([], daily)
    assert tp.next_trading_day(date(2025, 1, 10)) == date(2025, 1, 13)
    assert tp.next_trading_day(date(2025, 1, 11)) == date(2025, 1, 13)
    assert tp.next_trading_day(date(2025, 1, 13)) is None


# --------------------------------------------------------------------------- #
# attach: pure article-window math
# --------------------------------------------------------------------------- #
def test_index_articles_same_day_and_prev_after_hours():
    records = [
        # regular-hours article: same-day only, under both in-scope tickers
        (1, date(2025, 1, 6), 10, "2025-01-06 10:00", "A", "u1", "NVDA", ["AMD", "TSLA"]),
        # 17:00 ET article: same-day AND next-day "prev-after-hours"
        (2, date(2025, 1, 6), 17, "2025-01-06 17:00", "B", "u2", "NVDA", None),
        # out-of-scope ticker: ignored entirely
        (3, date(2025, 1, 6), 11, "2025-01-06 11:00", "C", "u3", "TSLA", []),
    ]
    out = ts.index_articles(records, {"NVDA", "AMD"})

    jan6_nvda = out[("NVDA", date(2025, 1, 6))]
    assert [(a["id"], a["when"]) for a in jan6_nvda] == [(1, "same-day"), (2, "same-day")]
    assert [(a["id"], a["when"]) for a in out[("AMD", date(2025, 1, 6))]] == [(1, "same-day")]
    assert [(a["id"], a["when"]) for a in out[("NVDA", date(2025, 1, 7))]] == [
        (2, "prev-after-hours")
    ]
    assert ("TSLA", date(2025, 1, 6)) not in out


def test_attach_shapes_csv_rows(tmp_path, monkeypatch):
    src = tmp_path / "scan.csv"
    dst = tmp_path / "out.csv"
    with open(src, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "ticker", "range_pct"])
        w.writeheader()
        w.writerow({"date": "2025-01-06", "ticker": "AAA", "range_pct": "7.0"})
        w.writerow({"date": "2025-01-07", "ticker": "AAA", "range_pct": "6.0"})

    arts = {("AAA", date(2025, 1, 6)): [
        {"id": 9, "published_et": "2025-01-06 10:00", "title": "T", "url": "u",
         "when": "same-day"},
    ]}
    monkeypatch.setattr(ts, "fetch_articles_by_ticker_day",
                        lambda tickers, start, end: arts)

    assert ts.attach(str(src), str(dst)) == 2

    with open(dst, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["article_count"] == "1"
    assert json.loads(rows[0]["articles"])[0]["id"] == 9
    assert rows[1]["article_count"] == "0"
    assert json.loads(rows[1]["articles"]) == []
    assert rows[0]["range_pct"] == "7.0"  # original columns preserved


def test_attach_rejects_missing_columns(tmp_path):
    src = tmp_path / "bad.csv"
    src.write_text("foo,bar\n1,2\n")
    try:
        ts.attach(str(src), str(tmp_path / "out.csv"))
    except ts.ScanError as exc:
        assert "date" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ScanError")


# --------------------------------------------------------------------------- #
# scan orchestration with stubbed fetches
# --------------------------------------------------------------------------- #
def test_scan_flags_days_and_filters_by_calm_nasdaq(monkeypatch, capsys):
    def fake_fetch_bars(ticker, *, span, frm, to, key=None, **kw):
        if ticker == "I:COMP":
            assert span == "day"
            return [
                _bar(_ms(2025, 1, 6, 12, 0), 100.0, 100.5, 100.0, 100.2),  # 0.5% calm
                _bar(_ms(2025, 1, 7, 12, 0), 100.0, 102.0, 100.0, 101.0),  # 2.0% volatile
            ]
        assert ticker == "AAA" and span == "hour"
        return [
            _bar(_ms(2025, 1, 6, 10, 0), 100.0, 110.0, 100.0, 108.0),  # 10% range
            _bar(_ms(2025, 1, 7, 10, 0), 100.0, 110.0, 100.0, 108.0),  # 10% range
            _bar(_ms(2025, 1, 8, 10, 0), 100.0, 101.0, 100.0, 100.5),  # 1% — not flagged
        ]

    monkeypatch.setattr(ts, "fetch_bars", fake_fetch_bars)
    rows = ts.scan(["AAA"], start="2025-01-06", end="2025-01-08", key="k",
                   segment_map={"AAA": "GPUs"})

    # Jan 7 flagged but dropped (NASDAQ 2.0% >= 1.2), Jan 8 never flagged.
    assert [r["date"] for r in rows] == ["2025-01-06"]
    r = rows[0]
    assert r["ticker"] == "AAA" and r["segment"] == "GPUs"
    assert r["range_pct"] == 10.0 and r["direction"] == "up"
    assert r["nasdaq_range_pct"] == 0.5


def test_scan_drops_days_without_index_data(monkeypatch):
    def fake_fetch_bars(ticker, *, span, frm, to, key=None, **kw):
        if ticker == "I:COMP":
            return []  # no index data at all
        return [_bar(_ms(2025, 1, 6, 10, 0), 100.0, 110.0, 100.0, 108.0)]

    monkeypatch.setattr(ts, "fetch_bars", fake_fetch_bars)
    rows = ts.scan(["AAA"], start="2025-01-06", end="2025-01-06", key="k")
    assert rows == []


def test_write_scan_csv(tmp_path):
    rows = [{
        "date": "2025-01-06", "ticker": "AAA", "segment": "GPUs", "low": 100.0,
        "high": 110.0, "range_pct": 10.0, "open": 100.0, "close": 108.0,
        "net_pct": 8.0, "direction": "up", "nasdaq_low": 100.0,
        "nasdaq_high": 100.5, "nasdaq_range_pct": 0.5,
    }]
    out = tmp_path / "scan.csv"
    ts.write_scan_csv(rows, str(out))
    with open(out, newline="") as fh:
        got = list(csv.DictReader(fh))
    assert got[0]["ticker"] == "AAA" and got[0]["nasdaq_range_pct"] == "0.5"
    assert list(got[0]) == ts.SCAN_FIELDS
