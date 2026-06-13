import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from ticker_news.research.candles import (
    CandleError,
    _require_charts,
    default_out,
    to_market_time,
)

ET = ZoneInfo("America/New_York")


# --- to_market_time -----------------------------------------------------------

def test_naive_string_assumed_et():
    dt = to_market_time("2025-01-03 10:30")
    assert dt.tzinfo is not None
    assert dt.utcoffset() == datetime(2025, 1, 3, 10, 30, tzinfo=ET).utcoffset()
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2025, 1, 3, 10, 30)


def test_iso_t_form_and_seconds():
    dt = to_market_time("2025-01-03T15:45:12")
    assert (dt.hour, dt.minute, dt.second) == (15, 45, 12)


def test_date_only_string():
    dt = to_market_time("2025-01-03")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2025, 1, 3, 0, 0)


def test_aware_string_converted_to_et():
    # January = EST (UTC-5): 15:30Z -> 10:30 ET.
    dt = to_market_time("2025-01-03T15:30:00Z")
    assert (dt.hour, dt.minute) == (10, 30)
    assert dt.utcoffset() == datetime(2025, 1, 3, tzinfo=ET).utcoffset()


def test_aware_string_with_offset():
    dt = to_market_time("2025-01-03 12:00:00+00:00")
    assert (dt.hour, dt.minute) == (7, 0)


def test_naive_string_with_custom_tz():
    # Chicago is one hour behind New York.
    dt = to_market_time("2025-01-03 09:30", tz="America/Chicago")
    assert (dt.hour, dt.minute) == (10, 30)


def test_aware_datetime_passthrough_converted():
    src = datetime(2025, 7, 1, 14, 0, tzinfo=timezone.utc)
    dt = to_market_time(src)
    # July = EDT (UTC-4): 14:00Z -> 10:00 ET.
    assert (dt.hour, dt.minute) == (10, 0)


def test_naive_datetime_assumed_tz():
    dt = to_market_time(datetime(2025, 1, 3, 10, 30))
    assert (dt.hour, dt.minute) == (10, 30)
    assert dt.tzinfo is not None


def test_unparseable_string_raises():
    with pytest.raises(CandleError, match="Could not parse timestamp"):
        to_market_time("not a timestamp")


# --- default_out --------------------------------------------------------------

def test_default_out_naming():
    ts = datetime(2025, 1, 3, 10, 30, tzinfo=ET)
    assert default_out("nvda", ts) == "NVDA_2025-01-03_1030.jpg"


def test_default_out_pads_hour_minute():
    ts = datetime(2025, 1, 3, 9, 5, tzinfo=ET)
    assert default_out("AAPL", ts) == "AAPL_2025-01-03_0905.jpg"


# --- _require_charts ----------------------------------------------------------

def test_require_charts_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mplfinance", None)  # forces ImportError
    with pytest.raises(RuntimeError, match=r"charts extra.*\[charts\]"):
        _require_charts()


def test_make_chart_fails_fast_without_charts(monkeypatch):
    from ticker_news.research.candles import make_chart

    monkeypatch.setitem(sys.modules, "mplfinance", None)
    # Guard fires before any API-key lookup or network call.
    with pytest.raises(RuntimeError, match="charts extra"):
        make_chart("NVDA", "2025-01-03 10:30")


# --- pandas-backed helpers (offline, no mplfinance needed) ---------------------

def test_bars_frame_sorts_and_renames():
    pd = pytest.importorskip("pandas")
    from ticker_news.research.candles import _bars_frame

    # Two bars out of order: 2025-01-03 09:31 and 09:30 ET (14:31/14:30 UTC).
    bars = [
        {"t": 1735914660000, "o": 2.0, "h": 2.5, "l": 1.5, "c": 2.2, "v": 200},
        {"t": 1735914600000, "o": 1.0, "h": 1.5, "l": 0.5, "c": 1.2, "v": 100},
    ]
    df = _bars_frame(bars, "NVDA", "2025-01-03")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.is_monotonic_increasing
    assert str(df.index.tz) == "America/New_York"
    assert df.iloc[0].Open == 1.0 and df.iloc[1].Open == 2.0


def test_bars_frame_empty_raises():
    pytest.importorskip("pandas")
    from ticker_news.research.candles import _bars_frame

    with pytest.raises(CandleError, match="No bars returned"):
        _bars_frame([], "NVDA", "2025-01-03")


def test_locate_candle_exact_and_nearest(capsys):
    pytest.importorskip("pandas")
    from ticker_news.research.candles import _bars_frame, locate_candle

    base = 1735914600000  # 2025-01-03 09:30 ET
    bars = [
        {"t": base + i * 60_000, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}
        for i in range(3)
    ]
    df = _bars_frame(bars, "NVDA", "2025-01-03")
    inside = datetime(2025, 1, 3, 9, 31, 30, tzinfo=ET)  # within the 09:31 bar
    assert locate_candle(df, inside, 1) == 1
    after = datetime(2025, 1, 3, 9, 40, tzinfo=ET)  # past the last bar -> nearest
    assert locate_candle(df, after, 1) == 2
