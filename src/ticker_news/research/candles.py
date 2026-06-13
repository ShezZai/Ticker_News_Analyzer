"""Render an intraday OHLC candle chart for one ticker/day from the Massive API.

Given a ticker and a timestamp, fetch every 1-minute bar for that timestamp's
*entire trading day* — premarket, regular session, and after-hours — and write
a JPG candlestick chart with the timestamp's own candle marked vividly (gold
band + magenta star + label), dark theme.

Timestamps without a timezone are interpreted as US market time
(America/New_York by default). Timestamps with an explicit offset / trailing
'Z' are converted to market time before selecting the day and candle.

pandas / mplfinance / matplotlib live in the ``charts`` optional extra and are
imported lazily — importing this module (and calling :func:`to_market_time` /
:func:`default_out`) needs none of them.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from ticker_news.research.market_data import (
    MARKET_TZ,
    REGULAR_CLOSE,
    REGULAR_OPEN,
    api_key,
    fetch_bars,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class CandleError(RuntimeError):
    pass


def _require_charts() -> None:
    try:
        import mplfinance  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "chart rendering needs the charts extra: pip install -e \".[charts]\""
        ) from exc


def to_market_time(ts: str | datetime, tz: str = "America/New_York") -> datetime:
    """Parse `ts` into a tz-aware datetime in market time (America/New_York).

    Accepts 'YYYY-MM-DD HH:MM[:SS]', ISO 'T' form, trailing 'Z', or a
    ``datetime``. Naive inputs are assumed to be in `tz`; aware inputs are
    converted. The result is always expressed in :data:`MARKET_TZ`.
    """
    zone = ZoneInfo(tz)
    if isinstance(ts, datetime):
        dt: datetime | None = ts
    else:
        txt = ts.strip().replace("Z", "+00:00")
        dt = None
        try:
            dt = datetime.fromisoformat(txt)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(txt, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            raise CandleError(
                f"Could not parse timestamp {ts!r}. "
                "Use e.g. '2025-01-03 10:30' or '2025-01-03T10:30:00'."
            )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=zone)
    return dt.astimezone(MARKET_TZ)


def default_out(ticker: str, ts_market: datetime) -> str:
    """Default output path: TICKER_DATE_HHMM.jpg (market-time fields)."""
    return f"{ticker.upper()}_{ts_market:%Y-%m-%d}_{ts_market:%H%M}.jpg"


def _bars_frame(bars: list[dict], ticker: str, day: str) -> "pd.DataFrame":
    """Raw aggregate bars -> OHLCV DataFrame indexed by market-time datetimes."""
    import pandas as pd

    if not bars:
        raise CandleError(
            f"No bars returned for {ticker.upper()} on {day}. "
            "Market holiday or no data?"
        )
    df = pd.DataFrame(bars)
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(MARKET_TZ)
    df = df.set_index("dt").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def locate_candle(df: "pd.DataFrame", ts: datetime, mult: int) -> int:
    """Return the integer row position of the bar covering `ts`.

    A bar labelled T covers [T, T+interval). If `ts` falls in a gap or outside
    the day's data, clamp to the nearest bar and warn on stderr.
    """
    import pandas as pd

    interval = timedelta(minutes=mult)
    starts = df.index
    for i, start in enumerate(starts):
        if start <= ts < start + interval:
            return i
    pos = int(starts.get_indexer([pd.Timestamp(ts)], method="nearest")[0])
    chosen = starts[pos]
    print(
        f"  note: {ts:%H:%M:%S} has no exact bar; marking nearest at "
        f"{chosen:%H:%M:%S} ET.",
        file=sys.stderr,
    )
    return pos


def _session_spans(
    df: "pd.DataFrame",
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Integer (start,end) positions for premarket and after-hours rows."""
    times = df.index.time
    pre = [i for i, t in enumerate(times) if t < REGULAR_OPEN]
    post = [i for i, t in enumerate(times) if t >= REGULAR_CLOSE]
    pre_span = (pre[0], pre[-1]) if pre else None
    post_span = (post[0], post[-1]) if post else None
    return pre_span, post_span


def _render(df: "pd.DataFrame", target_pos: int, ticker: str, day: str, output: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless / file output only
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    target_bar = df.iloc[target_pos]
    title = (
        f"{ticker.upper()}  {day} (ET)  —  marked candle {df.index[target_pos]:%H:%M}  "
        f"O:{target_bar.Open:g} H:{target_bar.High:g} L:{target_bar.Low:g} C:{target_bar.Close:g}"
    )

    mc = mpf.make_marketcolors(
        up="#26a269", down="#e01b24", edge="inherit",
        wick="inherit", volume="#7d8499",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds", marketcolors=mc, gridcolor="#2a2a35",
        facecolor="#16161e", figcolor="#16161e",
    )

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        returnfig=True,
        figsize=(16, 9),
        title=title,
        tight_layout=True,
        warn_too_much_data=len(df) + 1,
    )
    price_ax, vol_ax = axes[0], axes[2]

    # Shade premarket / after-hours so the regular session stands out.
    pre_span, post_span = _session_spans(df)
    for span, label in ((pre_span, "pre-market"), (post_span, "after-hours")):
        if span:
            for ax in (price_ax, vol_ax):
                ax.axvspan(span[0] - 0.5, span[1] + 0.5, color="#3b3b52", alpha=0.35, zorder=0)
            price_ax.text(
                (span[0] + span[1]) / 2, price_ax.get_ylim()[1], label,
                ha="center", va="top", fontsize=9, color="#b0b0c0", alpha=0.8,
            )

    # Mark the target candle vividly: gold vertical band + magenta star + label.
    price_ax.axvspan(target_pos - 0.5, target_pos + 0.5, color="#ffd700", alpha=0.18, zorder=1)
    price_ax.axvline(target_pos, color="#ff00ff", linewidth=1.1, alpha=0.7, zorder=5)
    vol_ax.axvline(target_pos, color="#ff00ff", linewidth=1.1, alpha=0.7, zorder=5)
    headroom = (df["High"].max() - df["Low"].min()) * 0.04
    price_ax.scatter(
        [target_pos], [target_bar.High + headroom],
        marker="*", s=320, color="#ffd700", edgecolors="#ff00ff",
        linewidths=1.2, zorder=6,
    )
    price_ax.annotate(
        f"{df.index[target_pos]:%H:%M}",
        xy=(target_pos, target_bar.High + headroom),
        xytext=(0, 12), textcoords="offset points",
        ha="center", color="#ffd700", fontsize=11, fontweight="bold", zorder=6,
    )

    fig.savefig(output, dpi=120, format="jpg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_chart(
    ticker: str,
    ts: datetime | str,
    *,
    out: str | None = None,
    interval: int = 1,
    tz: str = "America/New_York",
    key: str | None = None,
) -> str:
    """Fetch the day's bars and write a marked candlestick JPG. Returns the path."""
    _require_charts()
    k = api_key(key)
    ts_market = to_market_time(ts, tz)
    day = ts_market.strftime("%Y-%m-%d")
    output = out or default_out(ticker, ts_market)

    bars = fetch_bars(
        ticker.upper(), span="minute", multiplier=interval, frm=day, to=day, key=k
    )
    df = _bars_frame(bars, ticker, day)
    target_pos = locate_candle(df, ts_market, interval)
    _render(df, target_pos, ticker, day, output)
    return output
