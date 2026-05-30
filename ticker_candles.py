#!/usr/bin/env python3
"""Render an intraday OHLC candle chart for one ticker/day from the Massive API.

Given a ticker and a datetime timestamp, this fetches every 1-minute bar for
that timestamp's *entire trading day* — premarket, regular session, and
after-hours — and writes a JPG candlestick chart with the timestamp's own
candle marked vividly.

Usage:
    python ticker_candles.py AAPL "2025-01-03 10:30"
    python ticker_candles.py NVDA 2025-01-03T15:45:00 -o nvda.jpg
    python ticker_candles.py TSLA "2025-01-03 08:15" --interval 5 --tz America/New_York

Timestamps without a timezone are interpreted as US market time
(America/New_York). Timestamps with an explicit offset / trailing 'Z' are
converted to market time before selecting the day and candle.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")  # headless / file output only
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

MARKET_TZ = ZoneInfo("America/New_York")
AGGS_URL = "https://api.massive.com/v2/aggs/ticker/{ticker}/range/{mult}/minute/{day}/{day}"
REQUEST_TIMEOUT = 30

# Regular trading session in ET.
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


class CandleError(RuntimeError):
    pass


def _api_key(explicit: str | None) -> str:
    key = explicit or os.getenv("MASSIVE_API_KEY")
    if not key:
        raise CandleError("No API key. Pass --api-key or set MASSIVE_API_KEY in .env")
    return key


def parse_timestamp(raw: str, tz: ZoneInfo) -> datetime:
    """Parse a user timestamp into a tz-aware datetime in market time.

    Accepts 'YYYY-MM-DD HH:MM[:SS]', ISO 'T' form, and trailing 'Z'.
    Naive inputs are assumed to be in `tz`; aware inputs are converted to it.
    """
    txt = raw.strip().replace("Z", "+00:00")
    dt: datetime | None = None
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
            f"Could not parse timestamp {raw!r}. "
            "Use e.g. '2025-01-03 10:30' or '2025-01-03T10:30:00'."
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def fetch_bars(ticker: str, day: str, mult: int, api_key: str) -> pd.DataFrame:
    """Fetch all 1-minute (×mult) bars for `day` including extended hours."""
    url = AGGS_URL.format(ticker=ticker.upper(), mult=mult, day=day)
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results") or []
    if not results:
        raise CandleError(
            f"No bars returned for {ticker.upper()} on {day} "
            f"(status={payload.get('status')}). Market holiday or no data?"
        )
    df = pd.DataFrame(results)
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(MARKET_TZ)
    df = df.set_index("dt").sort_index()
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def locate_candle(df: pd.DataFrame, ts: datetime, mult: int) -> int:
    """Return the integer row position of the bar covering `ts`.

    A bar labelled T covers [T, T+interval). If `ts` predates the first bar or
    follows the last, clamp to the nearest end and warn.
    """
    interval = timedelta(minutes=mult)
    starts = df.index
    for i, start in enumerate(starts):
        if start <= ts < start + interval:
            return i
    # Not inside any bar (e.g. a gap or outside the day's data) -> nearest.
    pos = int(starts.get_indexer([pd.Timestamp(ts)], method="nearest")[0])
    chosen = starts[pos]
    print(
        f"  note: {ts:%H:%M:%S} has no exact bar; marking nearest at "
        f"{chosen:%H:%M:%S} ET.",
        file=sys.stderr,
    )
    return pos


def _session_spans(df: pd.DataFrame) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Integer (start,end) positions for premarket and after-hours rows."""
    times = df.index.time
    pre = [i for i, t in enumerate(times) if t < REGULAR_OPEN]
    post = [i for i, t in enumerate(times) if t >= REGULAR_CLOSE]
    pre_span = (pre[0], pre[-1]) if pre else None
    post_span = (post[0], post[-1]) if post else None
    return pre_span, post_span


def render(
    df: pd.DataFrame,
    target_pos: int,
    ticker: str,
    day: str,
    ts: datetime,
    output: str,
) -> None:
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
    timestamp: str,
    output: str | None = None,
    interval: int = 1,
    tz_name: str = "America/New_York",
    api_key: str | None = None,
) -> str:
    """Fetch the day's bars and write a marked candlestick JPG. Returns the path."""
    key = _api_key(api_key)
    tz = ZoneInfo(tz_name)
    ts = parse_timestamp(timestamp, tz).astimezone(MARKET_TZ)
    day = ts.strftime("%Y-%m-%d")
    out = output or f"{ticker.upper()}_{day}_{ts:%H%M}.jpg"

    df = fetch_bars(ticker, day, interval, key)
    target_pos = locate_candle(df, ts, interval)
    render(df, target_pos, ticker, day, ts, out)
    print(f"Wrote {out}  ({len(df)} bars, {df.index[0]:%H:%M}–{df.index[-1]:%H:%M} ET)")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    p.add_argument("timestamp", help="Datetime, e.g. '2025-01-03 10:30' or 2025-01-03T10:30:00Z")
    p.add_argument("-o", "--output", help="Output JPG path (default TICKER_DATE_HHMM.jpg)")
    p.add_argument("--interval", type=int, default=1, help="Candle size in minutes (default 1)")
    p.add_argument("--tz", default="America/New_York", help="Timezone for naive timestamps")
    p.add_argument("--api-key", help="Override MASSIVE_API_KEY")
    args = p.parse_args(argv)
    try:
        make_chart(args.ticker, args.timestamp, args.output, args.interval, args.tz, args.api_key)
    except CandleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except requests.HTTPError as exc:
        print(f"error: API request failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
