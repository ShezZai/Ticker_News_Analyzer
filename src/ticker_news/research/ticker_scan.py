"""Ticker-scan research suite: big-range days, article attach, catalyst returns.

Consolidates the legacy ticker-scan ``{scan_ranges,attach_articles,
catalyst_returns}.py`` scripts into one module. All Massive HTTP goes through
:mod:`ticker_news.research.market_data`; DB access goes through
:func:`ticker_news.shared.db.connect`.

Three tools:

* :func:`scan` — flag days whose low-to-high intraday range (extended hours
  included) meets a threshold, on days where the NASDAQ itself stayed calm.
* :func:`attach` — enrich a scan CSV with each (ticker, day)'s articles:
  same ET date plus the previous day's post-16:00-ET ("prev-after-hours") news.
* :func:`catalyst_run` / :func:`simulate` — buy at an article's publication
  minute (extended-hours minute bar), sell at the next regular close.

Conscious deviations from legacy:

* Catalyst ``entry_session`` labels come from ``market_data.session_of``:
  ``premarket`` / ``regular`` / ``after_hours`` / ``closed`` instead of the
  legacy hyphenated ``pre-market`` / ``after-hours`` (legacy also labelled
  everything >= 16:00 ET "after-hours"; >= 20:00 ET is now "closed").
* :func:`simulate` gates the exit off the *entry bar's* clock time exactly as
  legacy did: an entry bar before 16:00 ET sells at that day's close; at/after
  16:00 ET it is an extended-hours entry (excluded unless
  ``include_after_hours``) selling at the next trading day's close. An article
  published at/after 20:00 ET ("closed") has no bars left that evening, so its
  entry is the next session's first minute bar (typically premarket) and is
  gated by *that* bar's time — i.e. "closed" publications are treated like
  after-hours ones for entry purposes, the legacy behavior, kept on purpose.
* Errors are raised as :class:`ScanError` (the legacy attach script used bare
  ``SystemExit``); the CLI converts them to clean exits.
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Sequence

from ticker_news.research.market_data import (
    MARKET_TZ,
    REGULAR_CLOSE,
    api_key,
    fetch_bars,
    session_of,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

DEFAULT_THRESHOLD = 5.0
DEFAULT_INDEX = "I:COMP"  # NASDAQ Composite
DEFAULT_NAS_THRESHOLD = 1.2
# The news dataset starts here; default the scan to the same window.
DEFAULT_SCAN_START = "2024-11-01"

DEFAULT_CATALYST_START = "2025-02-01"
DEFAULT_CATALYST_END = "2025-03-30"
# "news, class action, etc" -> the categories that can actually move a stock.
DEFAULT_CATEGORIES = ["real news", "legal solicitation", "regulatory filing"]
# Buy can roll into the next session and an after-hours buy sells the next day,
# so fetch prices a little past the article window.
PRICE_TAIL_DAYS = 7

# Regular session closes at 16:00 ET; anything at/after this is extended hours.
AFTER_HOURS_HOUR = REGULAR_CLOSE.hour

SCAN_FIELDS = [
    "date", "ticker", "segment", "low", "high", "range_pct", "open", "close",
    "net_pct", "direction", "nasdaq_low", "nasdaq_high", "nasdaq_range_pct",
]
CATALYST_FIELDS = [
    "article_id", "ticker", "ticker_role", "category", "published_et",
    "entry_session", "buy_et", "buy_price", "hold_until", "sell_date",
    "sell_close", "gain_pct", "title",
]


class ScanError(RuntimeError):
    pass


def _bar_dt(bar: dict) -> datetime:
    return datetime.fromtimestamp(bar["t"] / 1000, MARKET_TZ)


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def load_universe() -> tuple[list[str], dict[str, str]]:
    """Return (tickers, segment_map) from public.ticker_data."""
    from ticker_news.shared import db

    try:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT ticker, segment FROM public.ticker_data ORDER BY ticker")
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
        raise ScanError(
            f"Could not read the default ticker universe from the database "
            f"({exc}).\nPass --tickers AAPL,NVDA,... or set DATABASE_URL."
        ) from exc
    tickers = [t.strip().upper() for t, _ in rows if t and t.strip()]
    if not tickers:
        raise ScanError("public.ticker_data is empty; populate it or pass --tickers.")
    segment_map = {t.strip().upper(): (seg or "") for t, seg in rows if t and t.strip()}
    return tickers, segment_map


# --------------------------------------------------------------------------- #
# Range scan (scan_ranges port)
# --------------------------------------------------------------------------- #
class DayRange:
    __slots__ = ("low", "high", "open", "close", "low_dt", "high_dt", "first_dt", "last_dt")

    def __init__(self, dt: datetime, o: float, h: float, l: float, c: float):  # noqa: E741
        self.low = l
        self.high = h
        self.open = o
        self.close = c
        self.low_dt = dt
        self.high_dt = dt
        self.first_dt = dt
        self.last_dt = dt

    def update(self, dt: datetime, o: float, h: float, l: float, c: float) -> None:  # noqa: E741
        if l < self.low:
            self.low, self.low_dt = l, dt
        if h > self.high:
            self.high, self.high_dt = h, dt
        if dt < self.first_dt:
            self.first_dt, self.open = dt, o
        if dt > self.last_dt:
            self.last_dt, self.close = dt, c

    def range_pct(self) -> Optional[float]:
        if self.low <= 0:
            return None
        return (self.high - self.low) / self.low * 100.0

    def net_pct(self) -> Optional[float]:
        if self.open <= 0:
            return None
        return (self.close - self.open) / self.open * 100.0

    def direction(self) -> str:
        # low printed before high -> the swing ran up; else down.
        return "up" if self.low_dt <= self.high_dt else "down"


def daily_ranges(bars: list[dict]) -> dict[date, DayRange]:
    """Collapse raw aggregate bars into one DayRange per ET calendar date."""
    days: dict[date, DayRange] = {}
    for b in bars:
        dt = _bar_dt(b)
        d = dt.date()
        if d in days:
            days[d].update(dt, b["o"], b["h"], b["l"], b["c"])
        else:
            days[d] = DayRange(dt, b["o"], b["h"], b["l"], b["c"])
    return days


def scan_ticker(
    ticker: str, bar: str, start: str, end: str, threshold: float, key: str
) -> list[dict]:
    """Return one dict per flagged day for `ticker` (range_pct >= threshold)."""
    bars = fetch_bars(ticker, span=bar, frm=start, to=end, key=key)
    flagged: list[dict] = []
    for d, dr in daily_ranges(bars).items():
        rng = dr.range_pct()
        if rng is None or rng < threshold:
            continue
        net = dr.net_pct()
        flagged.append(
            {
                "date": d.isoformat(),
                "ticker": ticker,
                "low": round(dr.low, 4),
                "high": round(dr.high, 4),
                "range_pct": round(rng, 2),
                "open": round(dr.open, 4),
                "close": round(dr.close, 4),
                "net_pct": round(net, 2) if net is not None else "",
                "direction": dr.direction(),
            }
        )
    return flagged


def index_ranges(
    index_ticker: str, dates: Sequence[date], key: str
) -> dict[date, tuple[float, float, float]]:
    """Daily (low, high, range_pct) for the index across the span of `dates`.

    One daily-bar request covers the whole window; only the dates actually
    flagged are returned.
    """
    if not dates:
        return {}
    frm, to = min(dates).isoformat(), max(dates).isoformat()
    wanted = set(dates)
    out: dict[date, tuple[float, float, float]] = {}
    try:
        bars = fetch_bars(index_ticker, span="day", frm=frm, to=to, key=key)
    except RuntimeError as exc:
        print(f"  warning: could not fetch index {index_ticker}: {exc}", file=sys.stderr)
        return {}
    for b in bars:
        d = _bar_dt(b).date()
        if d not in wanted:
            continue
        low, high = b["l"], b["h"]
        rng = (high - low) / low * 100.0 if low > 0 else None
        out[d] = (round(low, 4), round(high, 4), round(rng, 2) if rng is not None else "")
    return out


def scan(
    tickers: Sequence[str],
    *,
    start: str,
    end: str,
    threshold: float = DEFAULT_THRESHOLD,
    bar: str = "hour",
    index_ticker: str = DEFAULT_INDEX,
    nas_threshold: float = DEFAULT_NAS_THRESHOLD,
    workers: int = 8,
    key: str | None = None,
    segment_map: dict[str, str] | None = None,
) -> list[dict]:
    """Scan all tickers for flagged days, attach the index reference, filter.

    Returns the flagged rows (sorted by date then ticker) that also pass the
    calm-NASDAQ filter (``nasdaq_range_pct < nas_threshold``); rows whose date
    has no index data are dropped with the filter, like legacy.
    """
    k = api_key(key)
    print(
        f"Scanning {len(tickers)} ticker(s) {start}..{end} for >= {threshold:g}% "
        f"low-to-high days ({bar} bars, extended hours included) ..."
    )
    rows: list[dict] = []
    n_failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scan_ticker, tk, bar, start, end, threshold, k): tk
            for tk in tickers
        }
        completed = as_completed(futures)
        pbar = tqdm(completed, total=len(futures), unit="ticker") if tqdm else completed
        for fut in pbar:
            tk = futures[fut]
            try:
                rows.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                print(f"  {tk}: {exc}", file=sys.stderr)
            if tqdm:
                pbar.set_postfix_str(f"{len(rows)} flagged | {n_failed} failed")
    if n_failed:
        print(f"  {n_failed} ticker(s) failed", file=sys.stderr)

    # Attach the NASDAQ index reference for every flagged date.
    flagged_dates = sorted({date.fromisoformat(r["date"]) for r in rows})
    print(f"Fetching {index_ticker} reference for {len(flagged_dates)} date(s) ...")
    idx = index_ranges(index_ticker, flagged_dates, k)
    seg = segment_map or {}
    for r in rows:
        d = date.fromisoformat(r["date"])
        lo, hi, rng = idx.get(d, ("", "", ""))
        r["nasdaq_low"], r["nasdaq_high"], r["nasdaq_range_pct"] = lo, hi, rng
        r["segment"] = seg.get(r["ticker"], "")

    # Filter out days where the NASDAQ moved more than nas_threshold.
    before = len(rows)
    rows = [
        r for r in rows
        if isinstance(r["nasdaq_range_pct"], (int, float))
        and r["nasdaq_range_pct"] < nas_threshold
    ]
    if len(rows) < before:
        print(
            f"  nas_t filter (< {nas_threshold:g}%): dropped {before - len(rows)} row(s), "
            f"{len(rows)} remaining"
        )

    rows.sort(key=lambda r: (r["date"], r["ticker"]))
    return rows


def write_scan_csv(rows: Sequence[dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Article attach (attach_articles port)
# --------------------------------------------------------------------------- #
def _read_scan_rows(path: str) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "date" not in fields or "ticker" not in fields:
        raise ScanError(f"{path} must have 'date' and 'ticker' columns; got {fields}")
    return rows, fields


def index_articles(
    records: Iterable[tuple], tickers: set[str]
) -> dict[tuple[str, date], list[dict]]:
    """Pure half of the article-window match: DB rows -> {(ticker, day): articles}.

    `records` are (id, et_date, et_hour, published_et, title, url, primary,
    more_tickers) tuples. An article is indexed under every in-scope ticker it
    touches for its own ET date ("same-day") and -- when published at/after the
    16:00 ET close -- also the next calendar day ("prev-after-hours").
    """
    out: dict[tuple[str, date], list[dict]] = defaultdict(list)
    for aid, et_date, et_hour, published_et, title, url, primary, more in records:
        touched = set()
        if primary and primary.upper() in tickers:
            touched.add(primary.upper())
        for t in more or []:
            if t and t.upper() in tickers:
                touched.add(t.upper())
        base = {"id": aid, "published_et": published_et, "title": title, "url": url}
        for t in touched:
            out[(t, et_date)].append({**base, "when": "same-day"})
        if et_hour >= AFTER_HOURS_HOUR:  # after-hours -> also the next trading day
            nxt = et_date + timedelta(days=1)
            for t in touched:
                out[(t, nxt)].append({**base, "when": "prev-after-hours"})
    return out


def fetch_articles_by_ticker_day(
    tickers: set[str], start: date, end: date
) -> dict[tuple[str, date], list[dict]]:
    """Map (ticker, trading_date) -> [article dicts] for the scan's tickers.

    The window is widened one day earlier so the first row can still see its
    prior-day after-hours news.
    """
    from ticker_news.shared import db

    tlist = sorted(tickers)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id,
                   (published_utc AT TIME ZONE 'America/New_York')::date AS et_date,
                   EXTRACT(HOUR FROM published_utc AT TIME ZONE 'America/New_York')::int
                       AS et_hour,
                   to_char(published_utc AT TIME ZONE 'America/New_York',
                           'YYYY-MM-DD HH24:MI') AS published_et,
                   title, url, primary_ticker, more_tickers
            FROM public.articles
            WHERE (published_utc AT TIME ZONE 'America/New_York')::date
                      BETWEEN %s AND %s
              AND (primary_ticker = ANY(%s) OR more_tickers && %s::text[])
            ORDER BY published_utc
            """,
            (start - timedelta(days=1), end, tlist, tlist),
        )
        records = cur.fetchall()
    return index_articles(records, tickers)


def attach(input_csv: str, output_csv: str) -> int:
    """Add article_count/articles columns to a scan CSV; return the row count."""
    rows, fields = _read_scan_rows(input_csv)
    if not rows:
        raise ScanError(f"{input_csv} has no data rows.")

    tickers = {r["ticker"].strip().upper() for r in rows if r.get("ticker")}
    dates = [date.fromisoformat(r["date"]) for r in rows]
    by_day = fetch_articles_by_ticker_day(tickers, min(dates), max(dates))

    out_fields = fields + [f for f in ("article_count", "articles") if f not in fields]
    total = 0
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for r in rows:
            key = (r["ticker"].strip().upper(), date.fromisoformat(r["date"]))
            arts = by_day.get(key, [])
            total += len(arts)
            r["article_count"] = len(arts)
            r["articles"] = json.dumps(arts, ensure_ascii=False)
            writer.writerow(r)

    n_with = sum(1 for r in rows if r["article_count"])
    print(
        f"Wrote {len(rows)} row(s) to {output_csv}: "
        f"{n_with} had matching articles ({total} article-attachments total)."
    )
    return len(rows)


# --------------------------------------------------------------------------- #
# Catalyst returns (catalyst_returns port)
# --------------------------------------------------------------------------- #
def load_articles(start: str, end: str, categories: Sequence[str]) -> list[dict]:
    """Catalyst-candidate articles (with a primary_ticker) published in [start, end]."""
    from ticker_news.shared import db

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, primary_ticker, more_tickers, category, published_utc, title
            FROM public.articles
            WHERE category = ANY(%s)
              AND primary_ticker IS NOT NULL
              AND (published_utc AT TIME ZONE 'America/New_York')::date
                      BETWEEN %s AND %s
            ORDER BY published_utc
            """,
            (list(categories), start, end),
        )
        rows = cur.fetchall()
    out = []
    for aid, primary, more, category, pub_utc, title in rows:
        more_up = [t.strip().upper() for t in (more or []) if t and t.strip()]
        out.append({
            "id": aid,
            "primary": primary.strip().upper(),
            "more": more_up,
            "category": category,
            "published_et": pub_utc.astimezone(MARKET_TZ),
            "title": title,
        })
    return out


class TickerPrices:
    """Minute-bar entry prices + daily-close exits for one ticker."""

    def __init__(self, minute_bars: list[dict], daily_bars: list[dict]):
        # minute series: parallel arrays sorted by time, for bisect lookups
        self.times: list[datetime] = []
        self.opens: list[float] = []
        for b in minute_bars:
            self.times.append(_bar_dt(b))
            self.opens.append(b["o"])
        # daily closes keyed by ET trading date, plus the sorted trading-day list
        self.close: dict[date, float] = {}
        for b in daily_bars:
            self.close[_bar_dt(b).date()] = b["c"]
        self.trading_days: list[date] = sorted(self.close)

    def entry(self, ts: datetime) -> Optional[tuple[datetime, float]]:
        """First tradeable (time, open) at/after the headline minute, or None."""
        floored = ts.replace(second=0, microsecond=0)
        i = bisect.bisect_left(self.times, floored)
        if i >= len(self.times):
            return None
        return self.times[i], self.opens[i]

    def next_trading_day(self, d: date) -> Optional[date]:
        i = bisect.bisect_right(self.trading_days, d)
        return self.trading_days[i] if i < len(self.trading_days) else None


def fetch_prices(ticker: str, frm: str, to: str, key: str | None = None) -> TickerPrices:
    k = api_key(key)
    minute = fetch_bars(ticker, span="minute", frm=frm, to=to, key=k)
    daily = fetch_bars(ticker, span="day", frm=frm, to=to, key=k)
    return TickerPrices(minute, daily)


def simulate(
    article: dict, prices: TickerPrices, include_after_hours: bool = False
) -> dict | None:
    """Buy at the headline, sell at the next regular close; return a CSV row.

    `article` needs ``id``, ``ticker`` and ``published_et`` (an aware datetime);
    ``category`` and ``title`` are carried through when present. The exit is
    gated by the *entry bar's* time: before 16:00 ET -> that day's close;
    at/after 16:00 ET (after_hours or closed) -> next trading day's close,
    excluded unless `include_after_hours`.
    """
    entry = prices.entry(article["published_et"])
    if entry is None:
        return None
    buy_dt, buy_price = entry
    if buy_price <= 0:
        return None

    # entry before the close -> same-day close; extended hours -> next trading day.
    if buy_dt.time() < REGULAR_CLOSE:
        sell_date: Optional[date] = buy_dt.date()
        hold = "same-day-close"
    else:
        # extended-hours entry: excluded by default (only same-day exits are kept).
        if not include_after_hours:
            return None
        sell_date = prices.next_trading_day(buy_dt.date())
        hold = "next-day-close"
    if sell_date is None or sell_date not in prices.close:
        return None
    sell_close = prices.close[sell_date]

    gain = (sell_close - buy_price) / buy_price * 100.0
    return {
        "article_id": article["id"],
        "ticker": article["ticker"],
        "category": article.get("category", ""),
        "published_et": article["published_et"].strftime("%Y-%m-%d %H:%M"),
        "entry_session": session_of(buy_dt.time()),
        "buy_et": buy_dt.strftime("%Y-%m-%d %H:%M"),
        "buy_price": round(buy_price, 4),
        "hold_until": hold,
        "sell_date": sell_date.isoformat(),
        "sell_close": round(sell_close, 4),
        "gain_pct": round(gain, 2),
        "title": article.get("title", ""),
    }


def catalyst_run(
    *,
    start: str,
    end: str,
    categories: Sequence[str],
    workers: int = 8,
    include_after_hours: bool = False,
    all_tickers: bool = False,
    key: str | None = None,
) -> list[dict]:
    """Simulate buy-the-news for every catalyst article; return the CSV rows."""
    k = api_key(key)
    articles = load_articles(start, end, categories)
    if not articles:
        print("No catalyst articles found for the range/categories.")
        return []

    # one spec per ticker to simulate: just the primary, or (with all_tickers)
    # every ticker the article names -- primary + more_tickers -- all entered at
    # the SAME article moment. Each carries a ticker_role for the output.
    specs: list[tuple[dict, str, str]] = []  # (article-for-simulate, ticker, role)
    for art in articles:
        names = [art["primary"]] + (art["more"] if all_tickers else [])
        for tk in names:
            role = "primary" if tk == art["primary"] else "mentioned"
            specs.append(({**art, "ticker": tk}, tk, role))

    tickers = sorted({tk for _, tk, _ in specs})
    print(
        f"{len(articles)} catalyst article(s) -> {len(specs)} (article,ticker) pair(s) "
        f"across {len(tickers)} ticker(s) [{', '.join(categories)}]"
        + ("  [all-tickers]" if all_tickers else "")
    )

    price_to = (date.fromisoformat(end) + timedelta(days=PRICE_TAIL_DAYS)).isoformat()
    prices: dict[str, TickerPrices] = {}
    n_price_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_prices, tk, start, price_to, k): tk for tk in tickers}
        it = as_completed(futs)
        pbar = tqdm(it, total=len(futs), unit="ticker", desc="prices") if tqdm else it
        for fut in pbar:
            tk = futs[fut]
            try:
                prices[tk] = fut.result()
            except Exception as exc:  # noqa: BLE001
                n_price_fail += 1
                print(f"  prices {tk}: {exc}", file=sys.stderr)

    rows: list[dict] = []
    n_skip = 0
    for art, tk, role in specs:
        tp = prices.get(tk)
        row = simulate(art, tp, include_after_hours) if tp else None
        if row is None:
            n_skip += 1
            continue
        row["ticker_role"] = role
        rows.append(row)

    rows.sort(key=lambda r: (r["published_et"], r["article_id"], r["ticker"]))
    print(
        f"{len(rows)} row(s) "
        f"({n_skip} (article,ticker) pair(s) skipped for missing prices, "
        f"{n_price_fail} ticker fetch failure(s))."
    )
    return rows


def write_catalyst_csv(rows: Sequence[dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CATALYST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def catalyst_summary(rows: Sequence[dict]) -> list[str]:
    """Legacy end-of-run stats: overall and per-category average gain / win rate."""
    if not rows:
        return []
    wins = sum(1 for r in rows if r["gain_pct"] > 0)
    avg = sum(r["gain_pct"] for r in rows) / len(rows)
    lines = [f"  avg gain {avg:+.2f}% | {wins}/{len(rows)} positive"]
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["gain_pct"])
    for cat, gs in sorted(by_cat.items()):
        w = sum(1 for g in gs if g > 0)
        lines.append(f"  {cat:<20} n={len(gs):<4} avg={sum(gs)/len(gs):+.2f}%  {w}/{len(gs)} up")
    return lines
