"""Backtest analyst-panel sentiment verdicts against realized returns.

Loads ``public.article_sentiment`` rows (produced by the LangGraph analyst
panel in :mod:`ticker_news.sentiment`) for articles published in a date range
and replays each verdict with the ticker-scan trade simulation: enter at the
publication minute, exit at the next regular close
(:func:`ticker_news.research.ticker_scan.simulate`). A buy verdict scores the
raw gain, a sell verdict the negated gain.

This is the eval ground-truth generator: each output row pairs a verdict with
its realized return. A future evals milestone will write these rows as scores
onto the article's Langfuse trace — out of scope here.

Hold handling: hold verdicts are skipped by default (``skipped_hold``). With
``include_hold=True`` they are simulated like buys so their raw buy-the-news
``gain_pct`` lands in the CSV, but ``signed_return_pct`` stays ``None`` — a
hold has no direction to score, so holds count toward ``n`` and never toward
``win_rate`` / ``avg_return_pct``.

No silent drops: the summary accounts for every input verdict — ``total`` =
``evaluated`` + sum of the ``skipped`` reason buckets (``skipped_hold``,
``skipped_after_hours`` for publications at/after 16:00 ET that the default
``include_after_hours=False`` simulation excludes, ``skipped_no_prices`` for
missing bars or failed price fetches).
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Sequence

from ticker_news.research.market_data import MARKET_TZ, REGULAR_CLOSE, api_key
from ticker_news.research.ticker_scan import PRICE_TAIL_DAYS, fetch_prices, simulate

BACKTEST_FIELDS = [
    "article_id", "ticker", "action", "confidence", "published_et",
    "entry_session", "buy_et", "buy_price", "hold_until", "sell_date",
    "sell_close", "gain_pct", "signed_return_pct", "skip_reason", "url", "title",
]

# Simulation columns carried verbatim from ticker_scan.simulate's output row.
_SIM_CARRY = ("entry_session", "buy_et", "buy_price", "hold_until",
              "sell_date", "sell_close", "gain_pct")

SKIP_HOLD = "skipped_hold"
SKIP_AFTER_HOURS = "skipped_after_hours"
SKIP_NO_PRICES = "skipped_no_prices"

_CONF_BUCKETS = ("<0.7", "0.7-0.85", ">=0.85")


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_verdicts(conn, start: str, end: str) -> list[dict]:
    """article_sentiment JOIN articles for articles published in [start, end].

    The window is the article's ET calendar date, inclusive on both ends —
    the same BETWEEN convention as :func:`ticker_scan.load_articles`.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.article_id, a.url, s.ticker, s.action, s.confidence,
                   a.published_utc, a.title
            FROM public.article_sentiment s
            JOIN public.articles a ON a.id = s.article_id
            WHERE (a.published_utc AT TIME ZONE 'America/New_York')::date
                      BETWEEN %s AND %s
            ORDER BY a.published_utc, s.article_id, s.ticker
            """,
            (start, end),
        )
        rows = cur.fetchall()
    return [
        {
            "article_id": aid,
            "url": url,
            "ticker": (ticker or "").strip().upper(),
            "action": (action or "").strip().lower(),
            # article_sentiment.confidence is a Postgres real; round away the
            # float32 noise (0.7 -> 0.699999988) so bucket edges hold exactly.
            "confidence": round(float(confidence), 4),
            "published_utc": pub_utc,
            "title": title,
        }
        for aid, url, ticker, action, confidence, pub_utc, title in rows
    ]


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def _result(verdict: dict, sim: dict | None = None, skip_reason: str | None = None) -> dict:
    pub_et = verdict["published_utc"].astimezone(MARKET_TZ)
    row = {
        "article_id": verdict["article_id"],
        "ticker": verdict["ticker"],
        "action": verdict["action"],
        "confidence": verdict["confidence"],
        "published_et": pub_et.strftime("%Y-%m-%d %H:%M"),
        "skip_reason": skip_reason,
        "url": verdict.get("url", ""),
        "title": verdict.get("title", ""),
    }
    for field in _SIM_CARRY:
        row[field] = sim.get(field) if sim else None
    if sim is None or verdict["action"] == "hold":
        row["signed_return_pct"] = None  # holds are tracked but never scored
    else:
        gain = sim["gain_pct"]
        row["signed_return_pct"] = gain if verdict["action"] == "buy" else -gain
    return row


def _fetch_all_prices(verdicts: Sequence[dict], *, workers: int, key: str | None) -> dict:
    """One fetch per ticker; the window spans that ticker's publication dates
    plus PRICE_TAIL_DAYS of tail (mirrors catalyst_run's windowing)."""
    k = api_key(key)
    windows: dict[str, tuple[date, date]] = {}
    for v in verdicts:
        d = v["published_utc"].astimezone(MARKET_TZ).date()
        lo, hi = windows.get(v["ticker"], (d, d))
        windows[v["ticker"]] = (min(lo, d), max(hi, d))

    prices: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                fetch_prices, tk, lo.isoformat(),
                (hi + timedelta(days=PRICE_TAIL_DAYS)).isoformat(), k,
            ): tk
            for tk, (lo, hi) in windows.items()
        }
        for fut in as_completed(futs):
            tk = futs[fut]
            try:
                prices[tk] = fut.result()
            except Exception as exc:  # noqa: BLE001 - missing prices become a skip bucket
                print(f"  prices {tk}: {exc}", file=sys.stderr)
    return prices


def evaluate(
    verdicts: Sequence[dict],
    *,
    include_hold: bool = False,
    workers: int = 8,
    key: str | None = None,
) -> list[dict]:
    """One result row per input verdict, in input order.

    Evaluated rows carry the simulate() trade columns plus
    ``signed_return_pct`` (= gain_pct for buy, -gain_pct for sell, None for
    hold); rows that cannot be evaluated carry a ``skip_reason`` instead.
    ``include_after_hours`` stays False: a simulate() exclusion for a verdict
    published at/after 16:00 ET is reported as ``skipped_after_hours``, any
    other missing-bar/missing-price case as ``skipped_no_prices``.
    """
    todo = [v for v in verdicts if include_hold or v["action"] != "hold"]
    prices = _fetch_all_prices(todo, workers=workers, key=key) if todo else {}

    results: list[dict] = []
    for v in verdicts:
        if v["action"] == "hold" and not include_hold:
            results.append(_result(v, skip_reason=SKIP_HOLD))
            continue
        tp = prices.get(v["ticker"])
        if tp is None:
            results.append(_result(v, skip_reason=SKIP_NO_PRICES))
            continue
        pub_et = v["published_utc"].astimezone(MARKET_TZ)
        sim = simulate(
            {"id": v["article_id"], "ticker": v["ticker"],
             "published_et": pub_et, "title": v.get("title", "")},
            tp,
        )
        if sim is None:
            reason = SKIP_AFTER_HOURS if pub_et.time() >= REGULAR_CLOSE else SKIP_NO_PRICES
            results.append(_result(v, skip_reason=reason))
            continue
        results.append(_result(v, sim=sim))
    return results


# --------------------------------------------------------------------------- #
# Summarize (pure)
# --------------------------------------------------------------------------- #
def _conf_bucket(confidence: float) -> str:
    if confidence >= 0.85:
        return ">=0.85"
    if confidence >= 0.7:
        return "0.7-0.85"
    return "<0.7"


def _stats(rows: Sequence[dict]) -> dict:
    signed = [r["signed_return_pct"] for r in rows if r.get("signed_return_pct") is not None]
    return {
        "n": len(rows),
        "win_rate": (sum(1 for s in signed if s > 0) / len(signed)) if signed else None,
        "avg_return_pct": (sum(signed) / len(signed)) if signed else None,
    }


def summarize(results: Sequence[dict]) -> dict:
    """Win rate (signed_return_pct > 0) and average signed return overall,
    per action, and per confidence bucket; plus skip-reason accounting so
    every input row is visible (total == evaluated + sum(skipped))."""
    skipped: dict[str, int] = {}
    evaluated: list[dict] = []
    for r in results:
        reason = r.get("skip_reason")
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            evaluated.append(r)
    return {
        "total": len(results),
        "evaluated": len(evaluated),
        "skipped": skipped,
        "overall": _stats(evaluated),
        "by_action": {
            action: _stats([r for r in evaluated if r["action"] == action])
            for action in sorted({r["action"] for r in evaluated})
        },
        "by_confidence": {
            bucket: _stats([r for r in evaluated if _conf_bucket(r["confidence"]) == bucket])
            for bucket in _CONF_BUCKETS
        },
    }


def summary_lines(summary: dict) -> list[str]:
    """The end-of-run table, one string per line."""
    sk = summary["skipped"]
    skip_txt = (
        " (" + ", ".join(f"{k}={v}" for k, v in sorted(sk.items())) + ")" if sk else ""
    )
    lines = [
        f"{summary['total']} verdict(s): {summary['evaluated']} evaluated, "
        f"{sum(sk.values())} skipped{skip_txt}"
    ]

    def fmt(label: str, st: dict) -> str:
        wr = f"{st['win_rate'] * 100:5.1f}%" if st["win_rate"] is not None else "    -"
        avg = (
            f"{st['avg_return_pct']:+6.2f}%" if st["avg_return_pct"] is not None else "     -"
        )
        return f"  {label:<16} n={st['n']:<5} win={wr}  avg={avg}"

    lines.append(fmt("overall", summary["overall"]))
    for action, st in summary["by_action"].items():
        lines.append(fmt(f"action {action}", st))
    for bucket in _CONF_BUCKETS:
        lines.append(fmt(f"conf {bucket}", summary["by_confidence"][bucket]))
    return lines


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def write_csv(rows: Sequence[dict], path: str) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=BACKTEST_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in BACKTEST_FIELDS})


def run_backtest(
    *,
    start: str,
    end: str,
    include_hold: bool = False,
    out: str | None = None,
    workers: int = 8,
    key: str | None = None,
) -> dict:
    """Load verdicts, evaluate, write the per-verdict CSV, print the summary."""
    from ticker_news.shared import db

    conn = db.connect()
    try:
        verdicts = load_verdicts(conn, start, end)
    finally:
        conn.close()
    print(f"{len(verdicts)} verdict(s) on articles published {start}..{end}")

    results = evaluate(verdicts, include_hold=include_hold, workers=workers, key=key)
    path = out or f"backtest_verdicts_{start}_{end}.csv"
    write_csv(results, path)
    print(f"Wrote {len(results)} row(s) to {path}")

    summary = summarize(results)
    for line in summary_lines(summary):
        print(line)
    return summary
