#!/usr/bin/env python3
"""Backtest --top-2 sentiment on real-news articles, sourced directly from the DB.

Pipeline:
  1. Query the DB for 'real news' articles published before after-hours
     (before 16:00 ET) in --start / --end range.
  2. Randomly sample --n of them.
  3. Fetch intraday + daily prices from the Massive API for every ticker
     mentioned across the sample; compute actual buy-at-publish / sell-at-close
     gain_pct for each (article, ticker) pair.
  4. Run --top-2 sentiment inline on each article:
       - 3+ tickers: joint call over all tickers, then 2 separate re-runs for
         the top-2 non-hold picks.
       - <3 tickers:  one separate call per ticker (no joint call).
  5. Collect top2_separate verdicts with confidence >= --threshold (default 0.85).
  6. Print a table of predicted direction vs actual gain_pct and accuracy stats.

Nothing is written to the database.

Usage:
    python backtest_top2.py
    python backtest_top2.py --n 20 --threshold 0.85 --seed 42
    python backtest_top2.py --start 2025-02-01 --end 2025-03-30 --out r.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import psycopg
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "ticker_scan"))

load_dotenv()

from insight_sentiment import (  # noqa: E402
    DEFAULT_K, DEFAULT_MIN_SIMILARITY, DEFAULT_MONTHS_BEFORE, GEMINI_MODEL,
    _annotate, _fmt_et, ask_gemini, build_prompt, gather_related, load_article,
)
from search_articles_by_insights import get_conn, insights_of  # noqa: E402
from catalyst_returns import (  # noqa: E402
    PRICE_TAIL_DAYS, TickerPrices, _api_key, fetch_prices, simulate,
)

MARKET_TZ = ZoneInfo("America/New_York")
DEFAULT_START = "2025-02-01"
DEFAULT_END = "2025-03-30"
DEFAULT_N = 20
DEFAULT_THRESHOLD = 0.85
DEFAULT_SEED = 42


# ── DB article loading ─────────────────────────────────────────────────────────

def load_candidates(conn, start: str, end: str) -> List[dict]:
    """Real-news articles published before 16:00 ET in the date range."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, published_utc, primary_ticker, more_tickers
            FROM public.articles
            WHERE category = 'real news'
              AND primary_ticker IS NOT NULL
              AND (published_utc AT TIME ZONE 'America/New_York')::date
                      BETWEEN %s AND %s
              AND (published_utc AT TIME ZONE 'America/New_York')::time < '16:00:00'
            ORDER BY published_utc
            """,
            (start, end),
        )
        rows = cur.fetchall()
    out = []
    for aid, title, pub_utc, primary, more in rows:
        pub_et = pub_utc.astimezone(MARKET_TZ) if pub_utc else None
        out.append({
            "article_id": aid,
            "title": title,
            "published_utc": pub_utc,
            "published_et": pub_et.strftime("%Y-%m-%d %H:%M") if pub_et else None,
            "primary_ticker": (primary or "").strip().upper(),
            "more_tickers": [t.strip().upper() for t in (more or []) if t and t.strip()],
        })
    return out


# ── Price fetching & gain computation ─────────────────────────────────────────

def fetch_all_prices(
    articles: List[dict], start: str, end: str,
    api_key: str, workers: int = 8,
) -> Dict[str, TickerPrices]:
    tickers: set[str] = set()
    for a in articles:
        tickers.add(a["primary_ticker"])
        tickers.update(a["more_tickers"])
    tickers.discard("")

    price_to = (date.fromisoformat(end) + timedelta(days=PRICE_TAIL_DAYS)).isoformat()
    prices: Dict[str, TickerPrices] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_prices, tk, start, price_to, api_key): tk for tk in tickers}
        for fut in as_completed(futs):
            tk = futs[fut]
            try:
                prices[tk] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  prices {tk}: {exc}", file=sys.stderr)
    return prices


def gain_for(art: dict, ticker: str, prices: Dict[str, TickerPrices]) -> Optional[float]:
    tp = prices.get(ticker)
    if tp is None:
        return None
    pub_et = art["published_utc"].astimezone(MARKET_TZ)
    row = simulate(
        (art["article_id"], ticker, "real news", pub_et, art["title"]),
        tp, include_after_hours=False,
    )
    return round(row["gain_pct"], 2) if row else None


# ── Regular runner (no top-2) ─────────────────────────────────────────────────

def run_regular_inline(
    conn, article_id: int,
    months_before: int, k: int, exclusive: bool, min_sim: float, model: str,
) -> dict:
    """Judge every ticker the article names: one joint call (>=3 tickers) or one
    call per ticker (<3). No top-2 re-runs. Returns all verdicts under 'verdicts'."""
    article = load_article(conn, article_id)
    if not article["published_utc"]:
        raise ValueError(f"a#{article_id} has no published_utc")

    seen: set[str] = set()
    targets = [
        t.upper() for t in ([article["primary_ticker"]] + article["more_tickers"])
        if t and not (t.upper() in seen or seen.add(t.upper()))
    ]
    if not targets:
        raise ValueError(f"a#{article_id} has no tickers")

    seeds = insights_of(conn, article_id)
    related = gather_related(
        conn, article_id, article["published_utc"],
        months_before, k, exclusive, min_sim,
    )

    def judge(tks: list[str]) -> list[dict]:
        return _annotate(
            ask_gemini(build_prompt(tks, article, seeds, related), model), article
        )

    if len(targets) < 3:
        verdicts = []
        for tk in targets:
            verdicts.extend(judge([tk]))
    else:
        verdicts = judge(targets)

    return {
        "article_id": article["id"],
        "published_et": _fmt_et(article["published_utc"]),
        "title": article["title"],
        "context": {"article_insights": len(seeds), "related_insights": len(related)},
        "tickers_judged": targets,
        "verdicts": verdicts,
    }


# ── Top-2 runner ───────────────────────────────────────────────────────────────

def run_top2_inline(
    conn, article_id: int,
    months_before: int, k: int, exclusive: bool, min_sim: float, model: str,
) -> dict:
    article = load_article(conn, article_id)
    if not article["published_utc"]:
        raise ValueError(f"a#{article_id} has no published_utc")

    seen: set[str] = set()
    targets = [
        t.upper() for t in ([article["primary_ticker"]] + article["more_tickers"])
        if t and not (t.upper() in seen or seen.add(t.upper()))
    ]
    if not targets:
        raise ValueError(f"a#{article_id} has no tickers")

    seeds = insights_of(conn, article_id)
    related = gather_related(
        conn, article_id, article["published_utc"],
        months_before, k, exclusive, min_sim,
    )

    def judge(tks: list[str]) -> list[dict]:
        return _annotate(
            ask_gemini(build_prompt(tks, article, seeds, related), model), article
        )

    if len(targets) >= 3:
        joint = judge(targets)
        ranked = sorted(
            (v for v in joint if v["action"] != "hold"),
            key=lambda v: float(v["confidence"]), reverse=True,
        )
        picks = ranked[:2]
        separate = [s for v in picks for s in judge([v["ticker"].upper()])]
        top2_tickers = [v["ticker"].upper() for v in picks]
    else:
        # fewer than 3 tickers: one call per ticker, no joint
        separate = []
        for tk in targets:
            separate.extend(judge([tk]))
        ranked = sorted(
            (v for v in separate if v["action"] != "hold"),
            key=lambda v: float(v["confidence"]), reverse=True,
        )
        top2_tickers = [v["ticker"].upper() for v in ranked[:2]]
        joint = separate

    return {
        "article_id": article["id"],
        "published_et": _fmt_et(article["published_utc"]),
        "title": article["title"],
        "context": {"article_insights": len(seeds), "related_insights": len(related)},
        "tickers_judged": targets,
        "top2_tickers": top2_tickers,
        "joint": joint,
        "top2_separate": separate,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end",   default=DEFAULT_END)
    p.add_argument("--n",     type=int,   default=DEFAULT_N,
                   help=f"articles to sample (default {DEFAULT_N})")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"min confidence to report, inclusive (default {DEFAULT_THRESHOLD})")
    p.add_argument("--max-threshold", type=float, default=None,
                   help="max confidence to report, inclusive (default: no upper bound)")
    p.add_argument("--no-top2", dest="top2", action="store_false", default=True,
                   help="skip the top-2 re-runs; just judge every ticker once")
    p.add_argument("--seed",  type=int,   default=DEFAULT_SEED)
    p.add_argument("--months-before", type=int,   default=DEFAULT_MONTHS_BEFORE)
    p.add_argument("-k", "--k",         type=int,   default=DEFAULT_K)
    p.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-similarity",  type=float, default=DEFAULT_MIN_SIMILARITY)
    p.add_argument("--model",           default=GEMINI_MODEL)
    p.add_argument("--api-key",         help="Massive API key override")
    p.add_argument("--out",             default=None,
                   help="output JSON (default backtest_top2_<n>.json)")
    args = p.parse_args(argv)

    massive_key = _api_key(args.api_key)

    # ── Step 1: candidates from DB ─────────────────────────────────────────────
    conn = get_conn()
    try:
        candidates = load_candidates(conn, args.start, args.end)
    finally:
        conn.close()

    print(
        f"DB candidates: {len(candidates)} real-news articles before 16:00 ET "
        f"in {args.start}..{args.end}",
        file=sys.stderr,
    )
    if not candidates:
        print("Nothing to sample.", file=sys.stderr)
        return 1

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.n, len(candidates)))
    sample.sort(key=lambda a: a["published_et"] or "")

    # ── Step 2: prices for every ticker in the sample ─────────────────────────
    print(f"Fetching prices for {args.start}..{args.end} (+ tail) …", file=sys.stderr)
    prices = fetch_all_prices(sample, args.start, args.end, massive_key)

    # ── Steps 3-5: top-2 + collect verdicts ───────────────────────────────────
    lo, hi = args.threshold, args.max_threshold
    band = f"{lo:.2f}..{hi:.2f}" if hi is not None else f"≥{lo:.2f}"
    mode = "top-2" if args.top2 else "all-ticker (no top-2)"
    print(f"\nRunning {mode} sentiment on {len(sample)} articles (conf {band}) …\n",
          file=sys.stderr)

    def in_band(c: float) -> bool:
        return c >= lo and (hi is None or c <= hi)

    conn = get_conn()
    results = []
    all_high: list[dict] = []

    try:
        for i, art in enumerate(sample, 1):
            aid = art["article_id"]
            print(
                f"  [{i:>2}/{len(sample)}] a#{aid}  {art['published_et']}  {art['title'][:60]}",
                file=sys.stderr,
            )
            try:
                if args.top2:
                    res = run_top2_inline(
                        conn, aid,
                        args.months_before, args.k, args.exclusive, args.min_similarity, args.model,
                    )
                    pool_verdicts = res["top2_separate"]
                else:
                    res = run_regular_inline(
                        conn, aid,
                        args.months_before, args.k, args.exclusive, args.min_similarity, args.model,
                    )
                    pool_verdicts = res["verdicts"]
            except BaseException as exc:  # catches SystemExit from ask_gemini too
                print(f"    SKIP: {exc}", file=sys.stderr)
                results.append({"article_id": aid, "error": str(exc)})
                continue

            high = [v for v in pool_verdicts if in_band(float(v["confidence"]))]
            enriched: list[dict] = []
            for v in high:
                tk = str(v["ticker"]).upper()
                actual = gain_for(art, tk, prices)
                correct: Optional[bool] = None
                if actual is not None:
                    correct = (actual > 0) if v["action"] == "buy" else (actual < 0)

                row = {
                    "ticker": tk,
                    "action": v["action"],
                    "confidence": round(float(v["confidence"]), 3),
                    "justification": v.get("justification", ""),
                    "actual_gain_pct": actual,
                    "direction_correct": correct,
                }
                enriched.append(row)
                all_high.append(row | {"article_id": aid, "title": art["title"]})

                tag = ("✓" if correct else "✗") if correct is not None else "?"
                gain_str = f"{actual:+.2f}%" if actual is not None else "N/A"
                print(
                    f"    {tag} {tk:<6} {v['action'].upper():<5} "
                    f"conf={v['confidence']:.2f}  actual={gain_str}",
                    file=sys.stderr,
                )

            results.append({
                "article_id": aid,
                "published_et": res["published_et"],
                "title": res["title"],
                "tickers_judged": res["tickers_judged"],
                "banded_verdicts": enriched,
                "all_verdicts": pool_verdicts,
            })
    finally:
        conn.close()

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = args.out or f"backtest_top2_{len(sample)}.json"
    with open(out_path, "w") as fh:
        json.dump({
            "threshold": args.threshold, "max_threshold": args.max_threshold,
            "top2": args.top2, "n_articles": len(sample),
            "seed": args.seed, "model": args.model,
            "start": args.start, "end": args.end,
            "results": results,
        }, fh, indent=2, ensure_ascii=False)

    # ── Summary table ──────────────────────────────────────────────────────────
    W = 100
    print(f"\n{'═'*W}")
    print(f"  BACKTEST (DB real-news, pre-16:00 ET, {mode})"
          f"  |  n={len(sample)}  |  conf {band}  |  seed={args.seed}")
    print(f"{'═'*W}")
    if all_high:
        print(f"  {'a#':<7} {'ticker':<7} {'action':<6} {'conf':>5}  "
              f"{'actual%':>9}  {'correct':<8}  title")
        print(f"  {'─'*93}")
        for v in all_high:
            gain_str = f"{v['actual_gain_pct']:+.2f}%" if v["actual_gain_pct"] is not None else "    N/A"
            corr = "✓ YES" if v["direction_correct"] else ("✗ NO " if v["direction_correct"] is False else "  ?  ")
            print(f"  {v['article_id']:<7} {v['ticker']:<7} {v['action']:<6} "
                  f"{v['confidence']:>5.2f}  {gain_str:>9}  {corr}  {v['title'][:52]}")
    else:
        print("  No verdicts fell within the confidence band.")

    n_rated = sum(1 for v in all_high if v["direction_correct"] is not None)
    n_correct = sum(1 for v in all_high if v["direction_correct"])
    print(f"\n  DB candidates: {len(candidates)}  |  Sampled: {len(sample)}")
    print(f"  Verdicts in band ({band}): {len(all_high)}")
    if n_rated:
        print(f"  Direction correct: {n_correct}/{n_rated}  ({n_correct/n_rated*100:.0f}%)")
        buys  = [v for v in all_high if v["action"] == "buy"  and v["actual_gain_pct"] is not None]
        sells = [v for v in all_high if v["action"] == "sell" and v["actual_gain_pct"] is not None]
        if buys:
            avg = sum(v["actual_gain_pct"] for v in buys) / len(buys)
            ok  = sum(1 for v in buys if v["direction_correct"])
            print(f"  BUY  verdicts: {ok}/{len(buys)} correct | avg actual {avg:+.2f}%")
        if sells:
            avg = sum(v["actual_gain_pct"] for v in sells) / len(sells)
            ok  = sum(1 for v in sells if v["direction_correct"])
            print(f"  SELL verdicts: {ok}/{len(sells)} correct | avg actual {avg:+.2f}%")
    print(f"\n  Full results → {out_path}")
    print(f"{'═'*W}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
