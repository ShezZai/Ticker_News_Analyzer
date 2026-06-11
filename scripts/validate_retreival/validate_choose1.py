#!/usr/bin/env python3
"""Validate insight_sentiment's --choose-1 mode over the peaceful-days pool.

Unlike validate_sentiment.py (which always judges the PRIMARY ticker), --choose-1
shows the model ALL of an article's tickers in one prompt and asks it to return
only the single most-confident verdict. So the ticker that gets scored is whatever
the MODEL picked -- which may be the primary or any mentioned peer.

For each sampled article this:
  1. builds the two-phase-similarity related-insight context,
  2. runs the verdict with --include-bias + --choose-1 (one Gemini call),
  3. takes the chosen {ticker, action, confidence}, and
  4. pairs it with the realized buy-at-publish -> sell-at-close return for THAT
     chosen ticker (catalyst_returns.simulate).

Correctness = the action's sign matches the realized move (BUY needs gain>0,
SELL needs gain<0; HOLD is not scored directionally). Writes a Markdown table +
summary to --out.

Usage:
    python validate_choose1.py --n 100 --seed 42 --out /tmp/choose1.md

Requires MASSIVE_API_KEY + NEWS_DB_DSN + GOOGLE_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "search"))
sys.path.insert(0, os.path.join(_HERE, "..", "ticker_scan"))

import insight_sentiment as isent  # noqa: E402
from catalyst_returns import fetch_prices, simulate, PRICE_TAIL_DAYS  # noqa: E402

POOL = "docs/validations/peaceful_days_articles.csv"
MONTHS_BEFORE, K, MIN_SIM = 3, 5, 0.7

BUYLIKE = {"buy", "strong_buy"}
SELLLIKE = {"sell", "strong_sell"}
ACT_ORDER = ["strong_buy", "buy", "hold", "sell", "strong_sell"]


def targets_of(article: dict) -> List[str]:
    """All tickers the article names (primary first, then more_tickers), deduped."""
    seen: set[str] = set()
    out = []
    for t in [article["primary_ticker"]] + (article["more_tickers"] or []):
        if t and t.upper() not in seen:
            seen.add(t.upper())
            out.append(t.upper())
    return out


def choose_one_verdict(conn, article, seeds, related, targets, model) -> Optional[dict]:
    """Run the single-call --choose-1 + --include-bias verdict; return one dict."""
    prompt = isent.build_prompt(targets, article, seeds, related, isent.ACTIONS,
                                bias=True, choose_one=True)
    try:
        out = isent.ask_gemini(prompt, model, actions=isent.ACTIONS)
    except BaseException as exc:  # noqa: BLE001  (ask_gemini raises SystemExit on failure)
        print(f"  a#{article['id']}: verdict failed ({exc!r})", file=sys.stderr)
        return None
    return out[0] if out else None


def correctness(action: str, gain: float) -> str:
    if action in BUYLIKE:
        return "✓" if gain > 0 else "✗"
    if action in SELLLIKE:
        return "✓" if gain < 0 else "✗"
    return "—"  # hold: not scored directionally


def csv_date_plus(d: str, days: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=POOL)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oversample", type=int, default=60,
                   help="extra candidates to draw so N survive price/return gaps")
    p.add_argument("--model", default=isent.GEMINI_MODEL)
    p.add_argument("--out", default="/tmp/choose1.md")
    args = p.parse_args(argv)

    key = os.getenv("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY not set.")

    with open(args.pool) as fh:
        pool = [r for r in csv.DictReader(fh) if r["primary_ticker"]]
    random.Random(args.seed).shuffle(pool)
    cand = pool[: args.n + args.oversample]
    print(f"pool={len(pool)}  drawing {len(cand)} candidates for {args.n} rows",
          file=sys.stderr)

    # market_date per article (to bound the price window for the chosen ticker)
    mkt_date = {int(r["article_id"]): r["market_date"] for r in cand}

    # cache by (ticker, market_date): full-span minute fetches are slow (heavy
    # pagination), so fetch a narrow window per article instead -- ~1s each.
    prices: Dict[tuple, object] = {}

    def price_for(tkr: str, around: str):
        ckey = (tkr, around)
        if ckey not in prices:
            frm = around
            to = csv_date_plus(around, PRICE_TAIL_DAYS)
            try:
                prices[ckey] = fetch_prices(tkr, frm, to, key)
            except Exception as exc:  # noqa: BLE001
                print(f"  price fetch failed for {tkr}: {exc}", file=sys.stderr)
                prices[ckey] = None
        return prices[ckey]

    conn = isent.get_conn()
    rows = []
    n_unpriced = 0
    try:
        for r in cand:
            if len(rows) >= args.n:
                break
            aid = int(r["article_id"])
            article = isent.load_article(conn, aid)
            if not article["published_utc"] or not article["title"]:
                continue
            targets = targets_of(article)
            if not targets:
                continue

            seeds = isent.insights_of(conn, aid)
            related = isent.gather_related_two_phase(
                conn, aid, MONTHS_BEFORE, K, True, MIN_SIM, isent.DEF_NET_K,
                isent.DEF_TWO_PHASE_NET_MIN_SIM, isent.DEF_TWO_PHASE_TAU_INS,
                isent.DEF_TWO_PHASE_BUDGET)

            v = choose_one_verdict(conn, article, seeds, related, targets, args.model)
            if not v:
                continue
            chosen = str(v.get("ticker", "")).upper()
            if not chosen:
                continue
            role = "primary" if chosen == article["primary_ticker"].upper() else "mentioned"

            tp = price_for(chosen, mkt_date.get(aid, r["market_date"]))
            if tp is None:
                n_unpriced += 1
                continue
            pub_et = article["published_utc"].astimezone(isent.MARKET_TZ)
            sim = simulate((aid, chosen, article["category"], pub_et, article["title"]),
                           tp, include_after_hours=True)
            if sim is None:
                n_unpriced += 1
                continue

            rows.append({
                "aid": aid, "ticker": chosen, "role": role,
                "n_tickers": len(targets), "date": sim["sell_date"],
                "gain": sim["gain_pct"], "act": v["action"],
                "conf": float(v["confidence"]),
            })
            print(f"  [{len(rows)}/{args.n}] a#{aid} chose {chosen} ({role}, "
                  f"of {len(targets)}) gain={sim['gain_pct']:+.2f} "
                  f"{v['action']}({float(v['confidence']):.2f})", file=sys.stderr)
    finally:
        conn.close()

    write_report(rows, args, n_unpriced)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _fmt(m):
    return "n/a" if m is None else f"{m:+.2f}%"


def write_report(rows, args, n_unpriced: int) -> None:
    buys = [r for r in rows if r["act"] in BUYLIKE]
    sells = [r for r in rows if r["act"] in SELLLIKE]
    holds = [r for r in rows if r["act"] == "hold"]
    dirl = buys + sells
    hits = sum(1 for r in buys if r["gain"] > 0) + sum(1 for r in sells if r["gain"] < 0)
    hr = f"{hits / len(dirl) * 100:.0f}% ({hits}/{len(dirl)})" if dirl else "n/a"
    mb, ms = _mean([r["gain"] for r in buys]), _mean([r["gain"] for r in sells])
    spread = f"{mb - ms:+.2f}pt" if (mb is not None and ms is not None) else "n/a"

    lines = [
        f"Sample: {len(rows)} articles from `{os.path.basename(args.pool)}` "
        f"(seed={args.seed}, model={args.model}). One `--choose-1 --include-bias` "
        f"call per article over two-phase-similarity context; the model is shown ALL "
        f"the article's tickers and returns the single most-confident verdict. "
        f"`role` = was the chosen ticker the article's primary or a mentioned peer. "
        f"gain% = buy-at-publish → close of the CHOSEN ticker. ✓/✗ = action sign "
        f"matches the realized move; — = HOLD. ({n_unpriced} draws dropped: chosen "
        f"ticker had no tradeable price.)\n",
        "**Summary**\n",
        f"- **{len(buys)} BUY-like / {len(sells)} SELL-like / {len(holds)} HOLD**; "
        f"directional hit-rate **{hr}**.",
        f"- Mean gain: BUY-like **{_fmt(mb)}**, SELL-like **{_fmt(ms)}**, "
        f"HOLD {_fmt(_mean([r['gain'] for r in holds]))}. BUY−SELL spread **{spread}**.",
        f"- Chosen ticker was the **primary** in "
        f"{sum(1 for r in rows if r['role'] == 'primary')}/{len(rows)} articles "
        f"(mentioned peer in {sum(1 for r in rows if r['role'] == 'mentioned')}).\n",
        "**Per-action buckets**\n",
        "| action | n | mean gain% | dir hit-rate |",
        "|---|--:|--:|--:|",
    ]
    for a in [x for x in ACT_ORDER if any(r["act"] == x for r in rows)]:
        rs = [r for r in rows if r["act"] == a]
        if a in BUYLIKE:
            h = sum(1 for r in rs if r["gain"] > 0); hrr = f"{h/len(rs)*100:.0f}% ({h}/{len(rs)})"
        elif a in SELLLIKE:
            h = sum(1 for r in rs if r["gain"] < 0); hrr = f"{h/len(rs)*100:.0f}% ({h}/{len(rs)})"
        else:
            hrr = "—"
        lines.append(f"| {a} | {len(rs)} | {_fmt(_mean([r['gain'] for r in rs]))} | {hrr} |")

    # confidence buckets over directional (non-hold) calls
    lines += ["", "**Confidence calibration** (directional calls only)\n",
              "| conf band | n | mean gain% | dir hit-rate |", "|---|--:|--:|--:|"]
    bands = [(0.0, 0.7, "<0.70"), (0.7, 0.85, "0.70–0.85"), (0.85, 1.01, "≥0.85")]
    for lo, hi2, lab in bands:
        rs = [r for r in dirl if lo <= r["conf"] < hi2]
        if not rs:
            continue
        h = sum(1 for r in rs if (r["act"] in BUYLIKE and r["gain"] > 0)
                or (r["act"] in SELLLIKE and r["gain"] < 0))
        lines.append(f"| {lab} | {len(rs)} | {_fmt(_mean([r['gain'] for r in rs]))} | "
                     f"{h/len(rs)*100:.0f}% ({h}/{len(rs)}) |")

    lines += ["", "<details>",
              "<summary>Full table (chosen ticker · verdict · confidence · buy→close gain)</summary>",
              "",
              "| # | a# | chosen | role | #tk | sell date | gain% | verdict (act·conf) | ✓ |",
              "|--:|--:|:--|:--|--:|:--|--:|:--|:-:|"]
    for i, r in enumerate(rows, 1):
        c = correctness(r["act"], r["gain"])
        lines.append(
            f"| {i} | {r['aid']} | {r['ticker']} | {r['role']} | {r['n_tickers']} | "
            f"{r['date']} | {r['gain']:+.2f} | {r['act']} · {r['conf']:.2f} | {c} |")
    lines += ["", "</details>"]

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
