#!/usr/bin/env python3
"""Sentiment validation over the peaceful-days pool.

Samples N articles from docs/validations/peaceful_days_articles.csv, runs
insight_sentiment's verdict on the PRIMARY ticker under both retrievers
(`insight` and `two-phase-similarity`), and pairs each verdict with the realized
buy-at-publish -> sell-at-close return (catalyst_returns.simulate). Correctness =
the action's sign matches the realized move (BUY needs gain>0, SELL needs gain<0;
HOLD is not scored directionally).

Writes a consolidated Markdown table + summary to --out.

Usage:
    python validate_sentiment.py --n 50 --seed 42 --out /tmp/sent_val.md

Requires MASSIVE_API_KEY + NEWS_DB_DSN + GOOGLE_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
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


def verdict_for(conn, article, seeds, related, ticker, model, actions,
                bias: bool = False) -> Optional[dict]:
    prompt = isent.build_prompt([ticker], article, seeds, related, actions, bias)
    try:
        out = isent.ask_gemini(prompt, model, actions=actions)
    except BaseException as exc:  # noqa: BLE001  (ask_gemini raises SystemExit on failure)
        print(f"  a#{article['id']} {ticker}: verdict failed ({exc!r})", file=sys.stderr)
        return None
    for v in out:
        if str(v.get("ticker", "")).upper() == ticker.upper():
            return v
    return out[0] if out else None


def correctness(action: str, gain: float) -> str:
    if action in BUYLIKE:
        return "✓" if gain > 0 else "✗"
    if action in SELLLIKE:
        return "✓" if gain < 0 else "✗"
    return "—"  # hold: not scored directionally


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=POOL)
    p.add_argument("--db-range", nargs=2, metavar=("START", "END"), default=None,
                   help="instead of the pool CSV, sample real-news articles whose ET "
                        "date is in [START, END] (YYYY-MM-DD) straight from the DB")
    p.add_argument("--compare", choices=["retrievers", "actions", "bias", "bias-strong"],
                   default="retrievers",
                   help="retrievers: insight vs two-phase-similarity (both buy/sell/hold). "
                        "actions: two-phase-similarity with buy/sell/hold vs +--include-strong. "
                        "bias: two-phase-similarity without vs with +--include-bias. "
                        "bias-strong: plain two-phase vs two-phase +--include-bias +--include-strong.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oversample", type=int, default=40,
                   help="extra candidates to draw so N survive price/return gaps")
    p.add_argument("--model", default=isent.GEMINI_MODEL)
    p.add_argument("--out", default="/tmp/sent_val.md")
    args = p.parse_args(argv)

    key = os.getenv("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY not set.")

    if args.db_range:
        pool = load_db_range(*args.db_range)
        src = f"DB real news {args.db_range[0]}..{args.db_range[1]}"
    else:
        with open(args.pool) as fh:
            pool = [r for r in csv.DictReader(fh) if r["primary_ticker"]]
        src = os.path.basename(args.pool)
    args._src = src
    args._labels = {
        "actions": ("two-phase", "two-phase + strong"),
        "bias": ("two-phase", "two-phase + bias"),
        "bias-strong": ("two-phase", "two-phase + bias + strong"),
    }.get(args.compare, ("insight", "two-phase-similarity"))
    random.Random(args.seed).shuffle(pool)
    cand = pool[: args.n + args.oversample]
    print(f"pool={len(pool)}  drawing {len(cand)} candidates for {args.n} rows", file=sys.stderr)

    # fetch prices once per ticker over the span its candidates cover
    by_tkr: Dict[str, List[dict]] = defaultdict(list)
    for r in cand:
        by_tkr[r["primary_ticker"]].append(r)
    prices: Dict[str, object] = {}
    for tkr, rs in by_tkr.items():
        days = sorted(r["market_date"] for r in rs)
        frm, to = days[0], (csv_date_plus(days[-1], PRICE_TAIL_DAYS))
        try:
            prices[tkr] = fetch_prices(tkr, frm, to, key)
        except Exception as exc:  # noqa: BLE001
            print(f"  price fetch failed for {tkr}: {exc}", file=sys.stderr)

    conn = isent.get_conn()
    rows = []
    try:
        for r in cand:
            if len(rows) >= args.n:
                break
            aid = int(r["article_id"]); tkr = r["primary_ticker"]
            if tkr not in prices:
                continue
            article = isent.load_article(conn, aid)
            if not article["published_utc"] or not article["title"]:
                continue
            pub_et = article["published_utc"].astimezone(isent.MARKET_TZ)
            sim = simulate((aid, tkr, article["category"], pub_et, article["title"]),
                           prices[tkr], include_after_hours=True)
            if sim is None:
                continue  # no tradeable entry/close -> can't score
            gain = sim["gain_pct"]

            seeds = isent.insights_of(conn, aid)
            rel_two = isent.gather_related_two_phase(
                conn, aid, MONTHS_BEFORE, K, True, MIN_SIM, isent.DEF_NET_K,
                isent.DEF_TWO_PHASE_NET_MIN_SIM, isent.DEF_TWO_PHASE_TAU_INS,
                isent.DEF_TWO_PHASE_BUDGET)
            if args.compare == "actions":
                # same (two-phase) context; default action menu vs +strong
                v_a = verdict_for(conn, article, seeds, rel_two, tkr, args.model,
                                  isent.ACTIONS)
                v_b = verdict_for(conn, article, seeds, rel_two, tkr, args.model,
                                  isent.ACTIONS_STRONG)
            elif args.compare == "bias":
                # same (two-phase) context + actions; bias caveat off vs on
                v_a = verdict_for(conn, article, seeds, rel_two, tkr, args.model,
                                  isent.ACTIONS, bias=False)
                v_b = verdict_for(conn, article, seeds, rel_two, tkr, args.model,
                                  isent.ACTIONS, bias=True)
            elif args.compare == "bias-strong":
                # plain two-phase vs two-phase + bias caveat + strong action menu
                v_a = verdict_for(conn, article, seeds, rel_two, tkr, args.model,
                                  isent.ACTIONS, bias=False)
                v_b = verdict_for(conn, article, seeds, rel_two, tkr, args.model,
                                  isent.ACTIONS_STRONG, bias=True)
            else:
                rel_ins = isent.gather_related(conn, aid, article["published_utc"],
                                               MONTHS_BEFORE, K, True, MIN_SIM)
                v_a = verdict_for(conn, article, seeds, rel_ins, tkr, args.model,
                                  isent.ACTIONS)
                v_b = verdict_for(conn, article, seeds, rel_two, tkr, args.model,
                                  isent.ACTIONS)
            if not v_a or not v_b:
                continue
            rows.append({
                "aid": aid, "ticker": tkr, "date": sim["sell_date"], "gain": gain,
                "act_a": v_a["action"], "conf_a": float(v_a["confidence"]),
                "act_b": v_b["action"], "conf_b": float(v_b["confidence"]),
            })
            print(f"  [{len(rows)}/{args.n}] a#{aid} {tkr} gain={gain:+.2f}  "
                  f"a={v_a['action']}({v_a['confidence']:.2f}) "
                  f"b={v_b['action']}({v_b['confidence']:.2f})", file=sys.stderr)
    finally:
        conn.close()

    write_report(rows, args)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


def load_db_range(start: str, end: str) -> List[dict]:
    """Pool rows (article_id, primary_ticker, market_date) from the DB by ET date."""
    conn = isent.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, primary_ticker, "
                "       (published_utc AT TIME ZONE 'America/New_York')::date "
                "FROM public.articles "
                "WHERE category = 'real news' AND primary_ticker IS NOT NULL "
                "  AND (published_utc AT TIME ZONE 'America/New_York')::date BETWEEN %s AND %s",
                (start, end))
            return [{"article_id": str(r[0]), "primary_ticker": r[1].strip().upper(),
                     "market_date": r[2].isoformat()} for r in cur.fetchall()]
    finally:
        conn.close()


def csv_date_plus(d: str, days: int) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _fmt(m):
    return "n/a" if m is None else f"{m:+.2f}%"


def _summary_line(rows, akey) -> str:
    buys = [r for r in rows if r[akey] in BUYLIKE]
    sells = [r for r in rows if r[akey] in SELLLIKE]
    holds = [r for r in rows if r[akey] == "hold"]
    dirl = buys + sells
    hits = sum(1 for r in buys if r["gain"] > 0) + sum(1 for r in sells if r["gain"] < 0)
    hr = f"{hits / len(dirl) * 100:.0f}%" if dirl else "n/a"
    mb, ms = _mean([r["gain"] for r in buys]), _mean([r["gain"] for r in sells])
    spread = f"**{mb - ms:+.2f}pt**" if (mb is not None and ms is not None) else "n/a"
    return (f"- **{len(buys)} BUY-like / {len(sells)} SELL-like / {len(holds)} HOLD**; "
            f"directional hit-rate **{hr}** ({hits}/{len(dirl)}). "
            f"Mean gain: BUY-like **{_fmt(mb)}**, SELL-like **{_fmt(ms)}**, "
            f"HOLD {_fmt(_mean([r['gain'] for r in holds]))}. BUY−SELL spread {spread}.")


def _bucket_table(rows, akey) -> List[str]:
    out = ["| action | n | mean gain% | dir hit-rate |", "|---|--:|--:|--:|"]
    for a in [x for x in ACT_ORDER if any(r[akey] == x for r in rows)]:
        rs = [r for r in rows if r[akey] == a]
        if a in BUYLIKE:
            h = sum(1 for r in rs if r["gain"] > 0); hr = f"{h/len(rs)*100:.0f}% ({h}/{len(rs)})"
        elif a in SELLLIKE:
            h = sum(1 for r in rs if r["gain"] < 0); hr = f"{h/len(rs)*100:.0f}% ({h}/{len(rs)})"
        else:
            hr = "—"
        out.append(f"| {a} | {len(rs)} | {_fmt(_mean([r['gain'] for r in rs]))} | {hr} |")
    return out


def _transition(rows, la: str, lb: str) -> List[str]:
    """Verdict cross-tab (la -> lb), only the rows whose action changed."""
    from collections import Counter
    moved = Counter((r["act_a"], r["act_b"]) for r in rows if r["act_a"] != r["act_b"])
    if not moved:
        return ["_No verdict changed between the two variants._"]
    out = [f"_{sum(moved.values())} of {len(rows)} verdicts changed:_",
           "", f"| {la} | → {lb} | n |", "|---|---|--:|"]
    for (a, b), n in sorted(moved.items(), key=lambda kv: -kv[1]):
        out.append(f"| {a} | {b} | {n} |")
    return out


def write_report(rows, args) -> None:
    la, lb = getattr(args, "_labels", ("insight", "two-phase-similarity"))
    lines = [f"Sample: {len(rows)} articles from `{getattr(args, '_src', args.pool)}` "
             f"(seed={args.seed}, model={args.model}, compare={args.compare}). "
             f"Verdict on the **primary ticker**; gain = buy-at-publish → close. "
             f"✓/✗ = action sign matches the realized move; — = HOLD.\n",
             f"| # | a# | ticker | sell date | gain% | {la} (act·conf) | ✓ | "
             f"{lb} (act·conf) | ✓ |",
             "|--:|--:|:--|:--|--:|:--|:-:|:--|:-:|"]
    for i, r in enumerate(rows, 1):
        ca = correctness(r["act_a"], r["gain"])
        cb = correctness(r["act_b"], r["gain"])
        lines.append(
            f"| {i} | {r['aid']} | {r['ticker']} | {r['date']} | {r['gain']:+.2f} | "
            f"{r['act_a']} · {r['conf_a']:.2f} | {ca} | "
            f"{r['act_b']} · {r['conf_b']:.2f} | {cb} |")
    for label, akey in [(la, "act_a"), (lb, "act_b")]:
        lines.append(f"\n**Summary — `{label}`**")
        lines.append(_summary_line(rows, akey))
        lines += [""] + _bucket_table(rows, akey)
    if args.compare in ("actions", "bias", "bias-strong"):
        lines.append(f"\n**Verdict shift ({la} → {lb})**")
        lines += _transition(rows, la, lb)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
