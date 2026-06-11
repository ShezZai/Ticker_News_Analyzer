#!/usr/bin/env python3
"""Validate insight_sentiment's --clean-top-1 refinement over the peaceful-days pool.

--clean-top-1 = --choose-1 plus a refinement loop: after the model returns its single
most-confident verdict (ROUND 1), the context is (a) pruned to insights relevant to the
chosen ticker/verdict, (b) augmented with the top-k insights most similar to the verdict's
own justification (cosine >= floor, no lookahead), and the verdict is re-issued on that
cleaned+augmented context (ROUND 2 / "after cleaning").

For each sampled article this records BOTH rounds — action + confidence before and after the
cleaning phase — and pairs each round's verdict with the realized buy-at-publish -> sell-at-
close return of the ticker that round chose (the refinement can switch tickers). Correctness =
the action's sign matches the realized move (BUY needs gain>0, SELL needs gain<0; HOLD not
scored). Writes a Markdown table + before/after summary to --out.

Usage:
    python validate_clean_top1.py --n 100 --seed 42 --out /tmp/clean_top1.md

Requires MASSIVE_API_KEY + NEWS_DB_DSN + GOOGLE_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
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
    seen: set[str] = set()
    out = []
    for t in [article["primary_ticker"]] + (article["more_tickers"] or []):
        if t and t.upper() not in seen:
            seen.add(t.upper())
            out.append(t.upper())
    return out


def choose_one(article, seeds, related, targets, model) -> Optional[dict]:
    """One --choose-1 + --include-bias call; return the single chosen verdict."""
    prompt = isent.build_prompt(targets, article, seeds, related, isent.ACTIONS,
                                bias=True, choose_one=True)
    try:
        out = isent.ask_gemini(prompt, model, actions=isent.ACTIONS)
    except BaseException as exc:  # noqa: BLE001
        print(f"  a#{article['id']}: verdict failed ({exc!r})", file=sys.stderr)
        return None
    return out[0] if out else None


def correctness(action: str, gain: Optional[float]) -> str:
    if gain is None:
        return "·"
    if action in BUYLIKE:
        return "✓" if gain > 0 else "✗"
    if action in SELLLIKE:
        return "✓" if gain < 0 else "✗"
    return "—"


def csv_date_plus(d: str, days: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=POOL)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--oversample", type=int, default=70)
    p.add_argument("--refine-k", type=int, default=isent.DEFAULT_REFINE_K)
    p.add_argument("--refine-min-similarity", type=float,
                   default=isent.DEFAULT_REFINE_MIN_SIMILARITY)
    p.add_argument("--model", default=isent.GEMINI_MODEL)
    p.add_argument("--out", default="/tmp/clean_top1.md")
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
    mkt_date = {int(r["article_id"]): r["market_date"] for r in cand}

    prices: Dict[tuple, object] = {}

    def gain_for(tkr: str, around: str, article) -> Optional[float]:
        ckey = (tkr, around)
        if ckey not in prices:
            try:
                prices[ckey] = fetch_prices(tkr, around, csv_date_plus(around, PRICE_TAIL_DAYS), key)
            except Exception as exc:  # noqa: BLE001
                print(f"  price fetch failed for {tkr}: {exc}", file=sys.stderr)
                prices[ckey] = None
        tp = prices[ckey]
        if tp is None:
            return None
        pub_et = article["published_utc"].astimezone(isent.MARKET_TZ)
        sim = simulate((article["id"], tkr, article["category"], pub_et, article["title"]),
                       tp, include_after_hours=True)
        return None if sim is None else sim

    conn = isent.get_conn()
    rows = []
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

            # ROUND 1 (before cleaning)
            v0 = choose_one(article, seeds, related, targets, args.model)
            if not v0:
                continue
            tk0 = str(v0.get("ticker", "")).upper()

            # cleaning phase: prune to verdict, augment from justification
            cleaned, n_clean = isent.clean_related_for_choice(
                related, tk0, v0, article, args.model)
            present = {h.insight_id for h in cleaned} | {s.insight_id for s in seeds}
            extra = isent.retrieve_by_justification(
                conn, v0.get("justification", ""), article["published_utc"],
                MONTHS_BEFORE, True, aid, present,
                k=args.refine_k, min_sim=args.refine_min_similarity)
            new_related = isent.merge_related(cleaned, extra)

            # ROUND 2 (after cleaning)
            v1 = choose_one(article, seeds, new_related, targets, args.model)
            if not v1:
                continue
            tk1 = str(v1.get("ticker", "")).upper()

            # price both chosen tickers; require the FINAL one to be tradeable
            s0 = gain_for(tk0, mkt_date.get(aid, r["market_date"]), article)
            s1 = gain_for(tk1, mkt_date.get(aid, r["market_date"]), article)
            if s1 is None:
                continue
            g0 = s0["gain_pct"] if s0 else None
            g1 = s1["gain_pct"]

            rows.append({
                "aid": aid, "tk0": tk0, "tk1": tk1,
                "role1": "primary" if tk1 == article["primary_ticker"].upper() else "mentioned",
                "n_tickers": len(targets), "date": s1["sell_date"],
                "g0": g0, "g1": g1,
                "act0": v0["action"], "conf0": float(v0["confidence"]),
                "act1": v1["action"], "conf1": float(v1["confidence"]),
                "removed": n_clean, "added": len(extra),
            })
            chg = "" if (tk0 == tk1 and v0["action"] == v1["action"]) else "  *changed*"
            print(f"  [{len(rows)}/{args.n}] a#{aid} {tk0}->{tk1} g={g1:+.2f} "
                  f"{v0['action']}({v0['confidence']:.2f})->{v1['action']}({v1['confidence']:.2f}) "
                  f"-{n_clean}/+{len(extra)}{chg}", file=sys.stderr)
    finally:
        conn.close()

    write_report(rows, args)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(m):
    return "n/a" if m is None else f"{m:+.2f}%"


def _round_summary(rows, akey, ckey, gkey) -> str:
    b = [r for r in rows if r[akey] in BUYLIKE]
    s = [r for r in rows if r[akey] in SELLLIKE]
    h = [r for r in rows if r[akey] == "hold"]
    d = [r for r in (b + s) if r[gkey] is not None]
    hits = sum(1 for r in b if r[gkey] is not None and r[gkey] > 0) + \
        sum(1 for r in s if r[gkey] is not None and r[gkey] < 0)
    hr = f"{hits/len(d)*100:.0f}% ({hits}/{len(d)})" if d else "n/a"
    mb = _mean([r[gkey] for r in b]); ms = _mean([r[gkey] for r in s])
    spread = f"{mb-ms:+.2f}pt" if (mb is not None and ms is not None) else "n/a"
    return (f"| {len(b)} | {len(s)} | {len(h)} | {hr} | {_fmt(mb)} | {spread} |")


def write_report(rows, args) -> None:
    n = len(rows)
    changed_tk = sum(1 for r in rows if r["tk0"] != r["tk1"])
    changed_act = sum(1 for r in rows if r["act0"] != r["act1"])
    changed_any = sum(1 for r in rows
                      if r["tk0"] != r["tk1"] or r["act0"] != r["act1"]
                      or abs(r["conf0"] - r["conf1"]) > 1e-9)
    dconf = _mean([r["conf1"] - r["conf0"] for r in rows])
    mean_removed = _mean([r["removed"] for r in rows])
    mean_added = _mean([r["added"] for r in rows])

    lines = [
        f"Sample: {n} articles from `{os.path.basename(args.pool)}` (seed={args.seed}, "
        f"model={args.model}). `--choose-1 --include-bias` over two-phase-similarity context, "
        f"then the **clean-top-1** refinement (prune to the chosen verdict + augment with "
        f"top-{args.refine_k} insights ≥{args.refine_min_similarity} cosine to the justification, "
        f"no lookahead) and a re-judge. `before` = round 1, `after` = round 2 (post-cleaning). "
        f"gain% = buy-at-publish → close of each round's chosen ticker. ✓/✗ = action sign matches; "
        f"— = HOLD; · = unpriced.\n",
        "**Before vs after the cleaning phase**\n",
        "| round | BUY-like | SELL-like | HOLD | dir. hit-rate | BUY mean | BUY−SELL spread |",
        "|---|--:|--:|--:|--:|--:|--:|",
        "| before (round 1) " + _round_summary(rows, "act0", "conf0", "g0"),
        "| after (round 2) " + _round_summary(rows, "act1", "conf1", "g1"),
        "",
        "**Effect of the cleaning phase**\n",
        f"- Verdicts changed after cleaning: **{changed_any}/{n}** "
        f"(action changed {changed_act}, chosen ticker changed {changed_tk}).",
        f"- Mean confidence shift (after − before): **{dconf:+.3f}**.",
        f"- Mean insights pruned per article: **{mean_removed:.1f}**; "
        f"mean pulled from justification: **{mean_added:.1f}**.\n",
        "**Per-action buckets — after cleaning (round 2)**\n",
        "| action | n | mean gain% | dir hit-rate |",
        "|---|--:|--:|--:|",
    ]
    for a in [x for x in ACT_ORDER if any(r["act1"] == x for r in rows)]:
        rs = [r for r in rows if r["act1"] == a]
        if a in BUYLIKE:
            hh = sum(1 for r in rs if r["g1"] > 0); hr = f"{hh/len(rs)*100:.0f}% ({hh}/{len(rs)})"
        elif a in SELLLIKE:
            hh = sum(1 for r in rs if r["g1"] < 0); hr = f"{hh/len(rs)*100:.0f}% ({hh}/{len(rs)})"
        else:
            hr = "—"
        lines.append(f"| {a} | {len(rs)} | {_fmt(_mean([r['g1'] for r in rs]))} | {hr} |")

    lines += ["", "<details>",
              "<summary>Full table (chosen ticker · before/after verdict · confidence · buy→close gain)</summary>",
              "",
              "| # | a# | chosen | role | #tk | sell date | gain% | before (act·conf) | ✓ | after (act·conf) | ✓ | −rm/+add |",
              "|--:|--:|:--|:--|--:|:--|--:|:--|:-:|:--|:-:|:--|"]
    for i, r in enumerate(rows, 1):
        c0 = correctness(r["act0"], r["g0"])
        c1 = correctness(r["act1"], r["g1"])
        tkcol = r["tk1"] if r["tk0"] == r["tk1"] else f"{r['tk1']} (←{r['tk0']})"
        g0note = "" if (r["tk0"] == r["tk1"] or r["g0"] is None) else f" [r1 {r['g0']:+.2f}]"
        lines.append(
            f"| {i} | {r['aid']} | {tkcol} | {r['role1']} | {r['n_tickers']} | "
            f"{r['date']} | {r['g1']:+.2f}{g0note} | "
            f"{r['act0']} · {r['conf0']:.2f} | {c0} | "
            f"{r['act1']} · {r['conf1']:.2f} | {c1} | −{r['removed']}/+{r['added']} |")
    lines += ["", "</details>"]

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
