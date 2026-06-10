#!/usr/bin/env python3
"""Compare retrieval methods on the cluster harness: baselines vs hybrids.

Evaluates five retrievers against article_clusters.csv at the article and insight
levels, reusing the metric machinery from validate_retrieval.py:

    whole-article   baseline: similar_to() over articles.embedding   (recall-y)
    insight         baseline: search_by_insights()                   (precision-y)
    intersection    hybrid: insight ∩ whole-article (naive overlap)
    cascade         hybrid: wide whole-article net -> insight filter  (recommended)
    fusion          hybrid: reciprocal-rank fusion of the two rankings

Prints an article-level and an insight-level leaderboard (micro + macro), plus a
per-cluster article-level F1 matrix. Writes a JSON report with --json.

Usage:
    python compare_hybrids.py docs/validations/article_clusters.csv
    python compare_hybrids.py docs/validations/article_clusters.csv --json hyb.json

Requires OPENAI_API_KEY + NEWS_DB_DSN (content-rich DB).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "search"))

from validate_retrieval import (  # noqa: E402
    Metrics, _pct, article_facts, compute, ground_truth_insight_ids,
    load_clusters, macro, micro, window_universes,
)
from search_articles_by_insights import _seed_window, get_conn, search_by_insights  # noqa: E402
from search_articles import similar_to  # noqa: E402
import hybrid_retrieval as hyb  # noqa: E402

METHOD_ORDER = ["whole-article", "insight", "intersection", "cascade", "fusion", "two-phase-similarity"]


def _consolidate(groups) -> "tuple[Set[int], Set[int]]":
    ins, arts = set(), set()
    for g in groups:
        for h in g.hits:
            ins.add(h.insight_id)
            arts.add(h.article_id)
    return ins, arts


def evaluate_cluster(conn, c, args) -> Optional[dict]:
    if not c.later_ids:
        return None
    gt_col = c.news_before if args.ground_truth == "news" else c.all_before
    event = (set(gt_col) | set(c.later_ids))
    facts = article_facts(conn, event | set(c.news_before))

    dated = sorted((i for i in event
                    if facts.get(i) and facts[i]["published_utc"] and facts[i]["has_insights"]),
                   key=lambda i: facts[i]["published_utc"])
    if not dated:
        return None
    later_known = [i for i in c.later_ids
                   if facts.get(i) and facts[i]["published_utc"] and facts[i]["has_insights"]]
    last_id = max(later_known, key=lambda i: facts[i]["published_utc"]) if later_known else dated[-1]
    last_pub = facts[last_id]["published_utc"]

    if args.mode == "middle-forward":
        mids = [i for i in dated if i != last_id]
        seed = mids[len(mids) // 2] if mids else last_id
        seed_pub = facts[seed]["published_utc"]
        since, _ = _seed_window(seed_pub, args.months_before, exclusive=True)
        until = last_pub.isoformat()
        until_exclusive = False
    else:  # last-noforward
        seed = last_id
        seed_pub = last_pub
        since, until = _seed_window(seed_pub, args.months_before, exclusive=True)
        until_exclusive = True

    relevant_articles = set(event)
    relevant_articles.discard(seed)
    missing = [i for i in relevant_articles if i not in facts]
    if missing:
        facts.update(article_facts(conn, missing))

    def in_window(i):
        f = facts.get(i)
        if not (f and f["published_utc"]):
            return False
        p = f["published_utc"]
        if since and p.isoformat() < since:
            return False
        return p < seed_pub if until_exclusive else p <= last_pub

    retr_ins = {i for i in relevant_articles if in_window(i) and facts[i]["has_insights"]}
    retr_whole = {i for i in relevant_articles if in_window(i) and facts[i]["has_embedding"]}
    relevant_insights = ground_truth_insight_ids(conn, retr_ins)
    u_art_emb, u_art_ins, u_ins = window_universes(conn, since, until)

    win = dict(since=since, until=until, exclusive=until_exclusive)

    # ---- run each method ----
    # whole-article baseline
    whole = similar_to(seed, k=args.article_k, since=since, until=until,
                       min_similarity=args.min_similarity, until_exclusive=until_exclusive, conn=conn)
    R_whole_art = {h.id for h in whole}

    # insight baseline
    groups = search_by_insights(article_id=seed, k=args.k, since=since, until=until,
                                min_similarity=args.min_similarity,
                                until_exclusive=until_exclusive, conn=conn)
    R_ins_ins, R_ins_art = _consolidate(groups)

    # hybrids
    h_int = hyb.retrieve_intersection(conn, seed, **win, k=args.k,
                                      min_sim=args.min_similarity, article_k=args.article_k)
    h_cas = hyb.retrieve_cascade(conn, seed, **win, net_k=args.net_k,
                                 net_min_sim=args.net_min_similarity, tau_ins=args.tau_insight,
                                 k=args.k, min_sim=args.min_similarity)
    h_fus = hyb.retrieve_fusion(conn, seed, **win, k=args.k, min_sim=args.min_similarity,
                                net_k=args.net_k, net_min_sim=args.net_min_similarity,
                                rrf_c=args.rrf_c, budget=args.budget)
    h_two = hyb.retrieve_two_phase_similarity(conn, seed, **win, net_k=args.net_k,
                               net_min_sim=args.two_phase_net_min_similarity,
                               tau_ins=args.two_phase_tau_insight,
                               rrf_c=args.rrf_c, budget=args.two_phase_budget,
                               k=args.k, min_sim=args.min_similarity)

    # ---- metrics ----
    art: Dict[str, Metrics] = {}
    ins: Dict[str, Metrics] = {}

    art["whole-article"] = compute("whole-article", R_whole_art, relevant_articles,
                                   retr_whole, u_art_emb)
    art["insight"] = compute("insight", R_ins_art, relevant_articles, retr_ins, u_art_ins)
    ins["insight"] = compute("insight", R_ins_ins, relevant_insights, relevant_insights, u_ins)
    for name, res in [("intersection", h_int), ("cascade", h_cas), ("fusion", h_fus),
                      ("two-phase-similarity", h_two)]:
        art[name] = compute(name, res.article_ids, relevant_articles, retr_ins, u_art_ins)
        ins[name] = compute(name, res.insight_ids, relevant_insights, relevant_insights, u_ins)

    return {"theme": c.theme, "seed_id": seed, "seed_published": seed_pub.isoformat(),
            "article": art, "insight": ins}


# --------------------------------------------------------------------------- #
def _leaderboard(title: str, results: List[dict], level: str, methods: List[str]) -> None:
    print(f"\n  {title}")
    print(f"  {'method':<16} {'avgRet':>7} {'prec':>7} {'recall':>7} {'rec*':>7} "
          f"{'acc':>7} {'F1':>7}   (micro | macro F1)")
    print("  " + "-" * 92)
    for m in methods:
        ms = [r[level][m] for r in results if m in r[level]]
        if not ms:
            continue
        mi, ma = micro(ms, m), macro(ms, m)
        avg_ret = sum(x.n_retrieved for x in ms) / len(ms)
        print(f"  {m:<16} {avg_ret:>7.1f} {_pct(mi.precision)} {_pct(mi.recall)} "
              f"{_pct(mi.recall_achievable)} {_pct(mi.accuracy)} {_pct(mi.f1)}   "
              f"| {_pct(ma.f1)}")


def _f1_matrix(results: List[dict], level: str, methods: List[str]) -> None:
    print(f"\n  PER-CLUSTER F1 — {level} level")
    head = "  " + f"{'event':<34}" + "".join(f"{m[:11]:>12}" for m in methods)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in results:
        row = f"  {r['theme'][:33]:<34}"
        for m in methods:
            v = r[level].get(m)
            row += f"{('  n/a' if (v is None or math.isnan(v.f1)) else f'{v.f1*100:5.1f}%'):>12}"
        print(row)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("-k", "--k", type=int, default=hyb.DEF_K)
    p.add_argument("--article-k", type=int, default=hyb.DEF_ARTICLE_K)
    p.add_argument("--min-similarity", type=float, default=hyb.DEF_MIN_SIM)
    p.add_argument("--net-k", type=int, default=hyb.DEF_NET_K)
    p.add_argument("--net-min-similarity", type=float, default=hyb.DEF_NET_MIN_SIM,
                   help="cascade/fusion article-net cosine floor (default 0.45)")
    p.add_argument("--tau-insight", type=float, default=hyb.DEF_TAU_INS,
                   help="cascade insight cosine-to-seed floor (default 0.70)")
    # two-phase-similarity has its OWN floors/budget, decoupled from cascade/fusion above.
    p.add_argument("--two-phase-net-min-similarity", type=float,
                   default=hyb.DEF_TWO_PHASE_NET_MIN_SIM,
                   help=f"two-phase-similarity article-net floor (default {hyb.DEF_TWO_PHASE_NET_MIN_SIM})")
    p.add_argument("--two-phase-tau-insight", type=float, default=hyb.DEF_TWO_PHASE_TAU_INS,
                   help=f"two-phase-similarity insight cosine-to-seed floor (default {hyb.DEF_TWO_PHASE_TAU_INS})")
    p.add_argument("--two-phase-budget", type=int, default=hyb.DEF_TWO_PHASE_BUDGET,
                   help=f"two-phase-similarity insights to keep (default {hyb.DEF_TWO_PHASE_BUDGET})")
    p.add_argument("--rrf-c", type=int, default=hyb.DEF_RRF_C)
    p.add_argument("--budget", type=int, default=hyb.DEF_BUDGET)
    p.add_argument("--months-before", type=int, default=3)
    p.add_argument("--ground-truth", choices=["all", "news"], default="all")
    p.add_argument("--mode", choices=["last-noforward", "middle-forward"],
                   default="last-noforward",
                   help="last-noforward: seed=last article, no lookahead. "
                        "middle-forward: seed=middle article, window to last article date.")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    clusters = load_clusters(args.csv)
    conn = get_conn()
    results: List[dict] = []
    try:
        for c in clusters:
            r = evaluate_cluster(conn, c, args)
            if r is None:
                print(f"  skip (no seed): {c.theme}", file=sys.stderr)
                continue
            results.append(r)
            print(f"  done: {c.theme}  (seed a#{r['seed_id']})", file=sys.stderr)
    finally:
        conn.close()

    if not results:
        print("No usable clusters.", file=sys.stderr)
        return 1

    W = 96
    print("\n" + "=" * W)
    print(f"  HYBRID RETRIEVAL COMPARISON  |  mode={args.mode}  |  {len(results)} clusters  |  "
          f"k={args.k} article_k={args.article_k} min_sim={args.min_similarity} "
          f"net_k={args.net_k} net_min={args.net_min_similarity} tau={args.tau_insight} "
          f"budget={args.budget}  |  two-phase-similarity: net_min={args.two_phase_net_min_similarity} "
          f"tau={args.two_phase_tau_insight} budget={args.two_phase_budget}")
    print("=" * W)
    _leaderboard("ARTICLE LEVEL leaderboard (micro; macro-F1 at right)", results,
                 "article", METHOD_ORDER)
    _leaderboard("INSIGHT LEVEL leaderboard (micro; macro-F1 at right)", results,
                 "insight", [m for m in METHOD_ORDER if m != "whole-article"])
    _f1_matrix(results, "article", METHOD_ORDER)
    print("\n  rec* = achievable recall (vs in-window, embedded ground truth).")
    print("=" * W)

    if args.json:
        payload = {"params": vars(args), "clusters": [
            {"theme": r["theme"], "seed_id": r["seed_id"],
             "article": {m: vars(v) for m, v in r["article"].items()},
             "insight": {m: vars(v) for m, v in r["insight"].items()}}
            for r in results]}
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2, default=lambda o: None)
        print(f"\n  JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
