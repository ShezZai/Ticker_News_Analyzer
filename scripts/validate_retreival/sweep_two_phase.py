#!/usr/bin/env python3
"""Sweep the two-phase-similarity's (net_k, tau, budget) on the revisited table, Mode 1.

§9 found `two-phase-similarity` loses to `fusion` in the backtest frame (last-noforward) at the
default tau=0.70 / budget=50. The expensive parts of `two-phase-similarity` (the whole-article
net and each insight's cosine-to-seed) do NOT depend on tau/budget/net_k, so we
fetch them ONCE per cluster and then evaluate the whole grid cheaply in numpy.

For each (net_k, tau, budget): pool = articles in the top-net_k net that have an
insight with cosine >= tau; rerank that pool's insights by RRF(insight-cos,
article-rank); keep the top `budget`. Report article- and insight-level micro-F1,
flagging configs that beat the Mode-1 references from §9.

Usage:
    python sweep_two_phase.py docs/validations/article_clusters_revisited.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "search"))

from validate_retrieval import (  # noqa: E402
    article_facts, compute, ground_truth_insight_ids, load_clusters, micro,
    window_universes,
)
from search_articles_by_insights import _seed_window, get_conn  # noqa: E402
from search_articles import similar_to  # noqa: E402
from hybrid_retrieval import _insights_for_articles, _norm_rows, _seed_matrix  # noqa: E402

# Mode-1 references (from §9, revisited table) to beat.
REF = {"insight_art": 26.3, "fusion_art": 22.9, "insight_ins": 15.1, "fusion_ins": 18.4}

NET_KS = [50, 100, 150]
TAUS = [0.70, 0.74, 0.78, 0.82]
BUDGETS = [20, 30, 50, 80]
RRF_C = 60
NET_MIN_SIM = 0.45
MONTHS_BEFORE = 3


def cluster_ingredients(conn, c, months_before):
    """Everything Mode-1 two-phase-similarity needs, computed once (independent of tau/budget/net_k)."""
    gt_col = c.all_before
    event = set(gt_col) | set(c.later_ids)
    facts = article_facts(conn, event)
    later = [i for i in c.later_ids
             if facts.get(i) and facts[i]["published_utc"] and facts[i]["has_insights"]]
    if not later:
        return None
    seed = max(later, key=lambda i: facts[i]["published_utc"])
    seed_pub = facts[seed]["published_utc"]
    since, until = _seed_window(seed_pub, months_before, exclusive=True)

    relevant = set(event) - {seed}
    miss = [i for i in relevant if i not in facts]
    if miss:
        facts.update(article_facts(conn, miss))
    retr_ins = {i for i in relevant
                if facts.get(i) and facts[i]["published_utc"]
                and (not since or facts[i]["published_utc"].isoformat() >= since)
                and facts[i]["published_utc"] < seed_pub and facts[i]["has_insights"]}
    relevant_insights = ground_truth_insight_ids(conn, retr_ins)
    _u_emb, u_art_ins, u_ins = window_universes(conn, since, until)

    # the wide net (max net_k) + per-insight cosine to seed
    whole = similar_to(seed, k=max(NET_KS), since=since, until=until,
                       min_similarity=NET_MIN_SIM, until_exclusive=True, conn=conn)
    art_rank = {h.id: r for r, h in enumerate(whole, 1)}
    S, _ = _seed_matrix(conn, seed)
    rows = _insights_for_articles(conn, list(art_rank)) if (art_rank and S.shape[0]) else []
    if rows:
        iids = np.array([r[0] for r in rows])
        aids = np.array([r[1] for r in rows])
        E = _norm_rows(np.vstack([np.asarray(r[2], dtype=np.float32) for r in rows]))
        icos = (E @ S.T).max(axis=1)
        a_rank = np.array([art_rank[a] for a in aids])
    else:
        iids = aids = icos = a_rank = np.array([])

    return dict(relevant=relevant, retr_ins=retr_ins, relevant_insights=relevant_insights,
                u_art_ins=u_art_ins, u_ins=u_ins,
                iids=iids, aids=aids, icos=icos, a_rank=a_rank)


def two_phase_sets(ing, net_k, tau, budget):
    """Return (retrieved_articles, retrieved_insights) for one config, in numpy."""
    iids, aids, icos, a_rank = ing["iids"], ing["aids"], ing["icos"], ing["a_rank"]
    if iids.size == 0:
        return set(), set()
    in_net = a_rank <= net_k
    if not in_net.any():
        return set(), set()
    pool_arts = set(aids[in_net & (icos >= tau)].tolist())   # stage-1 cascade pool
    if not pool_arts:
        return set(), set()
    # stage-2 candidates: pool-article insights that THEMSELVES clear tau
    sel = in_net & np.isin(aids, list(pool_arts)) & (icos >= tau)
    ci, ca, cc, cr = iids[sel], aids[sel], icos[sel], a_rank[sel]
    # rank1 = insight-cosine rank (1-based), descending
    order = np.argsort(-cc)
    rank1 = np.empty(len(cc), dtype=int)
    rank1[order] = np.arange(1, len(cc) + 1)
    rrf = 1.0 / (RRF_C + rank1) + 1.0 / (RRF_C + cr)
    top = np.argsort(-rrf)[:budget]
    return set(ca[top].tolist()), set(ci[top].tolist())


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--months-before", type=int, default=MONTHS_BEFORE)
    args = p.parse_args(argv)

    conn = get_conn()
    ings = []
    try:
        for c in load_clusters(args.csv):
            if not c.later_ids:
                continue
            ing = cluster_ingredients(conn, c, args.months_before)
            if ing:
                ings.append(ing)
                print(f"  prepped: {c.theme[:50]}", file=sys.stderr)
    finally:
        conn.close()
    print(f"\n{len(ings)} clusters prepared. Sweeping "
          f"{len(NET_KS)}x{len(TAUS)}x{len(BUDGETS)} configs …\n", file=sys.stderr)

    best = {"art": (None, -1.0), "ins": (None, -1.0)}
    for level in ("art", "ins"):
        ref_b = REF["fusion_art" if level == "art" else "fusion_ins"]
        ref_i = REF["insight_art" if level == "art" else "insight_ins"]
        print(f"\n=== {('ARTICLE' if level=='art' else 'INSIGHT')} micro-F1  "
              f"(beat fusion {ref_b:.1f} / insight {ref_i:.1f})  rows=tau cols=budget ===")
        for net_k in NET_KS:
            print(f"\n  net_k={net_k}")
            print("    tau \\ budget " + "".join(f"{b:>8}" for b in BUDGETS))
            for tau in TAUS:
                cells = []
                for budget in BUDGETS:
                    ms = []
                    for ing in ings:
                        r_art, r_ins = two_phase_sets(ing, net_k, tau, budget)
                        if level == "art":
                            ms.append(compute("s", r_art, ing["relevant"],
                                              ing["retr_ins"], ing["u_art_ins"]))
                        else:
                            ms.append(compute("s", r_ins, ing["relevant_insights"],
                                              ing["relevant_insights"], ing["u_ins"]))
                    f1 = micro(ms, "s").f1 * 100
                    if f1 > best[level][1]:
                        best[level] = ((net_k, tau, budget), f1)
                    mark = "*" if f1 > ref_b else ("+" if f1 > ref_i else " ")
                    cells.append(f"{f1:>6.1f}{mark}")
                print(f"    {tau:>4.2f}        " + "".join(cells))
        cfg, f1 = best[level]
        print(f"\n  BEST {level}: net_k={cfg[0]} tau={cfg[1]} budget={cfg[2]} -> "
              f"micro-F1 {f1:.1f}   (* > fusion, + > insight baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
