#!/usr/bin/env python3
"""Measure what the `--remove-unuseful` screening does to the precision/recall
tradeoff, AFTER two-phase-similarity retrieval.

`compare_hybrids.py` scores retrieval only. This harness adds the production
screening step from `insight_sentiment.py` on top of the retrieved insights and
scores both, at the insight level, against the cluster ground truth:

    1. retrieve two-phase-similarity at the WIDE 0.45/0.70/50 point (high recall)
    2. hydrate the kept insight ids into InsightHit rows
    3. screen them with `screen_related()` (the headline-aware --remove-unuseful
       middle call), dropping incoherent / off-event insights
    4. compute insight-level precision/recall/F1 for (retrieved) vs (screened)

Seed/window/ground-truth mirror compare_hybrids' last-noforward mode. Use this to
answer: does the LLM screen trade a little recall for a lot of precision?

Usage:
    python eval_screen.py docs/validations/article_clusters.csv
    python eval_screen.py docs/validations/article_clusters.csv --model gemini-2.5-flash

Requires NEWS_DB_DSN (content-rich DB) + GOOGLE_API_KEY (screening).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "search"))

from validate_retrieval import (  # noqa: E402
    Metrics, _pct, article_facts, compute, ground_truth_insight_ids,
    load_clusters, macro, micro, window_universes,
)
from search_articles_by_insights import _seed_window, get_conn  # noqa: E402
import hybrid_retrieval as hyb  # noqa: E402
import insight_sentiment as isent  # noqa: E402

# the WIDE two-phase-similarity operating point under test.
NET_K = hyb.DEF_NET_K           # 150
NET_MIN_SIM = 0.45
TAU_INS = 0.70
BUDGET = 50
RRF_C = hyb.DEF_RRF_C           # 60
MONTHS_BEFORE = 3


def evaluate_cluster(conn, c, model) -> Optional[dict]:
    if not c.later_ids:
        return None
    event = set(c.all_before) | set(c.later_ids)
    facts = article_facts(conn, event | set(c.news_before))
    dated = sorted((i for i in event
                    if facts.get(i) and facts[i]["published_utc"] and facts[i]["has_insights"]),
                   key=lambda i: facts[i]["published_utc"])
    if not dated:
        return None
    later_known = [i for i in c.later_ids
                   if facts.get(i) and facts[i]["published_utc"] and facts[i]["has_insights"]]
    seed = max(later_known, key=lambda i: facts[i]["published_utc"]) if later_known else dated[-1]
    seed_pub = facts[seed]["published_utc"]
    since, until = _seed_window(seed_pub, MONTHS_BEFORE, exclusive=True)

    relevant = set(event) - {seed}
    miss = [i for i in relevant if i not in facts]
    if miss:
        facts.update(article_facts(conn, miss))

    def in_window(i):
        f = facts.get(i)
        if not (f and f["published_utc"]):
            return False
        p = f["published_utc"]
        if since and p.isoformat() < since:
            return False
        return p < seed_pub

    retr_ins_arts = {i for i in relevant if in_window(i) and facts[i]["has_insights"]}
    relevant_insights = ground_truth_insight_ids(conn, retr_ins_arts)
    _ue, _uai, u_ins = window_universes(conn, since, until)

    # 1. retrieve (wide two-phase-similarity) ------------------------------------
    res = hyb.retrieve_two_phase_similarity(
        conn, seed, since=since, until=until, exclusive=True, net_k=NET_K,
        net_min_sim=NET_MIN_SIM, tau_ins=TAU_INS, rrf_c=RRF_C, budget=BUDGET,
    )
    R_retr = set(res.insight_ids)

    # 2-3. hydrate + screen (the --remove-unuseful middle call) ------------------
    hits = isent._load_insight_hits(conn, res.scored)
    article = isent.load_article(conn, seed)
    targets = [t.upper() for t in ([article["primary_ticker"]] + article["more_tickers"]) if t]
    kept, removed = isent.screen_related(hits, article, targets, model) if hits else ([], 0)
    R_screen = {h.insight_id for h in kept}

    # 4. metrics (insight level) ------------------------------------------------
    m_retr = compute("retrieved", R_retr, relevant_insights, relevant_insights, u_ins)
    m_scr = compute("screened", R_screen, relevant_insights, relevant_insights, u_ins)
    return {"theme": c.theme, "seed": seed, "removed": removed,
            "retrieved": m_retr, "screened": m_scr}


def _row(theme, g, m: Metrics) -> str:
    return (f"  {theme:<38} {g:>4} {m.n_retrieved:>4} {m.tp:>4} "
            f"{_pct(m.precision)} {_pct(m.recall)} {_pct(m.f1)}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--model", default=isent.GEMINI_MODEL,
                   help=f"screening model (default {isent.GEMINI_MODEL})")
    args = p.parse_args(argv)

    conn = get_conn()
    results = []
    try:
        for c in load_clusters(args.csv):
            r = evaluate_cluster(conn, c, args.model)
            if r is None:
                print(f"  skip: {c.theme}", file=sys.stderr)
                continue
            results.append(r)
            print(f"  done: {c.theme[:46]}  (-{r['removed']} screened)", file=sys.stderr)
    finally:
        conn.close()
    if not results:
        print("No usable clusters.", file=sys.stderr)
        return 1

    for level in ("retrieved", "screened"):
        print(f"\n===== {level.upper()}  (two-phase-similarity {NET_MIN_SIM}/{TAU_INS}/{BUDGET}"
              f"{' + --remove-unuseful' if level == 'screened' else ''}) =====")
        print(f"  {'Cluster':<38} {'|G|':>4} {'ret':>4} {'TP':>4} {'prec':>7} {'recall':>7} {'F1':>7}")
        ms = [r[level] for r in results]
        for r in results:
            print(_row(r["theme"][:38], r[level].n_relevant, r[level]))
        mi, ma = micro(ms, level), macro(ms, level)
        print(f"  {'MICRO (pooled)':<38} {mi.n_relevant:>4} {mi.n_retrieved:>4} {mi.tp:>4} "
              f"{_pct(mi.precision)} {_pct(mi.recall)} {_pct(mi.f1)}")
        print(f"  {'MACRO (cluster mean)':<38} {'':>4} {'':>4} {'':>4} "
              f"{_pct(ma.precision)} {_pct(ma.recall)} {_pct(ma.f1)}")

    # headline delta
    rt = micro([r["retrieved"] for r in results], "retrieved")
    sc = micro([r["screened"] for r in results], "screened")
    print(f"\n  DELTA (screened - retrieved):  prec {(sc.precision-rt.precision)*100:+.1f}pt  "
          f"recall {(sc.recall-rt.recall)*100:+.1f}pt  F1 {(sc.f1-rt.f1)*100:+.1f}pt  "
          f"| total removed {sum(r['removed'] for r in results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
