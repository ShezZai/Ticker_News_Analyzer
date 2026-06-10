#!/usr/bin/env python3
"""Show which prior insights the TWO-PHASE-SIMILARITY retriever returns for a seed article.

This is an inspection tool for the two-stage ``two-phase-similarity`` hybrid in
``hybrid_retrieval.py`` (a WIDE whole-article net gated by an insight-cosine
threshold, then RRF reranked). Given an article id it runs the retrieval over
the no-lookahead ``--months-before`` window and prints the kept insights in full
-- topic, insight text, tickers, date, RRF score, and the source article -- so
you can eyeball exactly what context the sentiment flow would receive under
``insight_sentiment.py --retrieval two-phase-similarity``.

Usage:
    python show_two_phase_insights.py 4235
    python show_two_phase_insights.py 4235 --budget 30 --tau-insight 0.74
    python show_two_phase_insights.py 4235 --months-before 6 --no-exclusive
    python show_two_phase_insights.py 4235 --order date        # earliest -> latest
    python show_two_phase_insights.py 4235 --json

Connection comes from NEWS_DB_DSN / DATABASE_URL (point at the content-rich DB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_articles_by_insights import _seed_window, get_conn  # noqa: E402
from hybrid_retrieval import (  # noqa: E402
    DEF_K, DEF_MIN_SIM, DEF_NET_K, DEF_RRF_C, DEF_TWO_PHASE_BUDGET,
    DEF_TWO_PHASE_NET_MIN_SIM, DEF_TWO_PHASE_TAU_INS, retrieve,
)

MARKET_TZ = ZoneInfo("America/New_York")


def _fmt_et(dt: Optional[datetime]) -> str:
    return dt.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M") if dt else "(unknown)"


def load_seed(conn, article_id: int) -> dict:
    """Fetch the seed article's headline/date/tickers, or raise."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, published_utc, tickers FROM public.articles WHERE id = %s",
            (article_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"article {article_id} not found in the connected database.")
    return {"id": row[0], "title": row[1], "published_utc": row[2],
            "tickers": list(row[3] or [])}


def fetch_insight_rows(conn, scored) -> List[dict]:
    """Hydrate (insight_id, article_id, score) tuples into full insight dicts."""
    score_by_id = {iid: sc for iid, _aid, sc in scored}
    if not score_by_id:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ai.id, ai.article_id, ai.source_url, ai.article_headline, "
            "       ai.topic, ai.insight, a.tickers, a.published_utc "
            "FROM public.article_insights ai "
            "JOIN public.articles a ON a.id = ai.article_id "
            "WHERE ai.id = ANY(%s)",
            (list(score_by_id),),
        )
        rows = cur.fetchall()
    return [
        {
            "insight_id": r[0], "article_id": r[1], "source_url": r[2],
            "article_headline": r[3], "topic": r[4], "insight": r[5],
            "tickers": list(r[6] or []), "published_utc": r[7],
            "score": float(score_by_id.get(r[0], 0.0)),
        }
        for r in rows
    ]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("article_id", type=int, help="seed article id")
    p.add_argument("--months-before", type=int, default=3,
                   help="only use insights from N months before the seed (default 3)")
    p.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True,
                   help="anchor the window on the seed's exact publish time "
                        "(no same-day lookahead); on by default")
    p.add_argument("-k", "--k", type=int, default=DEF_K,
                   help=f"insight ANN top-k per seed insight (default {DEF_K})")
    p.add_argument("--min-similarity", type=float, default=DEF_MIN_SIM,
                   help=f"insight cosine floor (default {DEF_MIN_SIM})")
    p.add_argument("--net-k", type=int, default=DEF_NET_K,
                   help=f"whole-article net size (default {DEF_NET_K})")
    p.add_argument("--net-min-similarity", type=float, default=DEF_TWO_PHASE_NET_MIN_SIM,
                   help=f"article-level cosine floor for the net "
                        f"(default {DEF_TWO_PHASE_NET_MIN_SIM})")
    p.add_argument("--tau-insight", type=float, default=DEF_TWO_PHASE_TAU_INS,
                   help=f"insight-level cosine-to-seed floor: keep insights >= this "
                        f"(default {DEF_TWO_PHASE_TAU_INS})")
    p.add_argument("--rrf-c", type=int, default=DEF_RRF_C,
                   help=f"reciprocal-rank-fusion constant (default {DEF_RRF_C})")
    p.add_argument("--budget", type=int, default=DEF_TWO_PHASE_BUDGET,
                   help=f"how many reranked insights to keep (default {DEF_TWO_PHASE_BUDGET})")
    p.add_argument("--order", choices=("score", "date"), default="score",
                   help="display order: 'score' (RRF rank, default) or 'date' "
                        "(earliest -> latest)")
    p.add_argument("--json", action="store_true", help="print only the JSON result")
    args = p.parse_args(argv)

    conn = get_conn()
    try:
        seed = load_seed(conn, args.article_id)
        if seed["published_utc"] is None:
            raise SystemExit("seed article has no published_utc to anchor the window.")
        since, until = _seed_window(seed["published_utc"], args.months_before, args.exclusive)

        res = retrieve(
            args.article_id, method="two-phase-similarity", months_before=args.months_before,
            exclusive=args.exclusive, k=args.k, min_sim=args.min_similarity,
            net_k=args.net_k, net_min_sim=args.net_min_similarity,
            tau_ins=args.tau_insight, rrf_c=args.rrf_c, budget=args.budget, conn=conn,
        )
        rows = fetch_insight_rows(conn, res.scored)
    finally:
        conn.close()

    # res.scored is ranked by RRF desc; tag each row with its rank + mutual flag.
    rank_of = {iid: r for r, (iid, _a, _s) in enumerate(res.scored, 1)}
    for row in rows:
        row["rank"] = rank_of.get(row["insight_id"])
        row["mutual"] = row["insight_id"] in res.mutual
    if args.order == "date":
        rows.sort(key=lambda d: (d["published_utc"] is None, d["published_utc"]))
    else:
        rows.sort(key=lambda d: d["rank"])

    if args.json:
        out = {
            "seed_id": seed["id"],
            "published_et": _fmt_et(seed["published_utc"]),
            "window": {"since": since, "until": until, "exclusive": args.exclusive},
            "net_size": res.net_size,
            "kept_insights": len(res.insight_ids),
            "kept_articles": len(res.article_ids),
            "mutual_agreement": len(res.mutual),
            "insights": [
                {k: (_fmt_et(v) if k == "published_utc" else v) for k, v in row.items()}
                for row in rows
            ],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"\nSeed: a#{seed['id']} {_fmt_et(seed['published_utc'])} ET")
    print(f"  {seed['title'] or '(no title)'}")
    tick = ", ".join(seed["tickers"]) if seed["tickers"] else "-"
    print(f"  tickers: {tick}")
    print(f"  window: {since or '(open)'} .. {until} "
          f"({'exclusive ts' if args.exclusive else 'inclusive day'})")
    print(f"\ntwo-phase-similarity retrieval  net={res.net_size}  "
          f"kept {len(res.insight_ids)} insight(s) across {len(res.article_ids)} "
          f"article(s)  ({len(res.mutual)} mutual-agreement)")
    print(f"  knobs: net_k={args.net_k} net_min_sim={args.net_min_similarity} "
          f"tau={args.tau_insight} rrf_c={args.rrf_c} budget={args.budget}\n")

    if not rows:
        print("  (no insights returned in this window)")
        return 0

    for row in rows:
        flag = "*" if row["mutual"] else " "
        print(f"{flag} #{row['rank']:<3} rrf={row['score']:.4f} | "
              f"{_fmt_et(row['published_utc'])} ET | "
              f"[{','.join(row['tickers']) if row['tickers'] else '-'}] | "
              f"ai#{row['insight_id']} a#{row['article_id']}")
        print(f"      TOPIC: {row['topic'] or '(no topic)'}")
        if row["insight"]:
            print(f"      {row['insight']}")
        print(f"      from a#{row['article_id']}: "
              f"{row['article_headline'] or '(no headline)'}")
        print()
    print("(* = mutual agreement: also an insight-ANN hit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
