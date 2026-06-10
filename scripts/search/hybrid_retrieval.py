#!/usr/bin/env python3
"""Hybrid insight retrieval: combine whole-article recall with insight precision.

Validation (see docs/validations/validate_retrieval.md) showed the two retrievers
are complementary at a tight operating point:

    whole-article  (articles.embedding)        -> higher RECALL, lower precision
    insight-level  (article_insights.embedding) -> higher PRECISION, lower recall

A key fact about combining them: whichever method is applied LAST as a hard gate
caps your recall at that method's recall. So to get "best of both", the
high-recall method must be the wide net and the high-precision method the filter.

Three flows are provided, all anchored on a seed article and a no-lookahead
window (since, until):

  intersection   insight top-k  AND  whole-article top-k   (keep insights whose
                 article is in BOTH). Max precision; recall <= insight-alone.
                 This is the naive "keep only the overlap".

  cascade        WIDE whole-article net (large k, low floor) -> score every
                 insight inside the net against the seed's insights, keep those
                 >= tau. Recall ceiling = whole-article (the high one); precision
                 restored by the insight filter. Recommended.

  fusion         Reciprocal-rank fusion over two insight rankings: (1) insight
                 ANN cosine, (2) the rank of the insight's article in the
                 whole-article net (all of an article's insights inherit it).
                 Keep the top `budget`. Highest recall ceiling (a union), precision
                 controlled by the cutoff.

Each flow returns a HybridResult with the kept insight ids, their source article
ids, and the per-insight scores (for ranking / a confidence boost downstream).

Python API:
    from hybrid_retrieval import retrieve
    res = retrieve(seed_id, method="cascade", months_before=3, exclusive=True)
    print(res.insight_ids, res.article_ids)

CLI:
    python hybrid_retrieval.py 4235 --method cascade --months-before 3
    python hybrid_retrieval.py 4235 --method fusion --budget 50

Connection comes from NEWS_DB_DSN / DATABASE_URL (point at the content-rich DB).
Requires OPENAI_API_KEY only if a seed lookup needs re-embedding (it does not here).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from search_articles_by_insights import (  # noqa: E402
    _seed_window, get_conn, insights_of, search_by_insights,
)
from search_articles import similar_to  # noqa: E402

# Defaults (knobs differ per method by design).
DEF_K = 5               # insight ANN: top-k per seed insight
DEF_MIN_SIM = 0.7       # insight cosine floor (precision stage)
DEF_ARTICLE_K = 50      # whole-article top-k for the intersection method
DEF_NET_K = 150         # WIDE whole-article net for cascade / fusion
DEF_NET_MIN_SIM = 0.45  # low floor for the net (recall stage)
DEF_TAU_INS = 0.70      # cascade: keep net insights with max-cos to a seed >= tau
DEF_RRF_C = 60          # fusion: reciprocal-rank-fusion constant
DEF_BUDGET = 50         # fusion: how many insights to keep

METHODS = ("intersection", "cascade", "fusion", "super")


@dataclass
class HybridResult:
    method: str
    seed_id: int
    insight_ids: Set[int]
    article_ids: Set[int]
    # (insight_id, article_id, score) ranked desc; score meaning depends on method
    scored: List[Tuple[int, int, float]] = field(default_factory=list)
    net_size: int = 0          # whole-article net size (cascade/fusion)
    mutual: Set[int] = field(default_factory=set)  # insights both ANN-hit and in-net


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _norm_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _seed_matrix(conn, seed_id: int) -> Tuple[np.ndarray, List]:
    """Return (normalized seed-insight matrix, seed insight list)."""
    seeds = insights_of(conn, seed_id)
    if not seeds:
        return np.zeros((0, 0), dtype=np.float32), []
    mat = np.vstack([np.asarray(s.embedding, dtype=np.float32) for s in seeds])
    return _norm_rows(mat), seeds


def _insights_for_articles(conn, article_ids: Sequence[int]):
    """(insight_id, article_id, embedding) for embedded insights of these articles."""
    if not article_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, article_id, embedding FROM public.article_insights "
            "WHERE article_id = ANY(%s) AND embedding IS NOT NULL",
            (list(article_ids),),
        )
        return cur.fetchall()


def _insight_hits(conn, seed_id, k, min_sim, since, until, exclusive):
    """Flat list of (insight_id, article_id, cosine) from search_by_insights."""
    groups = search_by_insights(
        article_id=seed_id, k=k, since=since, until=until,
        min_similarity=min_sim, until_exclusive=exclusive, conn=conn,
    )
    # keep best cosine per insight_id
    best: Dict[int, Tuple[int, int, float]] = {}
    for g in groups:
        for h in g.hits:
            cur = best.get(h.insight_id)
            if cur is None or h.similarity > cur[2]:
                best[h.insight_id] = (h.insight_id, h.article_id, h.similarity)
    return list(best.values())


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def retrieve_intersection(conn, seed_id, *, since, until, exclusive,
                          k, min_sim, article_k) -> HybridResult:
    hits = _insight_hits(conn, seed_id, k, min_sim, since, until, exclusive)
    whole = similar_to(seed_id, k=article_k, since=since, until=until,
                       min_similarity=min_sim, until_exclusive=exclusive, conn=conn)
    net = {h.id for h in whole}
    kept = [(iid, aid, s) for (iid, aid, s) in hits if aid in net]
    kept.sort(key=lambda t: t[2], reverse=True)
    return HybridResult(
        method="intersection", seed_id=seed_id,
        insight_ids={t[0] for t in kept}, article_ids={t[1] for t in kept},
        scored=kept, net_size=len(net), mutual={t[0] for t in kept},
    )


def retrieve_cascade(conn, seed_id, *, since, until, exclusive,
                     net_k, net_min_sim, tau_ins,
                     k=DEF_K, min_sim=DEF_MIN_SIM) -> HybridResult:
    whole = similar_to(seed_id, k=net_k, since=since, until=until,
                       min_similarity=net_min_sim, until_exclusive=exclusive, conn=conn)
    net_ids = [h.id for h in whole]
    S, seeds = _seed_matrix(conn, seed_id)
    if not net_ids or S.shape[0] == 0:
        return HybridResult("cascade", seed_id, set(), set(), [], len(net_ids))

    rows = _insights_for_articles(conn, net_ids)
    if not rows:
        return HybridResult("cascade", seed_id, set(), set(), [], len(net_ids))
    iids = [r[0] for r in rows]
    aids = [r[1] for r in rows]
    E = _norm_rows(np.vstack([np.asarray(r[2], dtype=np.float32) for r in rows]))
    # max cosine of each candidate insight to ANY seed insight
    scores = (E @ S.T).max(axis=1)

    # which insights were ALSO insight-ANN hits (mutual agreement)
    ann = {iid for iid, _aid, _s in _insight_hits(conn, seed_id, k, min_sim,
                                                  since, until, exclusive)}
    kept = [(iids[i], aids[i], float(scores[i]))
            for i in range(len(iids)) if scores[i] >= tau_ins]
    kept.sort(key=lambda t: t[2], reverse=True)
    return HybridResult(
        method="cascade", seed_id=seed_id,
        insight_ids={t[0] for t in kept}, article_ids={t[1] for t in kept},
        scored=kept, net_size=len(net_ids),
        mutual={t[0] for t in kept if t[0] in ann},
    )


def retrieve_fusion(conn, seed_id, *, since, until, exclusive,
                    k, min_sim, net_k, net_min_sim, rrf_c, budget) -> HybridResult:
    # ranking 1: insight ANN cosine
    hits = _insight_hits(conn, seed_id, k, min_sim, since, until, exclusive)
    hits.sort(key=lambda t: t[2], reverse=True)
    rank1 = {iid: r for r, (iid, _a, _s) in enumerate(hits, 1)}
    art_of = {iid: aid for iid, aid, _s in hits}

    # ranking 2: whole-article net rank, inherited by every insight of the article
    whole = similar_to(seed_id, k=net_k, since=since, until=until,
                       min_similarity=net_min_sim, until_exclusive=exclusive, conn=conn)
    art_rank = {h.id: r for r, h in enumerate(whole, 1)}
    net_rows = _insights_for_articles(conn, list(art_rank))
    rank2: Dict[int, int] = {}
    for iid, aid, _emb in net_rows:
        rank2[iid] = art_rank[aid]
        art_of.setdefault(iid, aid)

    # reciprocal-rank fusion over the union of candidate insights
    candidates = set(rank1) | set(rank2)
    scored = []
    for iid in candidates:
        s = 0.0
        if iid in rank1:
            s += 1.0 / (rrf_c + rank1[iid])
        if iid in rank2:
            s += 1.0 / (rrf_c + rank2[iid])
        scored.append((iid, art_of[iid], s))
    scored.sort(key=lambda t: t[2], reverse=True)
    kept = scored[:budget]
    return HybridResult(
        method="fusion", seed_id=seed_id,
        insight_ids={t[0] for t in kept}, article_ids={t[1] for t in kept},
        scored=kept, net_size=len(art_rank),
        mutual={iid for iid in (t[0] for t in kept) if iid in rank1 and iid in rank2},
    )


def retrieve_super(conn, seed_id, *, since, until, exclusive,
                   net_k, net_min_sim, tau_ins, rrf_c, budget,
                   k=DEF_K, min_sim=DEF_MIN_SIM) -> HybridResult:
    """Two-stage 'super-hybrid':

      Stage 1 (cascade pool): a WIDE whole-article net -> keep the articles that
        have at least one insight with max-cosine-to-seed >= tau. This is the
        high-recall article candidate POOL (cascade's recall, ~50-60%).
      Stage 2 (fusion within the pool): reciprocal-rank fusion of (a) each pool
        insight's cosine-to-seed rank and (b) its article's rank in the net,
        keeping the top `budget`. Restores precision and caps volume.

    Recall ceiling = the cascade pool (high); precision = fusion's rerank + budget.
    """
    whole = similar_to(seed_id, k=net_k, since=since, until=until,
                       min_similarity=net_min_sim, until_exclusive=exclusive, conn=conn)
    art_rank = {h.id: r for r, h in enumerate(whole, 1)}
    net_ids = list(art_rank)
    S, _seeds = _seed_matrix(conn, seed_id)
    if not net_ids or S.shape[0] == 0:
        return HybridResult("super", seed_id, set(), set(), [], len(net_ids))

    rows = _insights_for_articles(conn, net_ids)
    if not rows:
        return HybridResult("super", seed_id, set(), set(), [], len(net_ids))
    iids = [r[0] for r in rows]
    aids = [r[1] for r in rows]
    E = _norm_rows(np.vstack([np.asarray(r[2], dtype=np.float32) for r in rows]))
    icos = (E @ S.T).max(axis=1)  # each net insight's max cosine to a seed insight

    # Stage 1 pool: articles with >= 1 insight clearing tau (cascade's article set)
    pool = {aids[i] for i in range(len(iids)) if icos[i] >= tau_ins}
    if not pool:
        return HybridResult("super", seed_id, set(), set(), [], len(net_ids))

    # Stage 2: fusion over ALL insights of the pool articles
    cand = [(iids[i], aids[i], float(icos[i])) for i in range(len(iids)) if aids[i] in pool]
    order = sorted(range(len(cand)), key=lambda j: cand[j][2], reverse=True)
    rank1 = {cand[j][0]: r for r, j in enumerate(order, 1)}  # insight-cosine rank
    ann = {iid for iid, _a, _s in _insight_hits(conn, seed_id, k, min_sim,
                                                since, until, exclusive)}
    scored = []
    for iid, aid, _ic in cand:
        s = 1.0 / (rrf_c + rank1[iid]) + 1.0 / (rrf_c + art_rank[aid])
        scored.append((iid, aid, s))
    scored.sort(key=lambda t: t[2], reverse=True)
    kept = scored[:budget]
    return HybridResult(
        method="super", seed_id=seed_id,
        insight_ids={t[0] for t in kept}, article_ids={t[1] for t in kept},
        scored=kept, net_size=len(net_ids),
        mutual={iid for iid, _a, _s in kept if iid in ann},
    )


def retrieve(seed_id: int, method: str = "cascade", *,
             months_before: Optional[int] = 3, exclusive: bool = True,
             k: int = DEF_K, min_sim: float = DEF_MIN_SIM,
             article_k: int = DEF_ARTICLE_K, net_k: int = DEF_NET_K,
             net_min_sim: float = DEF_NET_MIN_SIM, tau_ins: float = DEF_TAU_INS,
             rrf_c: int = DEF_RRF_C, budget: int = DEF_BUDGET,
             conn=None) -> HybridResult:
    """Dispatch to a hybrid method. Resolves the no-lookahead window from the seed."""
    own = conn is None
    conn = conn or get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT published_utc FROM public.articles WHERE id = %s", (seed_id,))
            row = cur.fetchone()
        if not row or row[0] is None:
            raise SystemExit(f"article {seed_id} not found / has no published_utc")
        since, until = _seed_window(row[0], months_before, exclusive)
        if method == "intersection":
            return retrieve_intersection(conn, seed_id, since=since, until=until,
                                         exclusive=exclusive, k=k, min_sim=min_sim,
                                         article_k=article_k)
        if method == "cascade":
            return retrieve_cascade(conn, seed_id, since=since, until=until,
                                    exclusive=exclusive, net_k=net_k,
                                    net_min_sim=net_min_sim, tau_ins=tau_ins,
                                    k=k, min_sim=min_sim)
        if method == "fusion":
            return retrieve_fusion(conn, seed_id, since=since, until=until,
                                   exclusive=exclusive, k=k, min_sim=min_sim,
                                   net_k=net_k, net_min_sim=net_min_sim,
                                   rrf_c=rrf_c, budget=budget)
        if method == "super":
            return retrieve_super(conn, seed_id, since=since, until=until,
                                  exclusive=exclusive, net_k=net_k,
                                  net_min_sim=net_min_sim, tau_ins=tau_ins,
                                  rrf_c=rrf_c, budget=budget, k=k, min_sim=min_sim)
        raise SystemExit(f"unknown method {method!r}; choose from {METHODS}")
    finally:
        if own:
            conn.close()


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seed_id", type=int, help="seed article id")
    p.add_argument("--method", choices=METHODS, default="cascade")
    p.add_argument("--months-before", type=int, default=3)
    p.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("-k", "--k", type=int, default=DEF_K)
    p.add_argument("--min-similarity", type=float, default=DEF_MIN_SIM)
    p.add_argument("--article-k", type=int, default=DEF_ARTICLE_K)
    p.add_argument("--net-k", type=int, default=DEF_NET_K)
    p.add_argument("--net-min-similarity", type=float, default=DEF_NET_MIN_SIM)
    p.add_argument("--tau-insight", type=float, default=DEF_TAU_INS)
    p.add_argument("--rrf-c", type=int, default=DEF_RRF_C)
    p.add_argument("--budget", type=int, default=DEF_BUDGET)
    args = p.parse_args(argv)

    res = retrieve(
        args.seed_id, method=args.method, months_before=args.months_before,
        exclusive=args.exclusive, k=args.k, min_sim=args.min_similarity,
        article_k=args.article_k, net_k=args.net_k, net_min_sim=args.net_min_similarity,
        tau_ins=args.tau_insight, rrf_c=args.rrf_c, budget=args.budget,
    )
    print(f"method={res.method}  seed=a#{res.seed_id}  net={res.net_size}  "
          f"kept {len(res.insight_ids)} insight(s) across {len(res.article_ids)} "
          f"article(s)  ({len(res.mutual)} mutual-agreement)")
    for iid, aid, score in res.scored[:30]:
        tag = "*" if iid in res.mutual else " "
        print(f"  {tag} ai#{iid:<7} a#{aid:<7} score={score:.4f}")
    if len(res.scored) > 30:
        print(f"  … and {len(res.scored) - 30} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
