#!/usr/bin/env python3
"""Validate retrieval quality against a hand-labelled cluster CSV.

``article_clusters.csv`` groups articles into real-world EVENTS. For each event it
records, among other columns:

    cluster theme                          -- the event name
    article ids                            -- every article in the cluster
    news_articles_ids                      -- the "news" subset
    later_news_article_ids                 -- the most-recent news article(s):
                                              we use the latest as the QUERY SEED
    all_less_than_3_months_before_later    -- every cluster article published in
                                              the 3 months before the seed: the
                                              GROUND TRUTH of what retrieval, run
                                              from the seed with a 3-month
                                              no-lookahead window, SHOULD surface
    news_less_than_3_months_before_later   -- the news subset of the above

For each usable cluster we:
  1. Take the latest ``later_news_article_ids`` article as the query seed.
  2. Run retrieval anchored on the seed over the no-lookahead window
     (``--months-before`` / ``--exclusive``), at both granularities:
       * INSIGHT level  -- search_by_insights(seed): top-k similar insights per
         seed insight, excluding the seed's own.
       * ARTICLE level  -- two methods:
           - insight-overlap: the distinct source articles of those insight hits
             (the pipeline's real article retriever, via related_articles).
           - whole-article : similar_to(seed) over articles.embedding.
  3. Compare retrieved vs ground-truth and compute precision / recall / accuracy
     (+ F1) at each level.

Ground truth:
  * article level   -- ``all_…`` (or ``news_…`` with --ground-truth news), minus
    the seed itself.
  * insight level   -- every embedded insight belonging to a ground-truth article
    ("from the should-be-retrieved articles").

Metrics (per level, with relevant set G, retrieved set R, retrievable universe U):
    precision = |R∩G| / |R|
    recall    = |R∩G| / |G|                  (raw, vs the full labelled G)
    recall*   = |R∩G| / |G∩U|                (achievable, vs the G that is even
                                              retrievable -- has embeddings/insights
                                              and is in the window)
    accuracy  = (TP+TN) / |U|,  TN = |U| - |R ∪ (G∩U)|
    f1        = harmonic mean of precision and recall

Aggregates are reported both micro (pool TP/FP/FN/TN across clusters) and macro
(mean of per-cluster metrics).

Usage:
    python validate_retrieval.py article_clusters.csv
    python validate_retrieval.py article_clusters.csv -k 5 --min-similarity 0.7
    python validate_retrieval.py article_clusters.csv --ground-truth news
    python validate_retrieval.py article_clusters.csv --json report.json --csv per_cluster.csv

Requires OPENAI_API_KEY (query embedding) and NEWS_DB_DSN / DATABASE_URL pointed at
the content-rich DB.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "search"))

from search_articles_by_insights import (  # noqa: E402
    _seed_window, get_conn, insights_of, related_articles, search_by_insights,
)
from search_articles import similar_to  # noqa: E402

DEFAULT_K = 5
DEFAULT_ARTICLE_K = 50          # flat top-k for the whole-article method
DEFAULT_MIN_SIM = 0.7
DEFAULT_MONTHS_BEFORE = 3


# --------------------------------------------------------------------------- #
# CSV parsing
# --------------------------------------------------------------------------- #
def _ids(cell: str) -> List[int]:
    return [int(x) for x in (cell or "").split() if x.strip().isdigit()]


@dataclass
class Cluster:
    theme: str
    all_ids: List[int]
    news_ids: List[int]
    later_ids: List[int]
    all_before: List[int]      # all_less_than_3_months_before_later
    news_before: List[int]     # news_less_than_3_months_before_later


def load_clusters(path: str) -> List[Cluster]:
    out: List[Cluster] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Cluster(
                theme=row["cluster theme"].strip(),
                all_ids=_ids(row.get("article ids", "")),
                news_ids=_ids(row.get("news_articles_ids", "")),
                later_ids=_ids(row.get("later_news_article_ids", "")),
                all_before=_ids(row.get("all_less_than_3_months_before_later", "")),
                news_before=_ids(row.get("news_less_than_3_months_before_later", "")),
            ))
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    label: str
    n_relevant: int          # |G|
    n_retrievable: int       # |G ∩ U|
    n_retrieved: int         # |R|
    tp: int
    fp: int
    fn: int                  # vs retrievable G (so tp+fn == retrievable)
    universe: int            # |U|
    precision: float
    recall: float            # vs full G
    recall_achievable: float # vs retrievable G
    accuracy: float
    f1: float


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def compute(label: str, retrieved: Set[int], relevant_full: Set[int],
            retrievable: Set[int], universe: int) -> Metrics:
    """retrieved ⊆ U; retrievable = relevant ∩ U (relevant that can be returned)."""
    tp = len(retrieved & retrievable)
    fp = len(retrieved - retrievable)
    fn = len(retrievable - retrieved)
    tn = max(universe - len(retrieved | retrievable), 0)
    precision = _safe_div(tp, len(retrieved))
    recall = _safe_div(tp, len(relevant_full))
    recall_ach = _safe_div(tp, len(retrievable))
    accuracy = _safe_div(tp + tn, universe)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (
        not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0
    ) else float("nan")
    return Metrics(
        label=label, n_relevant=len(relevant_full), n_retrievable=len(retrievable),
        n_retrieved=len(retrieved), tp=tp, fp=fp, fn=fn, universe=universe,
        precision=precision, recall=recall, recall_achievable=recall_ach,
        accuracy=accuracy, f1=f1,
    )


# --------------------------------------------------------------------------- #
# DB helpers
# --------------------------------------------------------------------------- #
def article_facts(conn, ids: Sequence[int]) -> Dict[int, dict]:
    """For each id: published_utc, has_embedding, has_insights."""
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.id, a.published_utc, (a.embedding IS NOT NULL), "
            "       EXISTS (SELECT 1 FROM public.article_insights ai "
            "               WHERE ai.article_id = a.id AND ai.embedding IS NOT NULL) "
            "FROM public.articles a WHERE a.id = ANY(%s)",
            (list(ids),),
        )
        return {r[0]: {"published_utc": r[1], "has_embedding": r[2], "has_insights": r[3]}
                for r in cur.fetchall()}


def ground_truth_insight_ids(conn, article_ids: Sequence[int]) -> Set[int]:
    """Embedded insight ids belonging to the ground-truth articles."""
    if not article_ids:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM public.article_insights "
            "WHERE article_id = ANY(%s) AND embedding IS NOT NULL",
            (list(article_ids),),
        )
        return {r[0] for r in cur.fetchall()}


def window_universes(conn, since: Optional[str], until: str) -> Tuple[int, int, int]:
    """(embedded articles, articles-with-insights, embedded insights) in [since, until)."""
    since_clause = "AND a.published_utc >= %s" if since else ""
    base_params = ([since, until] if since else [until])
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM public.articles a "
            f"WHERE a.embedding IS NOT NULL AND a.published_utc < %s {since_clause}",
            ([until, since] if since else [until]),
        )
        u_art_emb = cur.fetchone()[0]
        cur.execute(
            f"SELECT count(DISTINCT a.id) FROM public.articles a "
            f"JOIN public.article_insights ai ON ai.article_id = a.id "
            f"WHERE ai.embedding IS NOT NULL AND a.published_utc < %s {since_clause}",
            ([until, since] if since else [until]),
        )
        u_art_ins = cur.fetchone()[0]
        cur.execute(
            f"SELECT count(*) FROM public.article_insights ai "
            f"JOIN public.articles a ON a.id = ai.article_id "
            f"WHERE ai.embedding IS NOT NULL AND a.published_utc < %s {since_clause}",
            ([until, since] if since else [until]),
        )
        u_ins = cur.fetchone()[0]
    return u_art_emb, u_art_ins, u_ins


# --------------------------------------------------------------------------- #
# Per-cluster evaluation
# --------------------------------------------------------------------------- #
@dataclass
class ClusterResult:
    theme: str
    seed_id: int
    seed_published: str
    window_since: Optional[str]
    window_until: str
    article_insight_overlap: Metrics
    article_whole: Metrics
    insight: Metrics


def evaluate_cluster(conn, c: Cluster, args) -> Optional[ClusterResult]:
    if not c.later_ids:
        return None  # e.g. class-action clusters: no "later news" seed

    facts = article_facts(conn, set(c.later_ids) | set(c.all_before) | set(c.news_before))

    # seed = the latest-published of the later_news_article_ids that exists
    later_known = [i for i in c.later_ids if i in facts and facts[i]["published_utc"]]
    if not later_known:
        return None
    seed = max(later_known, key=lambda i: facts[i]["published_utc"])
    seed_pub = facts[seed]["published_utc"]

    since, until = _seed_window(seed_pub, args.months_before, args.exclusive)

    # ground-truth article set (minus all the later/seed ids themselves)
    gt_col = c.news_before if args.ground_truth == "news" else c.all_before
    relevant_articles = (set(gt_col) - set(c.later_ids))
    relevant_articles.discard(seed)

    # backfill facts for any GT ids not yet fetched
    missing = [i for i in relevant_articles if i not in facts]
    if missing:
        facts.update(article_facts(conn, missing))

    # retrievable GT (in DB, in window, with the needed vectors)
    def _in_window(i: int) -> bool:
        f = facts.get(i)
        if not f or not f["published_utc"]:
            return False
        p = f["published_utc"]
        if since and p.isoformat() < since:
            return False
        return p < seed_pub  # strictly before the seed (no lookahead)

    retr_for_insights = {i for i in relevant_articles
                         if _in_window(i) and facts[i]["has_insights"]}
    retr_for_whole = {i for i in relevant_articles
                      if _in_window(i) and facts[i]["has_embedding"]}

    relevant_insights = ground_truth_insight_ids(conn, retr_for_insights)

    u_art_emb, u_art_ins, u_ins = window_universes(conn, since, until)

    # ---- INSIGHT-level retrieval (also feeds insight-overlap article method) ----
    groups = search_by_insights(
        article_id=seed, k=args.k, since=since, until=until,
        min_similarity=args.min_similarity, until_exclusive=args.exclusive, conn=conn,
    )
    retrieved_insights: Set[int] = set()
    retrieved_articles_overlap: Set[int] = set()
    for g in groups:
        for h in g.hits:
            retrieved_insights.add(h.insight_id)
            retrieved_articles_overlap.add(h.article_id)

    if args.top_articles:  # optional cap, ranked by related_articles consolidation
        ranked = related_articles(
            article_id=seed, k=args.top_articles, per_insight_k=args.k,
            since=since, until=until, min_similarity=args.min_similarity,
            until_exclusive=args.exclusive, conn=conn,
        )
        retrieved_articles_overlap = {ra.article_id for ra in ranked}

    # ---- WHOLE-ARTICLE retrieval ----
    whole_hits = similar_to(
        seed, k=args.article_k, since=since, until=until,
        min_similarity=args.min_similarity, until_exclusive=args.exclusive, conn=conn,
    )
    retrieved_articles_whole = {h.id for h in whole_hits}

    return ClusterResult(
        theme=c.theme, seed_id=seed,
        seed_published=seed_pub.isoformat(),
        window_since=since, window_until=until,
        article_insight_overlap=compute(
            "article/insight-overlap", retrieved_articles_overlap,
            relevant_articles, retr_for_insights, u_art_ins,
        ),
        article_whole=compute(
            "article/whole-article", retrieved_articles_whole,
            relevant_articles, retr_for_whole, u_art_emb,
        ),
        insight=compute(
            "insight", retrieved_insights, relevant_insights, relevant_insights, u_ins,
        ),
    )


# --------------------------------------------------------------------------- #
# Aggregation + reporting
# --------------------------------------------------------------------------- #
def micro(metrics: List[Metrics], label: str) -> Metrics:
    tp = sum(m.tp for m in metrics)
    fp = sum(m.fp for m in metrics)
    fn = sum(m.fn for m in metrics)
    universe = sum(m.universe for m in metrics)
    n_rel = sum(m.n_relevant for m in metrics)
    n_retr = sum(m.n_retrieved for m in metrics)
    n_able = sum(m.n_retrievable for m in metrics)
    tn = max(universe - (tp + fp + fn), 0)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, n_rel)
    recall_ach = _safe_div(tp, n_able)
    accuracy = _safe_div(tp + tn, universe)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else float("nan")
    return Metrics(label, n_rel, n_able, n_retr, tp, fp, fn, universe,
                   precision, recall, recall_ach, accuracy, f1)


def macro(metrics: List[Metrics], label: str) -> Metrics:
    def avg(attr):
        vals = [getattr(m, attr) for m in metrics if not math.isnan(getattr(m, attr))]
        return sum(vals) / len(vals) if vals else float("nan")
    return Metrics(label, 0, 0, 0, 0, 0, 0, 0,
                   avg("precision"), avg("recall"), avg("recall_achievable"),
                   avg("accuracy"), avg("f1"))


def _pct(x: float) -> str:
    return "   n/a" if math.isnan(x) else f"{x*100:5.1f}%"


def print_report(results: List[ClusterResult], args) -> None:
    W = 112
    print("\n" + "=" * W)
    print(f"  RETRIEVAL VALIDATION  |  {len(results)} cluster(s)  |  k={args.k} "
          f"article_k={args.article_k}  min_sim={args.min_similarity}  "
          f"months_before={args.months_before}  exclusive={args.exclusive}  "
          f"ground_truth={args.ground_truth}")
    print("=" * W)

    for level, attr in [
        ("ARTICLE LEVEL — insight-overlap retrieval", "article_insight_overlap"),
        ("ARTICLE LEVEL — whole-article retrieval", "article_whole"),
        ("INSIGHT LEVEL — search_by_insights", "insight"),
    ]:
        print(f"\n  {level}")
        print(f"  {'event':<46} {'|G|':>4} {'ret':>4} {'TP':>4} "
              f"{'prec':>6} {'recall':>7} {'rec*':>6} {'acc':>6} {'F1':>6}")
        print("  " + "-" * (W - 2))
        ms = [getattr(r, attr) for r in results]
        for r in results:
            m = getattr(r, attr)
            print(f"  {r.theme[:45]:<46} {m.n_relevant:>4} {m.n_retrieved:>4} "
                  f"{m.tp:>4} {_pct(m.precision)} {_pct(m.recall)} "
                  f"{_pct(m.recall_achievable)} {_pct(m.accuracy)} {_pct(m.f1)}")
        mi = micro(ms, "micro")
        ma = macro(ms, "macro")
        print("  " + "-" * (W - 2))
        print(f"  {'MICRO (pooled TP/FP/FN)':<46} {mi.n_relevant:>4} {mi.n_retrieved:>4} "
              f"{mi.tp:>4} {_pct(mi.precision)} {_pct(mi.recall)} "
              f"{_pct(mi.recall_achievable)} {_pct(mi.accuracy)} {_pct(mi.f1)}")
        print(f"  {'MACRO (mean of clusters)':<46} {'':>4} {'':>4} "
              f"{'':>4} {_pct(ma.precision)} {_pct(ma.recall)} "
              f"{_pct(ma.recall_achievable)} {_pct(ma.accuracy)} {_pct(ma.f1)}")
    print("\n  rec* = achievable recall (vs ground-truth that is in-window and embedded).")
    print("  accuracy is dominated by true negatives over the window universe; "
          "precision / recall / F1 are the discriminating metrics.")
    print("=" * W)


def write_json(path: str, results: List[ClusterResult], args) -> None:
    payload = {
        "params": {"k": args.k, "article_k": args.article_k,
                   "min_similarity": args.min_similarity,
                   "months_before": args.months_before, "exclusive": args.exclusive,
                   "ground_truth": args.ground_truth, "top_articles": args.top_articles},
        "clusters": [
            {**{kk: vv for kk, vv in asdict(r).items()
                if kk not in ("article_insight_overlap", "article_whole", "insight")},
             "article_insight_overlap": asdict(r.article_insight_overlap),
             "article_whole": asdict(r.article_whole),
             "insight": asdict(r.insight)}
            for r in results
        ],
        "aggregates": {
            attr: {"micro": asdict(micro([getattr(r, attr) for r in results], "micro")),
                   "macro": asdict(macro([getattr(r, attr) for r in results], "macro"))}
            for attr in ("article_insight_overlap", "article_whole", "insight")
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def write_csv(path: str, results: List[ClusterResult]) -> None:
    fields = ["theme", "seed_id", "level", "n_relevant", "n_retrievable",
              "n_retrieved", "tp", "fp", "fn", "precision", "recall",
              "recall_achievable", "accuracy", "f1"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            for attr in ("article_insight_overlap", "article_whole", "insight"):
                m = getattr(r, attr)
                w.writerow({"theme": r.theme, "seed_id": r.seed_id, "level": m.label,
                            "n_relevant": m.n_relevant, "n_retrievable": m.n_retrievable,
                            "n_retrieved": m.n_retrieved, "tp": m.tp, "fp": m.fp, "fn": m.fn,
                            "precision": round(m.precision, 4) if not math.isnan(m.precision) else "",
                            "recall": round(m.recall, 4) if not math.isnan(m.recall) else "",
                            "recall_achievable": round(m.recall_achievable, 4) if not math.isnan(m.recall_achievable) else "",
                            "accuracy": round(m.accuracy, 4) if not math.isnan(m.accuracy) else "",
                            "f1": round(m.f1, 4) if not math.isnan(m.f1) else ""})


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="article_clusters.csv")
    p.add_argument("-k", "--k", type=int, default=DEFAULT_K,
                   help=f"top-k similar insights per seed insight (default {DEFAULT_K})")
    p.add_argument("--article-k", type=int, default=DEFAULT_ARTICLE_K,
                   help=f"top-k for the whole-article method (default {DEFAULT_ARTICLE_K})")
    p.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIM,
                   help=f"cosine floor for a match (default {DEFAULT_MIN_SIM})")
    p.add_argument("--months-before", type=int, default=DEFAULT_MONTHS_BEFORE,
                   help=f"no-lookahead window length (default {DEFAULT_MONTHS_BEFORE}; "
                        "the CSV ground-truth columns are built for 3)")
    p.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True,
                   help="strict pre-seed window on the exact timestamp (default on)")
    p.add_argument("--ground-truth", choices=["all", "news"], default="all",
                   help="which CSV column is the relevant set (default all)")
    p.add_argument("--top-articles", type=int, default=None,
                   help="cap the insight-overlap article set to the top-N consolidated "
                        "(default: all source articles of the retrieved insights)")
    p.add_argument("--json", default=None, help="write a JSON report to this path")
    p.add_argument("--csv-out", default=None, help="write a per-cluster CSV to this path")
    args = p.parse_args(argv)

    clusters = load_clusters(args.csv)
    conn = get_conn()
    results: List[ClusterResult] = []
    skipped: List[str] = []
    try:
        for c in clusters:
            res = evaluate_cluster(conn, c, args)
            if res is None:
                skipped.append(c.theme)
                print(f"  skip (no later-news seed): {c.theme}", file=sys.stderr)
                continue
            results.append(res)
            print(f"  done: {c.theme}  (seed a#{res.seed_id} @ {res.seed_published})",
                  file=sys.stderr)
    finally:
        conn.close()

    if not results:
        print("No usable clusters (none had a later_news_article_ids seed).", file=sys.stderr)
        return 1

    print_report(results, args)
    if args.json:
        write_json(args.json, results, args)
        print(f"\n  JSON report -> {args.json}")
    if args.csv_out:
        write_csv(args.csv_out, results)
        print(f"  Per-cluster CSV -> {args.csv_out}")
    if skipped:
        print(f"\n  Skipped {len(skipped)} cluster(s) without a seed "
              f"(e.g. class-action lists): {', '.join(skipped[:4])}"
              f"{' …' if len(skipped) > 4 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
