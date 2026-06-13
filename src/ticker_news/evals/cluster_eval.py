"""Retrieval precision/recall eval against the labelled article-cluster table.

Each row of ``docs/article_clusters.csv`` is a coherent news event: a ``later
act`` (real-news seed) article plus the hand-curated member articles that should
be surfaced as precedents when querying backward from the seed. We run the
production insight-precedent retrieval (``stages._ranked_insight_hits``) from the
seed and score the retrieved *articles* against the member lists.

Hit semantics are article-level: a member counts as retrieved if **any** of its
boxes is returned. Two corpora are compared:

- ``insight_orig``                          — ``public.article_insights``
- ``insights_distilled_two_pass``           — ``public.distilled_article_insights``
- ``insights_distilled_two_pass_drop_filtered`` — same, excluding ``second_label='DROP'``

Each is scored both against its *own* member column and, apples-to-apples,
against the **overlap** set (articles present in both corpora, so retrievable by
every method — the cleanest recall comparison).

CSV columns used (see ``docs/precedent-source-options.md`` for provenance):
- ``later act articles_ids``                                          — the seed
- ``article ids with insights - all_less_than_3_months_before_later`` — insights members
- ``article ids with insights distilled - two pass - ...``            — distilled members
- ``articles in both (insights ∩ distilled)``                         — overlap
- ``retrieval lookback days``                                         — per-cluster window

Run:
    ticker-news eval clusters --csv docs/article_clusters.csv --out docs/cluster_retrieval_eval.md
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass

from ticker_news.service import stages

# (display name, INSIGHT_SOURCES key, expected member column: "insights"|"distilled")
# Hybrid methods (source key ending in '-hybrid') run an article-level ANN
# prefilter at `hybrid_threshold`, then the insight-box ANN within that subset.
METHODS: list[tuple[str, str, str]] = [
    ("insight_orig", "insights", "insights"),
    ("insights_distilled_two_pass", "distilled-first", "distilled"),
    ("insights_distilled_two_pass_drop_filtered", "distilled-second", "distilled"),
    ("insights_hybrid", "insights-hybrid", "insights"),
    ("distilled_hybrid", "distilled-hybrid", "distilled"),
    ("distilled_hybrid_one_pass", "distilled-hybrid-one-pass", "distilled"),
]

# CSV header fragments (en-dash, matching the table) -> our keys.
_COL = {
    "theme": "cluster theme",
    "insights": "article ids with insights – all_less_than_3_months_before_later",
    "distilled": "article ids with insights distilled – two pass – all_less_than_3_months_before_later",
    "seed": "later act articles_ids",
    "lookback": "retrieval lookback days",
    "overlap": "articles in both (insights ∩ distilled)",
}


def _ids(cell: str | None) -> set[int]:
    return {int(tok) for tok in (cell or "").split() if tok.strip().lstrip("-").isdigit()}


@dataclass
class Cluster:
    theme: str
    seed: int
    lookback_days: int
    insights: set[int]
    distilled: set[int]
    overlap: set[int]

    def expected(self, kind: str) -> set[int]:
        return self.insights if kind == "insights" else self.distilled


def load_clusters(csv_path: str) -> list[Cluster]:
    out: list[Cluster] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if not (row.get(_COL["theme"]) or "").strip():
                continue
            seed_ids = _ids(row[_COL["seed"]])
            if not seed_ids:
                continue
            out.append(Cluster(
                theme=row[_COL["theme"]].strip(),
                seed=sorted(seed_ids)[0],
                lookback_days=int(row[_COL["lookback"]]),
                insights=_ids(row[_COL["insights"]]),
                distilled=_ids(row[_COL["distilled"]]),
                overlap=_ids(row[_COL["overlap"]]),
            ))
    return out


def retrieved_article_ids(conn, seed: int, source_key: str, *, threshold: float,
                          hybrid_article_threshold: float, hybrid_box_threshold: float,
                          limit: int, lookback_days: int) -> set[int]:
    """Distinct article ids retrieved from the seed under `source_key` (>=1 box).

    Hybrid sources use `hybrid_article_threshold` for the article-level prefilter
    and `hybrid_box_threshold` for the insight-box step; others use `threshold`.
    """
    src = stages.INSIGHT_SOURCES[source_key]
    box_thr = hybrid_box_threshold if src.hybrid else threshold
    art_thr = hybrid_article_threshold if src.hybrid else None
    ranked = stages.ranked_precedent_hits(
        conn, seed, src, threshold=box_thr, limit=limit, lookback_days=lookback_days,
        real_news_only=False, article_threshold=art_thr,
    )
    return {aid for _sim, aid, *_ in ranked}


def _metrics(expected: set[int], retrieved: set[int]) -> dict:
    TP = len(retrieved & expected)
    FP = len(retrieved - expected)
    FN = len(expected - retrieved)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(exp=len(expected), ret=len(retrieved), TP=TP, FP=FP, FN=FN,
                prec=P, rec=R, f1=F)


def evaluate(conn, clusters: list[Cluster], *, threshold: float,
             hybrid_article_threshold: float, hybrid_box_threshold: float,
             limit: int, lookback_plus: int) -> list[dict]:
    """Per-cluster metrics for every method, scored vs own column and vs overlap."""
    results = []
    for c in clusters:
        lb = c.lookback_days + lookback_plus
        retrieved = {
            name: retrieved_article_ids(
                conn, c.seed, key, threshold=threshold,
                hybrid_article_threshold=hybrid_article_threshold,
                hybrid_box_threshold=hybrid_box_threshold,
                limit=limit, lookback_days=lb)
            for name, key, _kind in METHODS
        }
        own = {name: _metrics(c.expected(kind), retrieved[name])
               for name, _key, kind in METHODS}
        aa = {name: _metrics(c.overlap, retrieved[name]) for name, _key, _kind in METHODS}
        results.append(dict(cluster=c, lookback=lb, own=own, aa=aa))
    return results


def _agg(results: list[dict], key: str, name: str) -> dict:
    rows = [r[key][name] for r in results]
    TP = sum(x["TP"] for x in rows); FP = sum(x["FP"] for x in rows); FN = sum(x["FN"] for x in rows)
    n = len(rows) or 1
    microP = TP / (TP + FP) if TP + FP else 0.0
    microR = TP / (TP + FN) if TP + FN else 0.0
    microF = 2 * microP * microR / (microP + microR) if microP + microR else 0.0
    return dict(microP=microP, microR=microR, microF=microF,
                macroP=sum(x["prec"] for x in rows) / n,
                macroR=sum(x["rec"] for x in rows) / n,
                macroF=sum(x["f1"] for x in rows) / n,
                TP=TP, FP=FP, FN=FN)


def format_report(results: list[dict], *, threshold: float,
                  hybrid_article_threshold: float, hybrid_box_threshold: float,
                  limit: int, lookback_plus: int) -> str:
    L = ["# Cluster retrieval evaluation\n",
         f"Seed = each cluster's `later act` real-news article. lookback = cluster column "
         f"**+{lookback_plus}d**. threshold {threshold} (hybrid: article {hybrid_article_threshold} "
         f"→ box {hybrid_box_threshold}), limit {limit}, all categories.",
         "Retrieval core = `stages.ranked_precedent_hits`; article-level hits (a target is hit "
         "if ≥1 of its boxes is retrieved). `*_hybrid` methods prefilter to an article-level "
         "ANN candidate subset, then run the insight-box ANN within it.",
         "`insight_orig` / `insights_hybrid` scored vs the article-insights column; distilled "
         "variants vs the distilled column.\n"]

    def section(title: str, key: str) -> None:
        L.append(f"\n# {title}\n")
        for name, _k, _kind in METHODS:
            L.append(f"\n## {name}\n")
            L.append("| cluster | seed | lookback | expected | retrieved | TP | FP | FN | "
                     "precision | recall | F1 |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in results:
                p = r[key][name]
                L.append(f"| {r['cluster'].theme} | {r['cluster'].seed} | {r['lookback']}d | "
                         f"{p['exp']} | {p['ret']} | {p['TP']} | {p['FP']} | {p['FN']} | "
                         f"{p['prec']:.2f} | {p['rec']:.2f} | {p['f1']:.2f} |")
            a = _agg(results, key, name)
            L.append(f"\n**Aggregate.** micro P/R/F1 = {a['microP']:.2f}/{a['microR']:.2f}/"
                     f"{a['microF']:.2f}; macro P/R/F1 = {a['macroP']:.2f}/{a['macroR']:.2f}/"
                     f"{a['macroF']:.2f} (ΣTP={a['TP']},ΣFP={a['FP']},ΣFN={a['FN']}).")
        L.append(f"\n### {title} — summary\n")
        L.append("| method | micro P | micro R | micro F1 | macro P | macro R | macro F1 |")
        L.append("|---|---|---|---|---|---|---|")
        for name, _k, _kind in METHODS:
            a = _agg(results, key, name)
            L.append(f"| {name} | {a['microP']:.2f} | {a['microR']:.2f} | {a['microF']:.2f} | "
                     f"{a['macroP']:.2f} | {a['macroR']:.2f} | {a['macroF']:.2f} |")

    section("Per-method (vs own member column)", "own")
    section("Apples-to-apples (vs overlap = articles in both corpora)", "aa")
    L += ["\n## Caveats\n",
          "- Precision is vs. the curated set → lower bound on true relevance.",
          "- Apples-to-apples uses the overlap set (retrievable by all methods); recall there "
          "is the cleanest comparison.",
          "- Many remaining misses sit at 0.60–0.70 similarity (just under the 0.7 floor) → "
          "recall is threshold-limited."]
    return "\n".join(L) + "\n"


def run(csv_path: str = "docs/article_clusters.csv",
        out_path: str | None = "docs/cluster_retrieval_eval.md",
        dsn: str | None = None, *, threshold: float = 0.7,
        hybrid_article_threshold: float = 0.4, hybrid_box_threshold: float = 0.65,
        limit: int = 60, lookback_plus: int = 1) -> str:
    """Evaluate, write the report (if out_path), print a summary; return the report."""
    from ticker_news.evals.pipeline_eval import connect_eval

    clusters = load_clusters(csv_path)
    if not clusters:
        raise SystemExit(f"no usable clusters in {csv_path}")
    conn = connect_eval(dsn)
    try:
        results = evaluate(conn, clusters, threshold=threshold,
                           hybrid_article_threshold=hybrid_article_threshold,
                           hybrid_box_threshold=hybrid_box_threshold, limit=limit,
                           lookback_plus=lookback_plus)
    finally:
        conn.rollback()
        conn.close()
    report = format_report(results, threshold=threshold,
                           hybrid_article_threshold=hybrid_article_threshold,
                           hybrid_box_threshold=hybrid_box_threshold, limit=limit,
                           lookback_plus=lookback_plus)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report)
    for key, label in (("own", "vs own column"), ("aa", "vs overlap")):
        print(f"\n{label}:")
        for name, _k, _kind in METHODS:
            a = _agg(results, key, name)
            print(f"  {name:46} micro P/R/F1 {a['microP']:.2f}/{a['microR']:.2f}/{a['microF']:.2f}"
                  f"  macro {a['macroP']:.2f}/{a['macroR']:.2f}/{a['macroF']:.2f}")
    if out_path:
        print(f"\nwrote {out_path}")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "docs/article_clusters.csv")
