"""Retrieval recall check against a labelled article-cluster CSV.

Each CSV row is a coherent news event (a cluster). We query the production
insight-precedent path *backward* from the cluster's `later` article and measure
how many of the cluster's earlier real-news members it surfaces — the precedent
retrieval only ever returns `category = 'real news'` sources, so the expected set
is the cluster's real-news members within the lookback window before `later`.

CSV columns used:
- later_news_article_ids:              the query article(s) (look back from here)
- news_less_than_3_months_before_later: expected real-news precedents (the recall target)

Run:
    python -m ticker_news.evals.cluster_eval docs/article_clusters_revisited.csv
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass

from ticker_news.service import stages


def _ids(cell: str) -> list[int]:
    return [int(tok) for tok in (cell or "").split() if tok.strip().isdigit()]


@dataclass
class Cluster:
    theme: str
    later: list[int]            # query article(s)
    expected: list[int]         # real-news members within window before `later`


# expected-precedent column by whether we restrict candidates to real news.
_EXPECTED_COL = {
    True: "news_less_than_3_months_before_later",   # real-news members in window
    False: "all_less_than_3_months_before_later",   # any-category members in window
}


def load_clusters(csv_path: str, *, real_news_only: bool = False) -> list[Cluster]:
    expected_col = _EXPECTED_COL[real_news_only]
    out: list[Cluster] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out.append(Cluster(
                theme=row["cluster theme"].strip(),
                later=_ids(row["later_news_article_ids"]),
                expected=_ids(row[expected_col]),
            ))
    return out


def retrieved_ids(conn, target: int, mode: str, *, threshold: float, limit: int,
                  lookback_days: int | None, real_news_only: bool) -> list[tuple[int, float]]:
    """(article_id, best similarity) precedents for one target under `mode`,
    ordered by similarity — via the exact production retrieval core."""
    src = stages.INSIGHT_SOURCES[mode]
    ranked = stages._ranked_insight_hits(
        conn, target, table=src.table, label_col=src.label_col,
        filter_drop=src.filter_drop, threshold=threshold, limit=limit,
        lookback_days=lookback_days, real_news_only=real_news_only,
    )
    best: dict[int, float] = {}
    for sim, aid, *_ in ranked:           # ranked is sorted desc; first wins
        best.setdefault(aid, sim)
    return list(best.items())


def run(csv_path: str, dsn: str | None = None, *,
        modes: tuple[str, ...] = ("insights", "distilled-first", "distilled-second"),
        threshold: float = 0.7, limit: int = 60, lookback_days: int = 90,
        real_news_only: bool = False) -> None:
    from ticker_news.evals.pipeline_eval import connect_eval

    clusters = load_clusters(csv_path, real_news_only=real_news_only)
    conn = connect_eval(dsn)
    label = "real-news" if real_news_only else "any-category"
    print(f"precedent candidates: {label}  (lookback={lookback_days}d, threshold={threshold}, limit={limit})")
    try:
        agg = {m: [0, 0] for m in modes}   # mode -> [hits, expected_total]
        for c in clusters:
            expected = set(c.expected) - set(c.later)
            print(f"\n=== {c.theme}")
            print(f"    query(later)={c.later or '—'}  expected {label} precedents={sorted(expected) or '—'}")
            if not expected:
                print("    (no eligible precedent in window — not testable)")
                continue
            for mode in modes:
                hit_ranks: dict[int, tuple[int, float]] = {}   # expected_id -> (rank, sim)
                for target in c.later:
                    for rank, (aid, sim) in enumerate(
                        sorted(retrieved_ids(conn, target, mode, threshold=threshold,
                                             limit=limit, lookback_days=lookback_days,
                                             real_news_only=real_news_only),
                               key=lambda t: -t[1]), start=1):
                        if aid in expected and aid not in hit_ranks:
                            hit_ranks[aid] = (rank, sim)
                hits = len(hit_ranks)
                agg[mode][0] += hits
                agg[mode][1] += len(expected)
                detail = ", ".join(
                    f"{eid}{'@rank%d sim%.2f' % hit_ranks[eid] if eid in hit_ranks else ' MISS'}"
                    for eid in sorted(expected)
                )
                print(f"    {mode:16} recall {hits}/{len(expected)}   [{detail}]")
        print("\n=== TOTALS (testable clusters)")
        for mode in modes:
            h, t = agg[mode]
            rate = f"{h/t:.0%}" if t else "n/a"
            print(f"    {mode:16} {h}/{t}  ({rate})")
    finally:
        conn.rollback()
        conn.close()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "docs/article_clusters_revisited.csv")
