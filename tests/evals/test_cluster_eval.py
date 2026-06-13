"""Offline unit tests for the cluster retrieval eval (no DB, no network)."""

from ticker_news.evals.cluster_eval import Cluster, _ids, _metrics, load_clusters

HEADER = (
    "cluster theme,"
    "article ids with insights – all_less_than_3_months_before_later,"
    "article ids with insights distilled – two pass – all_less_than_3_months_before_later,"
    "later act articles_ids,"
    "retrieval window start (earliest listed),"
    "retrieval lookback days,"
    "articles in both (insights ∩ distilled)"
)


def test_ids_parses_and_skips_junk():
    assert _ids("100 200 300") == {100, 200, 300}
    assert _ids("") == set()
    assert _ids(None) == set()
    assert _ids("  , x 42 ") == {42}


def test_metrics_precision_recall_f1():
    m = _metrics(expected={1, 2, 3}, retrieved={2, 3, 4})
    assert (m["TP"], m["FP"], m["FN"]) == (2, 1, 1)
    assert round(m["prec"], 3) == round(2 / 3, 3)
    assert round(m["rec"], 3) == round(2 / 3, 3)
    assert round(m["f1"], 3) == round(2 / 3, 3)


def test_metrics_empty_expected_is_zero_not_crash():
    m = _metrics(expected=set(), retrieved={1, 2})
    assert m["prec"] == 0.0 and m["rec"] == 0.0 and m["f1"] == 0.0


def test_load_clusters_reads_new_schema(tmp_path):
    csv = tmp_path / "clusters.csv"
    csv.write_text(
        HEADER + "\n"
        "Test event,100 200 300,200 300,999,2025-01-01,30,200 300\n"
        ",,,\n",  # blank row ignored
        encoding="utf-8",
    )
    clusters = load_clusters(str(csv))
    assert clusters == [Cluster(
        theme="Test event", seed=999, lookback_days=30,
        insights={100, 200, 300}, distilled={200, 300}, overlap={200, 300},
    )]
