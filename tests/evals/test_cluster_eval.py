from ticker_news.evals.cluster_eval import Cluster, _ids, load_clusters


def test_ids_parses_space_separated_and_skips_junk():
    assert _ids("2492 2510 2606") == [2492, 2510, 2606]
    assert _ids("") == []
    assert _ids(None) == []
    assert _ids("  ,, x ") == []


def _write_csv(tmp_path):
    csv = tmp_path / "clusters.csv"
    csv.write_text(
        "cluster theme,article ids,news_articles_ids,later_news_article_ids,"
        "all_less_than_3_months_before_later,news_less_than_3_months_before_later\n"
        "Stargate,2492 2606,2492,2606,2492 2510 2606,2492\n"
        "Empty news,619 691,,691,619 691,\n",
        encoding="utf-8",
    )
    return str(csv)


def test_load_clusters_default_uses_all_category_column(tmp_path):
    # default real_news_only=False -> all_less_than_3_months_before_later
    clusters = load_clusters(_write_csv(tmp_path))
    assert clusters == [
        Cluster(theme="Stargate", later=[2606], expected=[2492, 2510, 2606]),
        Cluster(theme="Empty news", later=[691], expected=[619, 691]),
    ]


def test_load_clusters_real_news_only_uses_news_column(tmp_path):
    clusters = load_clusters(_write_csv(tmp_path), real_news_only=True)
    assert clusters == [
        Cluster(theme="Stargate", later=[2606], expected=[2492]),
        Cluster(theme="Empty news", later=[691], expected=[]),
    ]
