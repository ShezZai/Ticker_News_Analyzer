"""Offline tests for ticker_news.research.insight_search (pure helpers + CLI).

DB-dependent paths (ANN queries, seed resolution, insights_of) are
intentionally not covered here; they are thin pass-throughs over psycopg +
pgvector. The shared ANN GUC helper is tested in test_search.py.
"""

import re
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

import ticker_news.cli as cli
from ticker_news.research import insight_search as mod
from ticker_news.research import search as search_mod
from ticker_news.research.insight_search import (
    InsightGroup,
    InsightHit,
    RelatedArticle,
    SeedInsight,
    build_filters,
    consolidate,
    format_consolidated,
    format_groups,
    format_meta,
    format_text_results,
)

runner = CliRunner()

WHEN = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _hit(**overrides):
    base = dict(
        insight_id=101,
        article_id=10,
        source_url="https://example.com/a",
        article_headline="NVDA pops",
        topic="Data center demand",
        insight="Demand is strong.",
        box_text="full box text",
        tickers=["NVDA", "AMD"],
        published_utc=WHEN,
        similarity=0.91237,
    )
    base.update(overrides)
    return InsightHit(**base)


def _seed(**overrides):
    base = dict(
        insight_id=1,
        box_index=0,
        topic="Seed topic",
        insight="Seed insight text.",
        box_text="seed box",
        embedding=[0.0],
    )
    base.update(overrides)
    return SeedInsight(**base)


def _ra(**overrides):
    base = dict(
        article_id=10,
        source_url="https://example.com/a",
        article_headline="NVDA pops",
        tickers=["NVDA"],
        published_utc=WHEN,
        best_similarity=0.95123,
        matched_insights=2,
        sum_similarity=1.8,
    )
    base.update(overrides)
    return RelatedArticle(**base)


# ------------------------------------------------------------- shared helpers

def test_window_and_date_helpers_are_shared_with_search():
    # the legacy file duplicated these; the port imports them instead
    assert mod.seed_window is search_mod.seed_window
    assert mod._is_date_only is search_mod._is_date_only
    assert mod._apply_ann_gucs is search_mod._apply_ann_gucs
    assert mod.DEFAULT_MIN_SIMILARITY == 0.7


# --------------------------------------------------------------- build_filters

def test_build_filters_always_requires_embedding():
    assert build_filters() == (["ai.embedding IS NOT NULL"], [])


def test_build_filters_full_set_and_clause_order():
    clauses, params = build_filters(
        tickers=["nvda ", " amd"], segment="Memory & Storage", domain="fool.com",
        since="2025-01-01", until="2025-08-31", exclude_article_id=42,
    )
    assert clauses == [
        "ai.embedding IS NOT NULL",
        "ai.article_id <> %s",
        "a.tickers && %s",
        "(a.primary_segment = %s OR a.more_segments @> %s)",
        "a.source_domain = %s",
        "a.published_utc >= %s",
        "a.published_utc < (%s::date + interval '1 day')",
    ]
    assert params == [
        42, ["NVDA", "AMD"], "Memory & Storage", ["Memory & Storage"],
        "fool.com", "2025-01-01", "2025-08-31",
    ]


def test_build_filters_until_exclusive_is_strict():
    clauses, params = build_filters(until="2025-08-31", until_exclusive=True)
    assert clauses[-1] == "a.published_utc < %s"
    assert params == ["2025-08-31"]


def test_build_filters_until_timestamp_is_inclusive():
    clauses, params = build_filters(until="2025-08-31T12:00:00")
    assert clauses[-1] == "a.published_utc <= %s"
    assert params == ["2025-08-31T12:00:00"]


# ----------------------------------------------------------------- consolidate

def test_consolidate_best_similarity_outranks_match_count():
    # A: one matched insight but the highest single similarity.
    # B: matched by three seed insights, all weaker. Legacy ranks A first.
    groups = [
        InsightGroup(seed=_seed(insight_id=1),
                     hits=[_hit(article_id=1, similarity=0.95),
                           _hit(article_id=2, similarity=0.90)]),
        InsightGroup(seed=_seed(insight_id=2),
                     hits=[_hit(article_id=2, similarity=0.89)]),
        InsightGroup(seed=_seed(insight_id=3),
                     hits=[_hit(article_id=2, similarity=0.88)]),
    ]
    ranked = consolidate(groups)
    assert [r.article_id for r in ranked] == [1, 2]
    a, b = ranked
    assert (a.best_similarity, a.matched_insights) == (0.95, 1)
    assert (b.best_similarity, b.matched_insights) == (0.90, 3)
    assert b.sum_similarity == pytest.approx(0.90 + 0.89 + 0.88)


def test_consolidate_tie_on_best_more_matched_insights_wins():
    groups = [
        InsightGroup(seed=_seed(insight_id=1),
                     hits=[_hit(article_id=1, similarity=0.9),
                           _hit(article_id=2, similarity=0.9)]),
        InsightGroup(seed=_seed(insight_id=2),
                     hits=[_hit(article_id=2, similarity=0.8)]),
    ]
    ranked = consolidate(groups)
    assert [r.article_id for r in ranked] == [2, 1]


def test_consolidate_tie_on_best_and_count_sum_similarity_wins():
    groups = [
        InsightGroup(seed=_seed(insight_id=1),
                     hits=[_hit(article_id=1, similarity=0.9),
                           _hit(article_id=2, similarity=0.9)]),
        InsightGroup(seed=_seed(insight_id=2),
                     hits=[_hit(article_id=1, similarity=0.7),
                           _hit(article_id=2, similarity=0.8)]),
    ]
    ranked = consolidate(groups)
    assert [r.article_id for r in ranked] == [2, 1]
    assert ranked[0].sum_similarity == pytest.approx(1.7)
    assert ranked[1].sum_similarity == pytest.approx(1.6)


def test_consolidate_counts_article_once_per_seed_insight():
    # two hits into the same article from ONE seed insight -> one match (best hit)
    groups = [
        InsightGroup(seed=_seed(),
                     hits=[_hit(article_id=1, similarity=0.8),
                           _hit(article_id=1, similarity=0.9)]),
    ]
    (ra,) = consolidate(groups)
    assert ra.matched_insights == 1
    assert ra.best_similarity == 0.9
    assert ra.sum_similarity == 0.9


def test_consolidate_empty():
    assert consolidate([]) == []
    assert consolidate([InsightGroup(seed=_seed())]) == []


# ------------------------------------------------------------------ formatting

def test_format_meta_full_and_missing():
    assert format_meta(["NVDA", "AMD"], WHEN) == "2025-01-02 03:04:05+0000 | [NVDA,AMD]"
    assert format_meta([], None) == "------------------- | [-]"


def test_format_text_results_empty():
    assert format_text_results([], "q") == "No matching insights for 'q'."


def test_format_text_results_layout():
    out = format_text_results([_hit()], "q")
    assert out.split("\n") == [
        "",
        "Top 1 insight(s) similar to 'q':",
        "",
        "1. 0.912 | Data center demand",
        "   Demand is strong.",
        "   2025-01-02 03:04:05+0000 | [NVDA,AMD]",
        "   from a#10: NVDA pops",
        "   https://example.com/a",
        "",
    ]


def test_format_text_results_optional_fields_omitted():
    out = format_text_results(
        [_hit(topic=None, insight=None, article_headline=None, source_url=None)], "q"
    )
    lines = out.split("\n")
    assert lines[3] == "1. 0.912 | (no topic)"
    assert lines[4] == "   2025-01-02 03:04:05+0000 | [NVDA,AMD]"
    assert lines[5] == "   from a#10: (no headline)"
    assert lines[6] == ""  # no insight line, no url line


def test_format_consolidated_empty():
    assert format_consolidated([], 5, None) == (
        "\nNo related articles found via insight overlap."
    )


def test_format_consolidated_layout_and_cap_note():
    out = format_consolidated([_ra()], 5, 10)
    assert out.split("\n") == [
        "",
        "1 consolidated article(s) from top-5 insight matches per seed insight"
        " (capped at 10):",
        "",
        "1. best 0.951 | 2 insight(s) matched | sum 1.80",
        "   NVDA pops",
        "   2025-01-02 03:04:05+0000 | [NVDA]",
        "   https://example.com/a",
        "",
    ]


def test_format_consolidated_no_cap_note_without_top_articles():
    out = format_consolidated([_ra()], 3, None)
    assert "capped" not in out
    assert "from top-3 insight matches per seed insight:" in out


def test_format_groups_empty():
    assert format_groups([]) == (
        "\nThis article has no embedded insights to search with."
    )


def test_format_groups_layout():
    groups = [
        InsightGroup(seed=_seed(), hits=[_hit()]),
        InsightGroup(seed=_seed(insight_id=2, topic=None, insight=None)),
    ]
    out = format_groups(groups)
    assert out.split("\n") == [
        "",
        "2 seed insight(s), 1 with matches:",
        "",
        "[1] TOPIC: Seed topic",
        "    INSIGHT: Seed insight text.",
        "    Top 1 similar insight(s):",
        "      1. 0.912 | Data center demand",
        "         Demand is strong.",
        "         2025-01-02 03:04:05+0000 | [NVDA,AMD]",
        "         from a#10: NVDA pops",
        "         https://example.com/a",
        "",
        "[2] TOPIC: (no topic)",
        "    (no similar insights matched the filters)",
        "",
    ]


# --------------------------------------------------- ANN plumbing pass-through

class _FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.closed = False

    def cursor(self):
        return _FakeCursor()

    def close(self):
        self.closed = True


def test_run_insight_search_threads_ef_search_to_gucs(monkeypatch):
    captured = {}

    def fake_apply(cur, ef_search=None):
        captured["ef_search"] = ef_search

    monkeypatch.setattr(mod, "_apply_ann_gucs", fake_apply)
    hits = mod._run_insight_search(_FakeConn(), [0.0], 5, ef_search=77)
    assert hits == []
    assert captured["ef_search"] == 77


# --------------------------------------------------------------------- the CLI

def _combined_output(result):
    try:
        out = result.output + (result.stderr or "")
    except ValueError:  # stderr not separately captured
        out = result.output
    # Rich wraps long BadParameter messages inside a bordered box; strip the
    # box-drawing characters and collapse whitespace so substring assertions
    # see the plain message even when it spans lines.
    return re.sub(r"\s+", " ", re.sub(r"[│┃┌┐└┘├┤─━╭╮╰╯]", " ", out))


def test_search_insights_help_renders():
    result = runner.invoke(cli.app, ["search-insights", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--like", "--like-url", "--per-insight", "--top-articles",
                 "--ticker", "--segment", "--domain", "--since", "--until",
                 "--months-before", "--min-similarity", "--exclusive",
                 "--ef-search"):
        assert flag in result.output


def test_search_insights_requires_some_subject(monkeypatch):
    monkeypatch.setattr(mod, "search_insights", lambda *a, **kw: pytest.fail("searched"))
    result = runner.invoke(cli.app, ["search-insights"])
    assert result.exit_code != 0
    assert "provide a query string" in _combined_output(result)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--per-insight"], "--per-insight requires --like or --like-url"),
        (["--top-articles", "3"], "--top-articles requires --like or --like-url"),
        (["--months-before", "3"], "--months-before requires --like or --like-url"),
        (["--exclusive"],
         "--exclusive requires --like or --like-url (it anchors on the seed date)"),
    ],
)
def test_seed_only_flags_require_seed(monkeypatch, extra, message):
    monkeypatch.setattr(mod, "search_insights", lambda *a, **kw: pytest.fail("searched"))
    result = runner.invoke(cli.app, ["search-insights", "hello", *extra])
    assert result.exit_code != 0
    assert message in _combined_output(result)


def test_search_insights_text_query_passes_filters(monkeypatch):
    captured = {}

    def fake_search_insights(query, **kw):
        captured["query"] = query
        captured.update(kw)
        return [_hit()]

    monkeypatch.setattr(mod, "search_insights", fake_search_insights)
    result = runner.invoke(
        cli.app,
        ["search-insights", "hello", "-k", "2", "--ticker", "nvda",
         "--ticker", "amd", "--segment", "Memory", "--domain", "fool.com",
         "--since", "2025-01-01", "--until", "2025-06-01",
         "--min-similarity", "0", "--ef-search", "99"],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "query": "hello", "k": 2, "tickers": ["nvda", "amd"], "segment": "Memory",
        "domain": "fool.com", "since": "2025-01-01", "until": "2025-06-01",
        "min_similarity": 0.0, "ef_search": 99,
    }
    assert "Top 1 insight(s) similar to 'hello':" in result.output


SEED_META = {
    "id": 5, "title": "Seed title", "url": "https://example.com/seed",
    "tickers": ["NVDA"], "published_utc": datetime(2025, 8, 15, 14, 30, 0),
}


def _patch_seed(monkeypatch, meta=SEED_META):
    fake_conn = _FakeConn()
    monkeypatch.setattr(mod, "_get_conn", lambda: fake_conn)
    monkeypatch.setattr(
        mod, "_resolve_article_id", lambda conn, **kw: (meta["id"], dict(meta))
    )
    return fake_conn


def test_search_insights_like_consolidated_default(monkeypatch):
    fake_conn = _patch_seed(monkeypatch)
    captured = {}

    def fake_related(**kw):
        captured.update(kw)
        return [_ra()]

    monkeypatch.setattr(mod, "related_articles", fake_related)
    result = runner.invoke(cli.app, ["search-insights", "--like", "5", "-k", "3",
                                     "--top-articles", "10"])
    assert result.exit_code == 0, result.output
    assert captured["article_id"] == 5
    assert captured["k"] == 10            # --top-articles caps the article list
    assert captured["per_insight_k"] == 3  # -k is per seed insight
    assert captured["conn"] is fake_conn
    assert fake_conn.closed
    assert "Seed article:" in result.output
    assert "  Seed title" in result.output
    assert "  2025-08-15 14:30:00 | [NVDA]" in result.output
    assert "  https://example.com/seed" in result.output
    assert ("1 consolidated article(s) from top-3 insight matches per seed "
            "insight (capped at 10):") in result.output


def test_search_insights_months_before_anchors_window(monkeypatch):
    _patch_seed(monkeypatch)
    captured = {}

    def fake_related(**kw):
        captured.update(kw)
        return []

    monkeypatch.setattr(mod, "related_articles", fake_related)
    result = runner.invoke(
        cli.app, ["search-insights", "--like", "5", "--months-before", "3"]
    )
    assert result.exit_code == 0, result.output
    assert captured["since"] == "2025-05-15"
    assert captured["until"] == "2025-08-15"
    assert captured["until_exclusive"] is False
    assert ("[window] 2025-05-15 .. 2025-08-15 (3 months before seed, "
            "inclusive day)") in result.output
    assert "No related articles found via insight overlap." in result.output


def test_search_insights_exclusive_uses_exact_timestamp(monkeypatch):
    _patch_seed(monkeypatch)
    captured = {}

    def fake_related(**kw):
        captured.update(kw)
        return []

    monkeypatch.setattr(mod, "related_articles", fake_related)
    result = runner.invoke(cli.app, ["search-insights", "--like", "5", "--exclusive"])
    assert result.exit_code == 0, result.output
    assert captured["since"] is None
    assert captured["until"] == "2025-08-15T14:30:00"
    assert captured["until_exclusive"] is True
    assert ("[window] (open) .. 2025-08-15T14:30:00 (up to seed, exclusive ts)"
            in result.output)


def test_search_insights_per_insight_breakdown(monkeypatch):
    _patch_seed(monkeypatch)
    captured = {}

    def fake_groups(**kw):
        captured.update(kw)
        return [InsightGroup(seed=_seed(), hits=[_hit()])]

    monkeypatch.setattr(mod, "search_by_insights", fake_groups)
    monkeypatch.setattr(
        mod, "related_articles", lambda **kw: pytest.fail("consolidated path used")
    )
    result = runner.invoke(
        cli.app, ["search-insights", "--like", "5", "--per-insight", "-k", "7"]
    )
    assert result.exit_code == 0, result.output
    assert captured["article_id"] == 5
    assert captured["k"] == 7
    assert "1 seed insight(s), 1 with matches:" in result.output
    assert "[1] TOPIC: Seed topic" in result.output


def test_search_insights_seed_without_published_utc_errors(monkeypatch):
    meta = dict(SEED_META, published_utc=None)
    _patch_seed(monkeypatch, meta)
    result = runner.invoke(
        cli.app, ["search-insights", "--like", "5", "--months-before", "3"]
    )
    assert result.exit_code != 0
    assert ("seed article has no published_utc to anchor the window"
            in _combined_output(result))


def test_search_insights_value_errors_become_clean_cli_errors(monkeypatch):
    monkeypatch.setattr(mod, "_get_conn", lambda: _FakeConn())

    def boom(conn, **kw):
        raise ValueError("article 999 not found")

    monkeypatch.setattr(mod, "_resolve_article_id", boom)
    result = runner.invoke(cli.app, ["search-insights", "--like", "999"])
    assert result.exit_code != 0
    assert "article 999 not found" in _combined_output(result)
