"""Offline tests for ticker_news.research.search (pure helpers + CLI plumbing).

DB-dependent paths (ANN queries, seed resolution) are intentionally not
covered here; they are thin pass-throughs over psycopg + pgvector.
"""

import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

import ticker_news.cli as cli
from ticker_news.research import search as mod
from ticker_news.research.search import (
    SearchHit,
    build_filters,
    format_hits,
    format_meta,
    load_statements,
    scope_to_filter,
    seed_window,
)

runner = CliRunner()


def _hit(**overrides):
    base = dict(
        id=1,
        url="https://example.com/a",
        title="NVDA pops",
        source_domain="example.com",
        publisher="Example",
        published_utc=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        tickers=["NVDA", "AMD"],
        similarity=0.91237,
    )
    base.update(overrides)
    return SearchHit(**base)


# ---------------------------------------------------------------- build_filters

def test_build_filters_empty():
    assert build_filters() == ("", [])


def test_build_filters_tickers_normalized():
    frag, params = build_filters(tickers=["nvda ", " amd"])
    assert frag == " AND tickers && %s"
    assert params == [["NVDA", "AMD"]]


def test_build_filters_tickers_and_dates_date_only_until():
    frag, params = build_filters(
        tickers=["NVDA"], since="2025-01-01", until="2025-08-31"
    )
    assert frag == (
        " AND tickers && %s"
        " AND published_utc >= %s"
        " AND published_utc < (%s::date + interval '1 day')"
    )
    assert params == [["NVDA"], "2025-01-01", "2025-08-31"]


def test_build_filters_until_exclusive_is_strict():
    frag, params = build_filters(until="2025-08-31", until_exclusive=True)
    assert frag == " AND published_utc < %s"
    assert params == ["2025-08-31"]


def test_build_filters_until_timestamp_is_inclusive():
    frag, params = build_filters(until="2025-08-31T12:00:00")
    assert frag == " AND published_utc <= %s"
    assert params == ["2025-08-31T12:00:00"]


def test_build_filters_segment_matches_primary_or_more():
    frag, params = build_filters(segment="Memory & Storage")
    assert frag == " AND (primary_segment = %s OR more_segments @> %s)"
    assert params == ["Memory & Storage", ["Memory & Storage"]]


def test_build_filters_domain_and_exclude_id():
    frag, params = build_filters(domain="fool.com", exclude_id=42)
    assert frag == " AND source_domain = %s AND id <> %s"
    assert params == ["fool.com", 42]


# ----------------------------------------------------------------- seed_window

SEED = datetime(2025, 8, 15, 14, 30, 0)


def test_seed_window_inclusive_day_open_start():
    assert seed_window(SEED, None, False) == (None, "2025-08-15")


def test_seed_window_exclusive_uses_exact_timestamp():
    assert seed_window(SEED, None, True) == (None, "2025-08-15T14:30:00")


def test_seed_window_months_before_day_anchor():
    assert seed_window(SEED, 3, False) == ("2025-05-15", "2025-08-15")


def test_seed_window_months_before_exclusive_timestamp_anchor():
    assert seed_window(SEED, 3, True) == ("2025-05-15T14:30:00", "2025-08-15T14:30:00")


def test_seed_window_month_end_clamps():
    since, until = seed_window(datetime(2025, 3, 31, 9, 0), 1, False)
    assert (since, until) == ("2025-02-28", "2025-03-31")


# ------------------------------------------------------------- scope_to_filter

def test_scope_to_filter_sector_is_unfiltered():
    assert scope_to_filter("sector", None) == {}
    assert scope_to_filter("", None) == {}   # blank defaults to sector
    assert scope_to_filter(None, None) == {}


def test_scope_to_filter_ticker_and_segment():
    assert scope_to_filter("ticker", "NVDA") == {"tickers": ["NVDA"]}
    assert scope_to_filter("Segment", "Memory") == {"segment": "Memory"}


def test_scope_to_filter_errors():
    with pytest.raises(ValueError):
        scope_to_filter("ticker", None)
    with pytest.raises(ValueError):
        scope_to_filter("segment", "")
    with pytest.raises(ValueError):
        scope_to_filter("bogus", "x")


# ------------------------------------------------------------- load_statements

def test_load_statements_defaults_scope_sector(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"statement": "a"}, {"statement": "b", "scope": "ticker", "value": "NVDA"}]),
                 encoding="utf-8")
    items = load_statements(str(p))
    assert items == [
        {"statement": "a", "scope": "sector", "value": None},
        {"statement": "b", "scope": "ticker", "value": "NVDA"},
    ]


def test_load_statements_rejects_non_array(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"statement": "a"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_statements(str(p))


def test_load_statements_rejects_missing_statement(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"scope": "sector"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_statements(str(p))


# ------------------------------------------------------------------ formatting

def test_format_meta_full():
    assert format_meta(_hit()) == "2025-01-02 03:04:05+0000 | [NVDA,AMD]"


def test_format_meta_missing_fields():
    h = _hit(published_utc=None, tickers=[])
    assert format_meta(h) == "------------------- | [-]"


def test_format_hits_empty():
    assert format_hits([], "'q'") == "No matching articles for 'q'."


def test_format_hits_layout():
    out = format_hits([_hit()], "'q'")
    lines = out.split("\n")
    assert lines[0] == "Found 1 article(s) similar to 'q':"
    assert lines[1] == ""
    assert lines[2] == "1. NVDA pops"
    assert lines[3] == "   0.912 | 2025-01-02 03:04:05+0000 | [NVDA,AMD]"
    assert lines[4] == "   https://example.com/a"
    assert lines[5] == ""


def test_format_hits_no_title_placeholder():
    out = format_hits([_hit(title=None)], "it")
    assert "1. (no title)" in out


# --------------------------------------------------------------------- the CLI

def _combined_output(result):
    try:
        return result.output + (result.stderr or "")
    except ValueError:  # stderr not separately captured
        return result.output


def test_search_help_renders():
    result = runner.invoke(cli.app, ["search", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--like", "--like-url", "--statement", "--ticker", "--segment",
                 "--domain", "--since", "--until", "--months-before",
                 "--min-similarity", "--exclusive", "--ef-search"):
        assert flag in result.output


def test_search_requires_some_subject(monkeypatch):
    monkeypatch.setattr(mod, "search", lambda *a, **kw: pytest.fail("searched"))
    result = runner.invoke(cli.app, ["search"])
    assert result.exit_code != 0
    assert "provide a query string" in _combined_output(result)


def test_statement_requires_seed(monkeypatch):
    monkeypatch.setattr(mod, "search", lambda *a, **kw: pytest.fail("searched"))
    result = runner.invoke(cli.app, ["search", "--statement", "nvidia is doomed"])
    assert result.exit_code != 0
    assert "--statement requires --like or --like-url" in _combined_output(result)


def test_months_before_requires_seed(monkeypatch):
    monkeypatch.setattr(mod, "search", lambda *a, **kw: pytest.fail("searched"))
    result = runner.invoke(cli.app, ["search", "rate cuts", "--months-before", "3"])
    assert result.exit_code != 0
    assert "--months-before requires --like or --like-url" in _combined_output(result)


def test_search_text_query_passes_filters(monkeypatch):
    captured = {}

    def fake_search(query, **kw):
        captured["query"] = query
        captured.update(kw)
        return [_hit()]

    monkeypatch.setattr(mod, "search", fake_search)
    result = runner.invoke(
        cli.app,
        ["search", "hello", "-k", "2", "--ticker", "nvda", "--ticker", "amd",
         "--segment", "Memory", "--domain", "fool.com", "--since", "2025-01-01",
         "--until", "2025-06-01", "--min-similarity", "0", "--exclusive"],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "query": "hello", "k": 2, "tickers": ["nvda", "amd"], "segment": "Memory",
        "domain": "fool.com", "since": "2025-01-01", "until": "2025-06-01",
        "min_similarity": 0.0, "until_exclusive": True,
    }
    assert "Found 1 article(s) similar to 'hello':" in result.output


def test_search_like_shows_seed_and_excludes_it(monkeypatch):
    seed = _hit(id=5, title="Seed title")
    captured = {}

    def fake_similar_to(article_id, **kw):
        captured["article_id"] = article_id
        return []

    monkeypatch.setattr(mod, "get_article", lambda **kw: seed)
    monkeypatch.setattr(mod, "similar_to", fake_similar_to)
    result = runner.invoke(cli.app, ["search", "--like", "5"])
    assert result.exit_code == 0, result.output
    assert captured["article_id"] == 5
    assert "Seed article:" in result.output
    assert "Seed title" in result.output
    assert "No matching articles for it." in result.output


def test_search_statement_anchors_window_on_seed(monkeypatch):
    seed = _hit(id=5, published_utc=datetime(2025, 8, 15, 14, 30, 0))
    captured = {}

    def fake_search(query, **kw):
        captured["query"] = query
        captured.update(kw)
        return []

    monkeypatch.setattr(mod, "get_article", lambda **kw: seed)
    monkeypatch.setattr(mod, "search", fake_search)
    result = runner.invoke(
        cli.app, ["search", "--like", "5", "--statement", "nvidia faces competition"]
    )
    assert result.exit_code == 0, result.output
    assert captured["query"] == "nvidia faces competition"
    assert captured["since"] is None
    assert captured["until"] == "2025-08-15"
    assert "[window] (open) .. 2025-08-15 (up to seed, inclusive day)" in result.output
    assert "Dating-anchor article:" in result.output
    assert "=== [1] statement ===" in result.output


def test_search_ef_search_overrides_global(monkeypatch):
    monkeypatch.setattr(mod, "EF_SEARCH", mod.EF_SEARCH)  # register restore
    monkeypatch.setattr(mod, "search", lambda *a, **kw: [])
    result = runner.invoke(cli.app, ["search", "hello", "--ef-search", "99"])
    assert result.exit_code == 0, result.output
    assert mod.EF_SEARCH == 99


def test_search_value_errors_become_clean_cli_errors(monkeypatch):
    def boom(**kw):
        raise ValueError("article 999 not found")

    monkeypatch.setattr(mod, "get_article", boom)
    result = runner.invoke(cli.app, ["search", "--like", "999", "--months-before", "3"])
    assert result.exit_code != 0
    assert "article 999 not found" in _combined_output(result)
