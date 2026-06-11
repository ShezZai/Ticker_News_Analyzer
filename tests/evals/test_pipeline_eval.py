"""Offline unit tests for the E2E pipeline eval. No DB, no network."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ticker_news.evals import pipeline_eval
from ticker_news.evals.pipeline_eval import (
    avg_directional_agreement,
    build_items,
    directional_agreement_evaluator,
    price_move_evaluator,
    reset_article,
    score_directional,
)


class TestScoreDirectional:
    def test_buy_and_price_up_agrees(self):
        value, comment = score_directional("buy", 2.5)
        assert value == 1.0
        assert "agree" in comment

    def test_buy_and_price_down_disagrees(self):
        value, comment = score_directional("buy", -1.2)
        assert value == 0.0
        assert "disagree" in comment

    def test_buy_and_flat_price_disagrees(self):
        value, _ = score_directional("buy", 0.0)
        assert value == 0.0

    def test_sell_and_price_down_agrees(self):
        value, _ = score_directional("sell", -3.0)
        assert value == 1.0

    def test_sell_and_price_up_disagrees(self):
        value, _ = score_directional("sell", 1.7)
        assert value == 0.0

    def test_sell_and_flat_price_disagrees(self):
        value, _ = score_directional("sell", 0.0)
        assert value == 0.0

    def test_hold_is_excluded(self):
        value, comment = score_directional("hold", 2.0)
        assert value is None
        assert "hold" in comment

    def test_no_verdict_is_excluded_with_reason(self):
        value, comment = score_directional(None, None, skip_reason="category=recap/review")
        assert value is None
        assert "category=recap/review" in comment

    def test_missing_price_data_is_excluded(self):
        value, comment = score_directional("buy", None, skip_reason="no tradeable entry/exit bar")
        assert value is None
        assert "no price data" in comment

    def test_unknown_action_is_excluded(self):
        value, comment = score_directional("strong-buy", 1.0)
        assert value is None
        assert "strong-buy" in comment


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    """Records every execute; returns canned rows for SELECTs."""

    def __init__(self, rows=None):
        self.executed: list[tuple[str, tuple | None]] = []
        self._rows = rows or []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return FakeCursor(self._rows)

    def commit(self):
        self.executed.append(("COMMIT", None))


PUBLISHED = datetime(2026, 5, 28, 11, 10, tzinfo=timezone.utc)


def _row(aid=20512, ticker="MRVL", published=PUBLISHED, status="ok", has_content=True):
    return (aid, f"https://example.com/{aid}", ticker, published, "Title", status, has_content)


class TestBuildItems:
    def test_builds_langfuse_local_items(self):
        conn = FakeConn(rows=[_row()])
        items = build_items(conn, [20512])
        assert items == [{
            "input": {
                "article_id": 20512,
                "url": "https://example.com/20512",
                "published_utc": "2026-05-28T11:10:00+00:00",
                "title": "Title",
            },
            "metadata": {"seed_ticker": "MRVL"},
        }]

    def test_missing_id_raises(self):
        conn = FakeConn(rows=[_row()])
        with pytest.raises(ValueError, match="not found.*99999"):
            build_items(conn, [20512, 99999])

    def test_article_without_content_raises(self):
        conn = FakeConn(rows=[_row(status="empty", has_content=False)])
        with pytest.raises(ValueError, match="no scraped content"):
            build_items(conn, [20512])

    def test_article_without_published_raises(self):
        conn = FakeConn(rows=[_row(published=None)])
        with pytest.raises(ValueError, match="published_utc"):
            build_items(conn, [20512])


class TestResetArticle:
    def test_clears_derived_fields_and_dependent_rows(self):
        conn = FakeConn()
        reset_article(conn, 20512)
        statements = [(s, p) for s, p in conn.executed if s != "COMMIT"]
        # every data statement is parametrized on the article id
        assert [p for _, p in statements] == [(20512,), (20512,), (20512,)]
        sentiment_sql, insights_sql, update_sql = [s for s, _ in statements]
        assert "DELETE FROM public.article_sentiment" in sentiment_sql
        assert "WHERE article_id = %s" in sentiment_sql
        assert "DELETE FROM public.article_insights" in insights_sql
        assert "WHERE article_id = %s" in insights_sql
        assert update_sql.startswith("UPDATE public.articles SET ")
        assert "WHERE id = %s" in update_sql
        for col in ("embedding", "category", "category_reason", "primary_ticker",
                    "primary_segment", "more_tickers", "more_segments",
                    "insights_extracted_at"):
            assert f"{col} = NULL" in update_sql
        assert conn.executed[-1] == ("COMMIT", None)


ITEM_INPUT = {
    "article_id": 20512,
    "url": "https://example.com/20512",
    "published_utc": "2026-05-28T11:10:00+00:00",
    "title": "Title",
}


class TestItemEvaluators:
    @pytest.fixture(autouse=True)
    def _clear_move_cache(self):
        pipeline_eval._cached_move.cache_clear()

    def test_buy_with_rising_price_scores_one(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (2.5, None))
        ev = directional_agreement_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.name == "directional_agreement"
        assert ev.value == 1.0

    def test_no_ticker_excluded(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (2.5, None))
        ev = directional_agreement_evaluator(
            input=ITEM_INPUT,
            output={"action": None, "ticker": None, "skip_reason": "no primary ticker"},
        )
        assert ev.value is None
        assert "no primary ticker" in ev.comment

    def test_price_move_recorded_even_for_hold(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (-1.3, None))
        ev = price_move_evaluator(
            input=ITEM_INPUT,
            output={"action": "hold", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.name == "price_move_pct"
        assert ev.value == -1.3

    def test_price_move_none_when_no_data(self, monkeypatch):
        monkeypatch.setattr(
            pipeline_eval, "realized_move", lambda t, p: (None, "no tradeable entry/exit bar")
        )
        ev = price_move_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.value is None
        assert "no tradeable" in ev.comment


def _result(*evals):
    return SimpleNamespace(evaluations=list(evals))


class TestRunEvaluator:
    def test_averages_only_scored_items(self):
        results = [
            _result(SimpleNamespace(name="directional_agreement", value=1.0)),
            _result(SimpleNamespace(name="directional_agreement", value=0.0)),
            _result(SimpleNamespace(name="directional_agreement", value=None)),
            _result(SimpleNamespace(name="price_move_pct", value=5.0)),
        ]
        ev = avg_directional_agreement(item_results=results)
        assert ev.name == "avg_directional_agreement"
        assert ev.value == 0.5
        assert "2/4" in ev.comment

    def test_no_scorable_items(self):
        ev = avg_directional_agreement(item_results=[])
        assert ev.value is None
