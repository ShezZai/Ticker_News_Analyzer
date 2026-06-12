"""Offline unit tests for the E2E pipeline eval. No DB, no network."""

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ticker_news.evals import pipeline_eval
from ticker_news.evals.pipeline_eval import (
    SKIPPABLE_STAGES,
    avg_directional_agreement,
    build_items,
    directional_agreement_evaluator,
    parse_ids_file,
    parse_skip_stages,
    price_move_evaluator,
    reset_article,
    score_directional,
    upsert_dataset_items,
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

    def test_no_ticker_becomes_categorical_skip_score(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (2.5, None))
        ev = directional_agreement_evaluator(
            input=ITEM_INPUT,
            output={"action": None, "ticker": None, "skip_reason": "no primary ticker"},
        )
        # Langfuse rejects value=None; exclusions become a categorical sibling score
        assert ev.name == "directional_agreement_skip"
        assert "no primary ticker" in ev.value

    def test_hold_becomes_categorical_skip_score(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (2.5, None))
        ev = directional_agreement_evaluator(
            input=ITEM_INPUT,
            output={"action": "hold", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.name == "directional_agreement_skip"
        assert "hold" in ev.value

    def test_price_move_recorded_even_for_hold(self, monkeypatch):
        monkeypatch.setattr(pipeline_eval, "realized_move", lambda t, p: (-1.3, None))
        ev = price_move_evaluator(
            input=ITEM_INPUT,
            output={"action": "hold", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.name == "price_move_pct"
        assert ev.value == -1.3

    def test_price_move_skip_score_when_no_data(self, monkeypatch):
        monkeypatch.setattr(
            pipeline_eval, "realized_move", lambda t, p: (None, "no tradeable entry/exit bar")
        )
        ev = price_move_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "ticker": "MRVL", "skip_reason": None},
        )
        assert ev.name == "price_move_pct_skip"
        assert "no tradeable" in ev.value


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
        assert ev.name == "avg_directional_agreement_skip"
        assert ev.value == "no scorable items"


class TestResetArticleKeep:
    def _update_sql(self, conn):
        updates = [s for s, _ in conn.executed if s.startswith("UPDATE")]
        return updates[0] if updates else ""

    def test_keep_embed_preserves_embedding(self):
        conn = FakeConn()
        reset_article(conn, 20512, keep=frozenset({"embed"}))
        sql = self._update_sql(conn)
        assert "embedding = NULL" not in sql
        assert "category = NULL" in sql
        assert any("article_insights" in s for s, _ in conn.executed)

    def test_keep_insights_preserves_rows_and_stamp(self):
        conn = FakeConn()
        reset_article(conn, 20512, keep=frozenset({"insights"}))
        assert not any("article_insights" in s for s, _ in conn.executed)
        sql = self._update_sql(conn)
        assert "insights_extracted_at = NULL" not in sql
        assert "embedding = NULL" in sql

    def test_keep_all_still_deletes_sentiment(self):
        conn = FakeConn()
        reset_article(conn, 20512, keep=frozenset(SKIPPABLE_STAGES))
        sqls = [s for s, _ in conn.executed if s != "COMMIT"]
        assert len(sqls) == 1
        assert "article_sentiment" in sqls[0]

    def test_default_resets_everything_as_before(self):
        conn = FakeConn()
        reset_article(conn, 20512)
        sql = self._update_sql(conn)
        for col in ("embedding", "category", "category_reason", "primary_ticker",
                    "primary_segment", "more_tickers", "more_segments",
                    "insights_extracted_at"):
            assert f"{col} = NULL" in sql


class TestParseSkipStages:
    def test_parses_comma_list(self):
        assert parse_skip_stages("embed, insights") == frozenset({"embed", "insights"})

    def test_none_and_empty_mean_no_skips(self):
        assert parse_skip_stages(None) == frozenset()
        assert parse_skip_stages("") == frozenset()

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="sentiment"):
            parse_skip_stages("embed,sentiment")


class TestRunPipelineTaskCategoryBackfill:
    def test_skipped_classify_reports_stored_category(self, monkeypatch):
        from ticker_news.evals import pipeline_eval
        from ticker_news.service import stages

        conn = FakeConn(rows=[("real news",)])  # SELECT category -> stored value
        conn.close = lambda: None
        monkeypatch.setattr(pipeline_eval, "connect_eval", lambda dsn: conn)
        monkeypatch.setattr(pipeline_eval, "reset_article",
                            lambda c, aid, keep=frozenset(): None)
        monkeypatch.setattr(stages.TagContext, "load", classmethod(lambda cls, c: None))
        monkeypatch.setattr(stages, "embed_stage", lambda c, u: None)
        monkeypatch.setattr(stages, "classify_stage", lambda c, u: None)  # no-op: kept
        monkeypatch.setattr(stages, "tag_stage", lambda c, u, t: None)
        monkeypatch.setattr(stages, "insights_stage", lambda c, u, t: None)
        monkeypatch.setattr(
            stages, "sentiment_stage",
            lambda c, u: {"ticker": "NVDA", "action": "buy", "confidence": 0.8},
        )
        task = pipeline_eval.make_task(dsn=None, skip_stages=frozenset({"classify"}))
        out = task(item={"input": {"article_id": 20512, "url": "https://x/20512"}})
        assert out["category"] == "real news"
        assert out["action"] == "buy"

    def test_task_names_the_trace_after_experiment_and_article(self, monkeypatch):
        import langfuse

        from ticker_news.evals import pipeline_eval
        from ticker_news.service import stages

        seen = {}

        @contextmanager
        def fake_propagate(**kwargs):
            seen.update(kwargs)
            yield

        monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
        conn = FakeConn(rows=[("real news",)])
        conn.close = lambda: None
        monkeypatch.setattr(pipeline_eval, "connect_eval", lambda dsn: conn)
        monkeypatch.setattr(pipeline_eval, "reset_article",
                            lambda c, aid, keep=frozenset(): None)
        monkeypatch.setattr(stages.TagContext, "load", classmethod(lambda cls, c: None))
        monkeypatch.setattr(stages, "embed_stage", lambda c, u: None)
        monkeypatch.setattr(stages, "classify_stage", lambda c, u: "real news")
        monkeypatch.setattr(stages, "tag_stage", lambda c, u, t: None)
        monkeypatch.setattr(stages, "insights_stage", lambda c, u, t: None)
        monkeypatch.setattr(
            stages, "sentiment_stage",
            lambda c, u: {"ticker": "NVDA", "action": "buy", "confidence": 0.8},
        )
        task = pipeline_eval.make_task(dsn=None)
        task(item={"input": {"article_id": 20512, "url": "https://x/20512"}})
        assert seen["trace_name"] == "pipeline-e2e:article-20512"


class TestParseIdsFile:
    def test_parses_one_id_per_line(self, tmp_path):
        f = tmp_path / "ids.csv"
        f.write_text("671\n685\n\n694\n", encoding="utf-8")
        assert parse_ids_file(f) == [671, 685, 694]

    def test_tolerates_bom_and_trailing_commas(self, tmp_path):
        f = tmp_path / "ids.csv"
        f.write_text("671,\n685\n", encoding="utf-8-sig")
        assert parse_ids_file(f) == [671, 685]

    def test_non_integer_line_raises_with_line_number(self, tmp_path):
        f = tmp_path / "ids.csv"
        f.write_text("671\nabc\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 2.*abc"):
            parse_ids_file(f)


class FakeLangfuseClient:
    """Records create_dataset_item calls; serves canned existing items."""

    def __init__(self, existing_items=()):
        self._existing = list(existing_items)
        self.created: list[dict] = []

    def get_dataset(self, name):
        return SimpleNamespace(items=self._existing)

    def create_dataset_item(self, **kwargs):
        self.created.append(kwargs)


class TestUpsertDatasetItems:
    def test_new_items_get_dataset_scoped_ids(self):
        client = FakeLangfuseClient()
        upsert_dataset_items(client, "ds-a", [
            {"input": {"article_id": 595}, "metadata": {"m": 1},
             "expected_output": {"act": "NO"}},
        ])
        assert client.created == [{
            "dataset_name": "ds-a", "id": "ds-a:article-595",
            "input": {"article_id": 595}, "metadata": {"m": 1},
            "expected_output": {"act": "NO"},
        }]

    def test_existing_items_updated_via_their_own_id(self):
        legacy = SimpleNamespace(id="article-595", input={"article_id": 595})
        client = FakeLangfuseClient(existing_items=[legacy])
        upsert_dataset_items(client, "ds-a", [
            {"input": {"article_id": 595, "title": "T"}},
        ])
        assert client.created[0]["id"] == "article-595"   # reuse, don't duplicate

    def test_expected_output_omitted_when_absent(self):
        client = FakeLangfuseClient()
        upsert_dataset_items(client, "ds-a", [{"input": {"article_id": 1}}])
        assert "expected_output" not in client.created[0]
        assert client.created[0]["metadata"] is None
