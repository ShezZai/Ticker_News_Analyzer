"""Offline unit tests for the E2E pipeline eval. No DB, no network."""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ticker_news.evals import pipeline_eval
from ticker_news.evals.pipeline_eval import (
    SKIPPABLE_STAGES,
    act_decision_evaluator,
    act_metrics,
    avg_verdict_score,
    build_items,
    parse_ids_file,
    parse_skip_stages,
    price_move_evaluator,
    reset_article,
    score_act,
    score_decision,
    truth_act,
    upsert_dataset_items,
    verdict_evaluator,
)


class TestTruthAct:
    def test_pure_directional_is_should_act(self):
        assert truth_act(["buy"]) is True
        assert truth_act(["sell"]) is True

    def test_hold_in_set_is_no_act_ok(self):
        assert truth_act(["buy", "hold"]) is False
        assert truth_act(["sell", "hold"]) is False

    def test_case_insensitive(self):
        assert truth_act(["BUY", "HOLD"]) is False

    def test_missing_set_is_unscorable(self):
        assert truth_act(None) is None
        assert truth_act([]) is None


class TestScoreAct:
    def test_act_on_mover_is_tp(self):
        cell, comment = score_act(True, ["buy"])
        assert cell == "TP"
        assert "TP" in comment

    def test_act_on_flat_is_fp(self):
        cell, _ = score_act(True, ["sell", "hold"])
        assert cell == "FP"

    def test_skip_on_mover_is_fn(self):
        cell, _ = score_act(False, ["sell"])
        assert cell == "FN"

    def test_skip_on_flat_is_tn(self):
        cell, _ = score_act(False, ["buy", "hold"])
        assert cell == "TN"

    def test_unscorable_when_no_acceptable_set(self):
        cell, comment = score_act(True, None)
        assert cell is None
        assert "acceptable_verdicts" in comment


class TestScoreDecision:
    def test_direction_in_set_scores_one(self):
        value, comment = score_decision("buy", ["buy", "hold"])
        assert value == 1.0
        assert "correct" in comment

    def test_hold_in_set_scores_one(self):
        value, _ = score_decision("hold", ["sell", "hold"])
        assert value == 1.0

    def test_wrong_direction_scores_zero(self):
        value, comment = score_decision("buy", ["sell"])
        assert value == 0.0
        assert "wrong" in comment

    def test_none_action_treated_as_hold(self):
        value, _ = score_decision(None, ["buy"], skip_reason="no insights")
        assert value == 0.0  # hold not acceptable on a directional mover

    def test_no_acceptable_set_is_unscorable(self):
        value, comment = score_decision("buy", None)
        assert value is None
        assert "acceptable_verdicts" in comment


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


class TestActDecisionEvaluator:
    def test_act_on_mover_emits_correct_and_tp(self):
        evs = act_decision_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "is_act": True, "skip_reason": None},
            expected_output={"acceptable_verdicts": ["buy"], "gain_pct": 2.5},
        )
        by_name = {e.name: e.value for e in evs}
        assert by_name["act_correct"] == 1.0
        assert by_name["act_confusion"] == "TP"

    def test_skip_on_mover_is_wrong_and_fn(self):
        evs = act_decision_evaluator(
            input=ITEM_INPUT,
            output={"action": None, "is_act": False, "skip_reason": "not actionable"},
            expected_output={"acceptable_verdicts": ["sell"], "gain_pct": -2.0},
        )
        by_name = {e.name: e.value for e in evs}
        assert by_name["act_correct"] == 0.0
        assert by_name["act_confusion"] == "FN"

    def test_act_on_flat_is_wrong_and_fp(self):
        evs = act_decision_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "is_act": True, "skip_reason": None},
            expected_output={"acceptable_verdicts": ["buy", "hold"], "gain_pct": 0.1},
        )
        by_name = {e.name: e.value for e in evs}
        assert by_name["act_correct"] == 0.0
        assert by_name["act_confusion"] == "FP"

    def test_skip_on_flat_is_correct_and_tn(self):
        evs = act_decision_evaluator(
            input=ITEM_INPUT,
            output={"action": None, "is_act": False, "skip_reason": "not actionable"},
            expected_output={"acceptable_verdicts": ["sell", "hold"], "gain_pct": -0.1},
        )
        by_name = {e.name: e.value for e in evs}
        assert by_name["act_correct"] == 1.0
        assert by_name["act_confusion"] == "TN"

    def test_no_expected_output_becomes_single_skip_score(self):
        ev = act_decision_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "is_act": True, "skip_reason": None},
            expected_output=None,
        )
        assert ev.name == "act_decision_skip"
        assert "acceptable_verdicts" in ev.value


class TestVerdictEvaluator:
    def test_actionable_acceptable_direction_scores_one(self):
        ev = verdict_evaluator(
            input=ITEM_INPUT,
            output={"action": "hold", "is_act": True, "skip_reason": None},
            expected_output={"acceptable_verdicts": ["sell", "hold"], "gain_pct": -0.1},
        )
        assert ev.name == "verdict_score"
        assert ev.value == 1.0

    def test_actionable_wrong_direction_scores_zero(self):
        ev = verdict_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "is_act": True, "skip_reason": None},
            expected_output={"acceptable_verdicts": ["sell"], "gain_pct": -2.0},
        )
        assert ev.name == "verdict_score"
        assert ev.value == 0.0

    def test_no_act_item_is_excluded(self):
        ev = verdict_evaluator(
            input=ITEM_INPUT,
            output={"action": None, "is_act": False, "skip_reason": "not actionable"},
            expected_output={"acceptable_verdicts": ["buy"]},
        )
        assert ev.name == "verdict_excluded"

    def test_missing_expected_output_becomes_skip_score(self):
        ev = verdict_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "is_act": True, "skip_reason": None},
            expected_output=None,
        )
        assert ev.name == "verdict_score_skip"


class TestPriceMoveEvaluator:
    def test_price_move_recorded_even_for_hold(self):
        ev = price_move_evaluator(
            input=ITEM_INPUT,
            output={"action": "hold", "is_act": True, "skip_reason": None},
            expected_output={"gain_pct": -1.3},
        )
        assert ev.name == "price_move_pct"
        assert ev.value == -1.3

    def test_price_move_skip_score_when_no_prefetched_gain(self):
        ev = price_move_evaluator(
            input=ITEM_INPUT,
            output={"action": "buy", "is_act": True, "skip_reason": None},
            expected_output=None,
        )
        assert ev.name == "price_move_pct_skip"
        assert "gain_pct" in ev.value


def _result(*evals, output=None, expected_output=None):
    return SimpleNamespace(
        evaluations=list(evals),
        output=output,
        item=SimpleNamespace(expected_output=expected_output),
    )


class TestActMetrics:
    def test_computes_accuracy_precision_recall_f1(self):
        # 1 TP, 1 FP, 1 FN, 1 TN -> acc .5, prec .5, rec .5, f1 .5
        results = [
            _result(output={"is_act": True}, expected_output={"acceptable_verdicts": ["buy"]}),       # TP
            _result(output={"is_act": True}, expected_output={"acceptable_verdicts": ["buy", "hold"]}),  # FP
            _result(output={"is_act": False}, expected_output={"acceptable_verdicts": ["sell"]}),      # FN
            _result(output={"is_act": False}, expected_output={"acceptable_verdicts": ["sell", "hold"]}),  # TN
        ]
        evs = act_metrics(item_results=results)
        by_name = {e.name: e.value for e in evs}
        assert by_name["act_accuracy"] == pytest.approx(0.5)
        assert by_name["act_precision"] == pytest.approx(0.5)
        assert by_name["act_recall"] == pytest.approx(0.5)
        assert by_name["act_f1"] == pytest.approx(0.5)

    def test_unscorable_items_are_ignored(self):
        results = [
            _result(output={"is_act": True}, expected_output={"acceptable_verdicts": ["buy"]}),  # TP
            _result(output={"is_act": True}, expected_output=None),  # local item, ignored
        ]
        evs = act_metrics(item_results=results)
        by_name = {e.name: e.value for e in evs}
        assert by_name["act_accuracy"] == 1.0
        assert "n=1" in evs[0].comment

    def test_no_scorable_items(self):
        ev = act_metrics(item_results=[
            _result(output={"is_act": True}, expected_output=None),
        ])
        assert ev.name == "act_metrics_skip"


class TestAvgVerdictScore:
    def test_averages_only_actionable_scored_items(self):
        results = [
            SimpleNamespace(evaluations=[SimpleNamespace(name="verdict_score", value=1.0)]),
            SimpleNamespace(evaluations=[SimpleNamespace(name="verdict_score", value=1.0)]),
            SimpleNamespace(evaluations=[SimpleNamespace(name="verdict_score", value=0.0)]),
            SimpleNamespace(evaluations=[SimpleNamespace(name="verdict_excluded", value="no-act")]),
        ]
        ev = avg_verdict_score(item_results=results)
        assert ev.name == "avg_verdict_score"
        assert ev.value == pytest.approx(2 / 3)
        assert "3 actionable" in ev.comment

    def test_no_scorable_items(self):
        ev = avg_verdict_score(item_results=[])
        assert ev.name == "avg_verdict_score_skip"


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

    def test_all_expands_to_every_skippable_stage(self):
        assert parse_skip_stages("all") == frozenset(SKIPPABLE_STAGES)

    def test_all_cannot_combine_with_other_stages(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            parse_skip_stages("all,embed")


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
        monkeypatch.setattr(stages, "classify_stage", lambda c, u, classify_model=None: None)  # no-op: kept
        monkeypatch.setattr(stages, "tag_stage", lambda c, u, t: None)
        monkeypatch.setattr(stages, "insights_stage", lambda c, u, t: None)
        monkeypatch.setattr(
            stages, "sentiment_stage",
            lambda c, u, precedent_source=None, verdict_model=None: {"ticker": "NVDA", "action": "buy", "confidence": 0.8},
        )
        task = pipeline_eval.make_task(dsn=None, skip_stages=frozenset({"classify"}))
        out = asyncio.run(task(item={"input": {"article_id": 20512, "url": "https://x/20512"}}))
        assert out["category"] == "real news"
        assert out["action"] == "buy"

    def test_task_threads_precedent_source_into_sentiment(self, monkeypatch):
        from ticker_news.evals import pipeline_eval
        from ticker_news.service import stages

        conn = FakeConn(rows=[("real news",)])
        conn.close = lambda: None
        monkeypatch.setattr(pipeline_eval, "connect_eval", lambda dsn: conn)
        monkeypatch.setattr(pipeline_eval, "reset_article",
                            lambda c, aid, keep=frozenset(): None)
        monkeypatch.setattr(stages.TagContext, "load", classmethod(lambda cls, c: None))
        monkeypatch.setattr(stages, "embed_stage", lambda c, u: None)
        monkeypatch.setattr(stages, "classify_stage", lambda c, u, classify_model=None: "real news")
        monkeypatch.setattr(stages, "tag_stage", lambda c, u, t: None)
        monkeypatch.setattr(stages, "insights_stage", lambda c, u, t: None)
        seen = {}
        monkeypatch.setattr(
            stages, "sentiment_stage",
            lambda c, u, precedent_source=None, verdict_model=None: seen.update(src=precedent_source, model=verdict_model)
            or {"ticker": "NVDA", "action": "buy", "confidence": 0.8},
        )
        task = pipeline_eval.make_task(dsn=None, precedent_source="insights")
        asyncio.run(task(item={"input": {"article_id": 20512, "url": "https://x/20512"}}))
        assert seen["src"] == "insights"

    def test_task_threads_verdict_model_into_sentiment(self, monkeypatch):
        from ticker_news.evals import pipeline_eval
        from ticker_news.service import stages

        conn = FakeConn(rows=[("real news",)])
        conn.close = lambda: None
        monkeypatch.setattr(pipeline_eval, "connect_eval", lambda dsn: conn)
        monkeypatch.setattr(pipeline_eval, "reset_article",
                            lambda c, aid, keep=frozenset(): None)
        monkeypatch.setattr(stages.TagContext, "load", classmethod(lambda cls, c: None))
        monkeypatch.setattr(stages, "embed_stage", lambda c, u: None)
        monkeypatch.setattr(stages, "classify_stage", lambda c, u, classify_model=None: "real news")
        monkeypatch.setattr(stages, "tag_stage", lambda c, u, t: None)
        monkeypatch.setattr(stages, "insights_stage", lambda c, u, t: None)
        seen = {}
        monkeypatch.setattr(
            stages, "sentiment_stage",
            lambda c, u, precedent_source=None, verdict_model=None:
                seen.update(model=verdict_model)
                or {"ticker": "NVDA", "action": "buy", "confidence": 0.8},
        )
        task = pipeline_eval.make_task(dsn=None, verdict_model="gemini-2.5-flash")
        asyncio.run(task(item={"input": {"article_id": 20512, "url": "https://x/20512"}}))
        assert seen["model"] == "gemini-2.5-flash"

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
        monkeypatch.setattr(stages, "classify_stage", lambda c, u, classify_model=None: "real news")
        monkeypatch.setattr(stages, "tag_stage", lambda c, u, t: None)
        monkeypatch.setattr(stages, "insights_stage", lambda c, u, t: None)
        monkeypatch.setattr(
            stages, "sentiment_stage",
            lambda c, u, precedent_source=None, verdict_model=None: {"ticker": "NVDA", "action": "buy", "confidence": 0.8},
        )
        task = pipeline_eval.make_task(dsn=None)
        asyncio.run(task(item={"input": {"article_id": 20512, "url": "https://x/20512"}}))
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


class TestRunEvalMetadata:
    """run_eval always records effective precedent_source, even when not passed."""

    class _FakeSettings:
        massive_api_key = "mak"
        google_api_key = "gak"
        openai_api_key = "oak"
        precedent_source = "article"  # the settings default
        sentiment_verdict_model = "gemini-2.5-flash-lite"

    class _FakeLFClient:
        def __init__(self):
            self.experiments: list[dict] = []

        def run_experiment(self, **kwargs):
            self.experiments.append(kwargs)
            return SimpleNamespace(item_results=[])

        def flush(self):
            pass

    def _patch_infra(self, monkeypatch, fake_client):
        """Wire the minimal fakes run_eval needs before the metadata block."""
        from ticker_news.shared import observability as obs

        # Wrap the fake as a callable with cache_clear so the autouse teardown
        # fixture (which calls observability.client.cache_clear()) doesn't break.
        def _client():
            return fake_client
        _client.cache_clear = lambda: None  # satisfy conftest teardown

        monkeypatch.setattr(obs, "client", _client)
        monkeypatch.setattr(
            pipeline_eval, "connect_eval", lambda dsn: SimpleNamespace(close=lambda: None)
        )
        monkeypatch.setattr(pipeline_eval, "ensure_eval_schema", lambda conn: None)
        monkeypatch.setattr(pipeline_eval, "build_items", lambda conn, ids: [
            {"input": {"article_id": ids[0], "url": "https://x/1",
                       "published_utc": "2026-05-28T11:10:00+00:00", "title": "T"}}
        ])
        monkeypatch.setattr(pipeline_eval, "make_task", lambda *a, **kw: lambda **kw2: {})
        monkeypatch.setattr(obs, "flush", lambda: None)

    def _patch_settings(self, monkeypatch):
        """Replace the lru_cache'd get_settings with one returning _FakeSettings.

        run_eval imports get_settings locally, so we must patch the canonical
        location in ticker_news.shared.config and clear the cache first.
        The replacement carries a no-op cache_clear so the conftest teardown
        fixture (which calls get_settings.cache_clear()) doesn't break.
        """
        import ticker_news.shared.config as cfg_mod

        fake = self._FakeSettings()

        def _get_settings():
            return fake
        _get_settings.cache_clear = lambda: None  # satisfy conftest teardown

        monkeypatch.setattr(cfg_mod, "get_settings", _get_settings)

    def test_no_precedent_source_records_settings_default(self, monkeypatch):
        """When precedent_source=None, metadata carries the settings default."""
        from ticker_news.evals import pipeline_eval as pe

        fake_client = self._FakeLFClient()
        self._patch_infra(monkeypatch, fake_client)
        self._patch_settings(monkeypatch)
        # Call with no precedent_source; settings default is "article"
        pe.run_eval([1], dsn=None, precedent_source=None)
        assert fake_client.experiments, "run_experiment was not called"
        metadata = fake_client.experiments[0]["metadata"]
        assert metadata["precedent_source"] == "article"

    def test_explicit_precedent_source_is_preserved(self, monkeypatch):
        """When precedent_source is passed explicitly, it overrides the default."""
        from ticker_news.evals import pipeline_eval as pe

        fake_client = self._FakeLFClient()
        self._patch_infra(monkeypatch, fake_client)
        self._patch_settings(monkeypatch)
        pe.run_eval([1], dsn=None, precedent_source="distilled-second")
        metadata = fake_client.experiments[0]["metadata"]
        assert metadata["precedent_source"] == "distilled-second"


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
