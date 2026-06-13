"""Offline unit tests for the single-pass classification experiments. No DB, no network."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ticker_news.classification.variants import (
    BinaryClassification,
    Classifier,
    is_act_binary,
)
from ticker_news.evals import classify_eval
from ticker_news.evals.classify_eval import (
    EXPERIMENTS,
    GEMINI_PRICES_USD_PER_1M,
    cost_evaluator,
    item_cost_usd,
    label_accuracy_evaluator,
    latency_evaluator,
    predicted_label_evaluator,
)


class TestPriceTable:
    def test_known_models_priced(self):
        assert GEMINI_PRICES_USD_PER_1M["gemini-2.5-flash-lite"] == (0.10, 0.40)
        assert GEMINI_PRICES_USD_PER_1M["gemini-2.5-flash"] == (0.30, 2.50)

    def test_item_cost_flash_lite(self):
        out = {"model": "gemini-2.5-flash-lite", "input_tokens": 1_000_000,
               "output_tokens": 500_000}
        assert item_cost_usd(out) == pytest.approx(0.10 + 0.20)

    def test_item_cost_matches_prefixed_model_name(self):
        out = {"model": "models/gemini-2.5-flash-lite", "input_tokens": 1_000_000,
               "output_tokens": 0}
        assert item_cost_usd(out) == pytest.approx(0.10)

    def test_flash_lite_does_not_match_flash_price(self):
        # "gemini-2.5-flash" is a substring of "gemini-2.5-flash-lite";
        # the longest key must win.
        out = {"model": "gemini-2.5-flash-lite", "input_tokens": 1_000_000,
               "output_tokens": 1_000_000}
        assert item_cost_usd(out) == pytest.approx(0.50)  # not 2.80

    def test_unknown_model_returns_none(self):
        out = {"model": "gpt-x", "input_tokens": 10, "output_tokens": 10}
        assert item_cost_usd(out) is None

    def test_missing_usage_returns_none(self):
        assert item_cost_usd({"model": "gemini-2.5-flash"}) is None
        assert item_cost_usd(None) is None


class TestItemEvaluators:
    def test_correct_label_scores_one(self):
        ev = label_accuracy_evaluator(
            output={"predicted": "real news", "label": "YES"},
            expected_output={"label": "YES"},
        )
        assert ev.name == "label_accuracy"
        assert ev.value == 1.0
        assert "real news" in ev.comment

    def test_wrong_label_scores_zero(self):
        ev = label_accuracy_evaluator(
            output={"predicted": "conference-PR", "label": "conference-PR"},
            expected_output={"label": "legal-call"},
        )
        assert ev.value == 0.0
        assert "legal-call" in ev.comment

    def test_missing_output_scores_zero(self):
        ev = label_accuracy_evaluator(output=None, expected_output={"label": "NO"})
        assert ev.value == 0.0
        assert "no output" in ev.comment

    def test_no_expected_label_skips(self):
        ev = label_accuracy_evaluator(
            output={"predicted": "x", "label": "YES"}, expected_output=None
        )
        assert ev.name == "label_accuracy_skip"

    def test_predicted_label_is_categorical(self):
        ev = predicted_label_evaluator(output={"predicted": "legal-call"})
        assert ev.name == "predicted_label"
        assert ev.value == "legal-call"

    def test_predicted_label_handles_missing_output(self):
        ev = predicted_label_evaluator(output=None)
        assert ev.value == "<none>"

    def test_latency_numeric(self):
        ev = latency_evaluator(output={"latency_s": 1.25})
        assert ev.name == "latency_s"
        assert ev.value == 1.25

    def test_latency_skips_when_absent(self):
        ev = latency_evaluator(output=None)
        assert ev.name == "latency_s_skip"

    def test_cost_numeric(self):
        ev = cost_evaluator(output={"model": "gemini-2.5-flash-lite",
                                    "input_tokens": 1000, "output_tokens": 100})
        assert ev.name == "cost_usd"
        assert ev.value == pytest.approx((1000 * 0.10 + 100 * 0.40) / 1e6)

    def test_cost_skips_unknown_model(self):
        ev = cost_evaluator(output={"model": "mystery", "input_tokens": 1,
                                    "output_tokens": 1})
        assert ev.name == "cost_usd_skip"
        assert "mystery" in ev.value


def _ir(expected_label, output):
    """Minimal stand-in for an SDK item result: .item and .output."""
    item = {"expected_output": {"label": expected_label} if expected_label else None}
    return SimpleNamespace(item=item, output=output)


def _out(label, *, latency=1.0, tin=1000, tout=100, model="gemini-2.5-flash-lite"):
    return {"predicted": label, "label": label, "latency_s": latency,
            "input_tokens": tin, "output_tokens": tout, "model": model}


def _by_name(evaluations):
    return {e.name: e for e in evaluations}


class TestTotalsRunEvaluator:
    def test_time_cost_token_totals(self, monkeypatch):
        from ticker_news.evals.classify_eval import make_totals_run_evaluator

        monkeypatch.setattr(classify_eval.time, "monotonic", lambda: 110.0)
        totals = make_totals_run_evaluator(started_monotonic=100.0)
        results = [
            _ir("YES", _out("YES", latency=2.0)),
            _ir("NO", _out("NO", latency=4.0)),
        ]
        evals = _by_name(totals(item_results=results))
        assert evals["total_time_s"].value == pytest.approx(10.0)
        assert evals["avg_time_per_item_s"].value == pytest.approx(3.0)
        per_item = (1000 * 0.10 + 100 * 0.40) / 1e6
        assert evals["total_cost_usd"].value == pytest.approx(2 * per_item)
        assert evals["total_tokens"].value == 2200
        assert "input=2000" in evals["total_tokens"].comment

    def test_errored_items_excluded_from_averages_but_not_total_time(self, monkeypatch):
        from ticker_news.evals.classify_eval import make_totals_run_evaluator

        monkeypatch.setattr(classify_eval.time, "monotonic", lambda: 105.0)
        totals = make_totals_run_evaluator(started_monotonic=100.0)
        evals = _by_name(totals(item_results=[_ir("YES", None), _ir("NO", _out("NO", latency=2.0))]))
        assert evals["total_time_s"].value == pytest.approx(5.0)
        assert evals["avg_time_per_item_s"].value == pytest.approx(2.0)
        assert "1/2" in evals["total_cost_usd"].comment

    def test_empty_run_only_reports_total_time(self, monkeypatch):
        from ticker_news.evals.classify_eval import make_totals_run_evaluator

        monkeypatch.setattr(classify_eval.time, "monotonic", lambda: 101.0)
        evals = _by_name(make_totals_run_evaluator(100.0)(item_results=[]))
        assert set(evals) == {"total_time_s"}


class TestLabelAccuracyRunEvaluator:
    def test_average_counts_errored_as_wrong(self):
        from ticker_news.evals.classify_eval import label_accuracy_run_evaluator

        results = [
            _ir("YES", _out("YES")), _ir("NO", _out("YES")), _ir("YES", None),
        ]
        evals = _by_name(label_accuracy_run_evaluator(item_results=results))
        assert evals["label_accuracy_avg"].value == pytest.approx(1 / 3)

    def test_unlabeled_items_excluded(self):
        from ticker_news.evals.classify_eval import label_accuracy_run_evaluator

        results = [_ir(None, _out("YES")), _ir("YES", _out("YES"))]
        evals = _by_name(label_accuracy_run_evaluator(item_results=results))
        assert evals["label_accuracy_avg"].value == pytest.approx(1.0)

    def test_all_unlabeled_skips(self):
        from ticker_news.evals.classify_eval import label_accuracy_run_evaluator

        evals = _by_name(label_accuracy_run_evaluator(item_results=[_ir(None, _out("x"))]))
        assert set(evals) == {"label_accuracy_avg_skip"}


class TestBinaryConfusionRunEvaluator:
    def test_confusion_metrics(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        results = [
            _ir("YES", _out("YES")), _ir("YES", _out("YES")),  # TP x2
            _ir("NO", _out("YES")),                            # FP
            _ir("YES", _out("NO")),                            # FN
            _ir("NO", _out("NO")), _ir("NO", _out("NO")),      # TN x2
        ]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert evals["act_precision"].value == pytest.approx(2 / 3)
        assert evals["act_recall"].value == pytest.approx(2 / 3)
        assert evals["act_f1"].value == pytest.approx(2 / 3)
        assert "TP=2 FP=1 FN=1 TN=2" in evals["act_precision"].comment

    def test_errored_yes_item_is_false_negative(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        results = [_ir("YES", None), _ir("YES", _out("YES"))]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert evals["act_recall"].value == pytest.approx(0.5)

    def test_errored_no_item_is_false_positive(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        # an errored task on a NO item must hurt (FP), not count as TN:
        # TP=0 FP=1 -> precision exists and is 0.0; no YES items -> recall skips
        results = [_ir("NO", None), _ir("NO", _out("NO"))]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert evals["act_precision"].value == 0.0
        assert "TP=0 FP=1 FN=0 TN=1" in evals["act_precision"].comment
        assert "no YES items" in evals["act_recall_skip"].value

    def test_no_yes_predictions_skips_precision(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        results = [_ir("YES", _out("NO")), _ir("NO", _out("NO"))]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert "act_precision" not in evals
        assert "no YES predictions" in evals["act_precision_skip"].value
        assert evals["act_recall"].value == 0.0
        assert "act_f1" not in evals

    def test_empty_skips_everything(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        evals = _by_name(binary_confusion_run_evaluator(item_results=[]))
        assert set(evals) == {"act_metrics_skip"}

    def test_malformed_output_on_no_item_is_false_positive(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        # a present-but-labelless output must hurt like an errored one
        results = [_ir("NO", {}), _ir("NO", _out("NO"))]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert "TP=0 FP=1 FN=0 TN=1" in evals["act_precision"].comment

    def test_f1_is_zero_not_absent_when_all_wrong(self):
        from ticker_news.evals.classify_eval import binary_confusion_run_evaluator

        # TP=0 FP=1 FN=1 -> P=0, R=0 -> F1 must be 0.0, not missing
        results = [_ir("NO", _out("YES")), _ir("YES", _out("NO"))]
        evals = _by_name(binary_confusion_run_evaluator(item_results=results))
        assert evals["act_f1"].value == 0.0


class TestDerivedActRunEvaluator:
    def test_miscategorization_within_same_side_still_counts(self):
        from ticker_news.evals.classify_eval import derived_act_run_evaluator

        results = [
            # expected NEWS subtype, predicted different NEWS subtype -> act agrees
            _ir("earnings-reporting", _out("news-event")),
            # expected NOT-NEWS, predicted NOT-NEWS -> act agrees
            _ir("legal-call", _out("marketing fluff")),
            # crosses the boundary -> act disagrees
            _ir("legal-event", _out("legal-call")),
        ]
        evals = _by_name(derived_act_run_evaluator(item_results=results))
        assert evals["derived_act_accuracy"].value == pytest.approx(2 / 3)

    def test_errored_item_counts_as_wrong(self):
        from ticker_news.evals.classify_eval import derived_act_run_evaluator

        results = [_ir("recap/review", None), _ir("recap/review", _out("other"))]
        evals = _by_name(derived_act_run_evaluator(item_results=results))
        assert evals["derived_act_accuracy"].value == pytest.approx(0.5)

    def test_no_labeled_items_skips(self):
        from ticker_news.evals.classify_eval import derived_act_run_evaluator

        evals = _by_name(derived_act_run_evaluator(item_results=[_ir(None, _out("x"))]))
        assert set(evals) == {"derived_act_accuracy_skip"}


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows=None):
        self.executed = []
        self._rows = rows or []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return FakeCursor(self._rows)

    def close(self):
        pass


class TestFinegrainedConfusionRunEvaluator:
    def test_per_class_macro_and_weighted_metrics(self):
        from ticker_news.evals.classify_eval import finegrained_confusion_run_evaluator

        # gold -> predicted
        results = [
            _ir("A", _out("A")), _ir("A", _out("A")),  # tp_A x2
            _ir("A", _out("B")),                        # fn_A, fp_B
            _ir("B", _out("B")),                        # tp_B
            _ir("B", _out("A")),                        # fn_B, fp_A
            _ir("C", _out("C")),                        # tp_C
        ]
        # A: tp=2 fp=1 fn=1 -> P=2/3 R=2/3 F1=2/3
        # B: tp=1 fp=1 fn=1 -> P=1/2 R=1/2 F1=1/2
        # C: tp=1 fp=0 fn=0 -> P=1   R=1   F1=1
        evals = _by_name(finegrained_confusion_run_evaluator(item_results=results))
        macro = (2 / 3 + 1 / 2 + 1) / 3
        assert evals["cat_macro_precision"].value == pytest.approx(macro)
        assert evals["cat_macro_recall"].value == pytest.approx(macro)
        assert evals["cat_macro_f1"].value == pytest.approx(macro)
        # weighted by gold support (A=3, B=2, C=1): (3*2/3 + 2*1/2 + 1*1)/6
        assert evals["cat_weighted_f1"].value == pytest.approx(4 / 6)
        # the comment carries a per-class breakdown
        assert "A" in evals["cat_macro_f1"].comment

    def test_errored_item_is_false_negative_only(self):
        from ticker_news.evals.classify_eval import finegrained_confusion_run_evaluator

        # an errored (None) prediction hurts recall of its gold class but is
        # never a false positive for any class.
        results = [_ir("A", _out("A")), _ir("A", None)]
        # A: tp=1 fp=0 fn=1 -> P=1.0 R=0.5
        evals = _by_name(finegrained_confusion_run_evaluator(item_results=results))
        assert evals["cat_macro_precision"].value == pytest.approx(1.0)
        assert evals["cat_macro_recall"].value == pytest.approx(0.5)

    def test_spurious_predicted_class_penalizes_macro_not_weighted(self):
        from ticker_news.evals.classify_eval import finegrained_confusion_run_evaluator

        # predicting a class that has no gold support adds a zero-metric class
        # to the macro average; weighted-F1 (gold support) is unaffected.
        results = [_ir("A", _out("A")), _ir("A", _out("Z"))]
        # A: tp=1 fp=0 fn=1 -> P=1 R=0.5 F1=2/3 ; Z: tp=0 fp=1 -> all 0
        evals = _by_name(finegrained_confusion_run_evaluator(item_results=results))
        assert evals["cat_macro_precision"].value == pytest.approx(0.5)
        assert evals["cat_macro_recall"].value == pytest.approx(0.25)
        assert evals["cat_weighted_f1"].value == pytest.approx(2 / 3)

    def test_unlabeled_items_excluded(self):
        from ticker_news.evals.classify_eval import finegrained_confusion_run_evaluator

        results = [_ir(None, _out("A")), _ir("A", _out("A"))]
        evals = _by_name(finegrained_confusion_run_evaluator(item_results=results))
        assert evals["cat_macro_recall"].value == pytest.approx(1.0)

    def test_no_labeled_items_skips(self):
        from ticker_news.evals.classify_eval import finegrained_confusion_run_evaluator

        evals = _by_name(finegrained_confusion_run_evaluator(item_results=[_ir(None, _out("A"))]))
        assert set(evals) == {"cat_metrics_skip"}


class TestPrefetchArticles:
    def test_returns_id_to_title_content(self):
        from ticker_news.evals.classify_eval import prefetch_articles

        conn = FakeConn(rows=[(595, "Title 595", "Body 595"), (14682, None, "Body")])
        articles = prefetch_articles(conn, [595, 14682])
        assert articles == {595: ("Title 595", "Body 595"), 14682: ("", "Body")}
        # one parametrized query for all ids
        assert len(conn.executed) == 1
        assert conn.executed[0][1] == ([595, 14682],)

    def test_missing_ids_raise(self):
        from ticker_news.evals.classify_eval import prefetch_articles

        conn = FakeConn(rows=[(595, "T", "B")])
        with pytest.raises(ValueError, match="not found.*14682"):
            prefetch_articles(conn, [595, 14682])

    def test_empty_content_raises(self):
        from ticker_news.evals.classify_eval import prefetch_articles

        conn = FakeConn(rows=[(595, "T", "   ")])
        with pytest.raises(ValueError, match="no scraped content.*595"):
            prefetch_articles(conn, [595])


class StubChain:
    def __init__(self, *verdicts):
        self._verdicts = list(verdicts)
        self.calls = []
        self.configs = []

    async def ainvoke(self, inputs, config=None):
        self.calls.append(inputs)
        self.configs.append(config or {})
        return self._verdicts.pop(0)


def _binary_classifier(*verdicts):
    chain = StubChain(*verdicts)
    return Classifier(
        chain=chain, variant="binary", model="gemini-2.5-flash-lite",
        label_of=lambda v: v.label,
        dataset_label_of=lambda v: "YES" if is_act_binary(v.label) else "NO",
    ), chain


class TestMakeTask:
    def test_task_returns_output_dict(self):
        from ticker_news.evals.classify_eval import make_task

        clf, chain = _binary_classifier(
            BinaryClassification(label="real news", confidence=0.7, reason="earnings")
        )
        task = make_task(clf, {595: ("Title", "Body text")}, "classify-binary")
        out = asyncio.run(task(item={"input": {"article_id": 595}}))
        assert out["predicted"] == "real news"
        assert out["label"] == "YES"
        assert out["confidence"] == 0.7
        assert out["reason"] == "earnings"
        assert out["model"] == "gemini-2.5-flash-lite"
        assert out["latency_s"] >= 0
        # no usage captured from the stub chain -> tokens None, cost skips
        assert out["input_tokens"] is None
        assert out["output_tokens"] is None
        # prefetched text was used; no DB call possible (no conn anywhere)
        assert chain.calls[0]["title"] == "Title"

    def test_task_appends_usage_handler_to_config(self):
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        from ticker_news.evals.classify_eval import make_task

        clf, chain = _binary_classifier(BinaryClassification(label="none news"))
        task = make_task(clf, {1: ("T", "B")}, "classify-binary")
        asyncio.run(task(item={"input": {"article_id": 1}}))
        callbacks = chain.configs[0].get("callbacks", [])
        assert any(isinstance(cb, UsageMetadataCallbackHandler) for cb in callbacks)

    def test_task_names_the_trace(self, monkeypatch):
        import langfuse

        from ticker_news.evals.classify_eval import make_task

        seen = {}

        @contextmanager
        def fake_propagate(**kwargs):
            seen.update(kwargs)
            yield

        monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
        clf, _ = _binary_classifier(BinaryClassification(label="none news"))
        task = make_task(clf, {595: ("T", "B")}, "classify-binary")
        asyncio.run(task(item={"input": {"article_id": 595}}))
        assert seen["trace_name"] == "classify-binary:article-595"


class TestExperimentSpecs:
    def test_spec_table(self):
        assert set(EXPERIMENTS) == {"binary", "finegrained"}
        b, f = EXPERIMENTS["binary"], EXPERIMENTS["finegrained"]
        assert b.dataset == "140-articles-act-no-act"
        assert b.experiment_name == "classify-binary"
        assert f.dataset == "140-articles-categories"
        assert f.experiment_name == "classify-finegrained"

    def test_binary_has_confusion_finegrained_has_derived_act(self):
        from ticker_news.evals.classify_eval import (
            binary_confusion_run_evaluator,
            derived_act_run_evaluator,
            label_accuracy_run_evaluator,
        )

        assert binary_confusion_run_evaluator in EXPERIMENTS["binary"].run_evaluators
        assert derived_act_run_evaluator in EXPERIMENTS["finegrained"].run_evaluators
        for spec in EXPERIMENTS.values():
            assert label_accuracy_run_evaluator in spec.run_evaluators
            assert label_accuracy_evaluator in spec.evaluators
            assert cost_evaluator in spec.evaluators

    def test_finegrained_has_category_confusion(self):
        from ticker_news.evals.classify_eval import finegrained_confusion_run_evaluator

        assert (finegrained_confusion_run_evaluator
                in EXPERIMENTS["finegrained"].run_evaluators)
        assert (finegrained_confusion_run_evaluator
                not in EXPERIMENTS["binary"].run_evaluators)


class TestWarnFailedItems:
    def _result(self, *article_ids):
        return SimpleNamespace(item_results=[
            SimpleNamespace(item={"input": {"article_id": aid}}, output=None)
            for aid in article_ids
        ])

    def test_silent_when_all_items_completed(self, capsys):
        from ticker_news.evals.classify_eval import _warn_failed_items

        _warn_failed_items(self._result(1, 2), [1, 2], "binary", "lite", "run-x")
        assert capsys.readouterr().out == ""

    def test_warns_with_missing_ids_and_rerun_hint(self, capsys):
        from ticker_news.evals.classify_eval import _warn_failed_items

        _warn_failed_items(self._result(1), [1, 2, 3], "binary", "lite",
                           "binary-lite-20260613-101500")
        out = capsys.readouterr().out
        assert "2 item(s) errored" in out
        assert "[2, 3]" in out
        assert "--variant binary --model lite" in out
        assert "--ids 2,3" in out
        assert "--run-name binary-lite-20260613-101500" in out


class TestWithMissingItems:
    def _dataset_item(self, article_id, expected_label):
        return {"input": {"article_id": article_id},
                "expected_output": {"label": expected_label}}

    def test_sdk_dropped_items_count_as_wrong(self):
        from ticker_news.evals.classify_eval import (
            _with_missing_items,
            binary_confusion_run_evaluator,
            label_accuracy_run_evaluator,
        )

        data = [self._dataset_item(1, "YES"), self._dataset_item(2, "NO"),
                self._dataset_item(3, "YES")]
        # the SDK delivered only item 1 (correct); items 2 and 3 errored away
        survived = [_ir("YES", _out("YES"))]
        survived[0].item["input"] = {"article_id": 1}

        acc = _by_name(_with_missing_items(label_accuracy_run_evaluator, data)(
            item_results=survived))
        assert acc["label_accuracy_avg"].value == pytest.approx(1 / 3)

        conf = _by_name(_with_missing_items(binary_confusion_run_evaluator, data)(
            item_results=survived))
        # item 3 (YES, errored) -> FN; item 2 (NO, errored) -> FP
        assert "TP=1 FP=1 FN=1 TN=0" in conf["act_precision"].comment

    def test_no_injection_when_all_items_survived(self):
        from ticker_news.evals.classify_eval import (
            _with_missing_items,
            label_accuracy_run_evaluator,
        )

        data = [self._dataset_item(1, "YES")]
        survived = [_ir("YES", _out("YES"))]
        survived[0].item["input"] = {"article_id": 1}
        acc = _by_name(_with_missing_items(label_accuracy_run_evaluator, data)(
            item_results=survived))
        assert acc["label_accuracy_avg"].value == pytest.approx(1.0)
