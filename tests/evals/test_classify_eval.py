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
