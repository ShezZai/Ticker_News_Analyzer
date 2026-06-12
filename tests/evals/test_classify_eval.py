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
