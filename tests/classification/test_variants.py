"""Offline tests for the experimental classification variants."""

import pytest
from pydantic import ValidationError

from ticker_news.classification.variants import (
    BINARY_LABELS,
    FINEGRAINED_CATEGORIES,
    NEWS_SUBTYPES,
    BinaryClassification,
    FinegrainedClassification,
    is_act_binary,
    is_act_finegrained,
)


class TestSchemas:
    def test_binary_labels(self):
        assert BINARY_LABELS == ["real news", "none news"]

    def test_binary_schema_accepts_valid_label(self):
        v = BinaryClassification(label="real news", confidence=0.9, reason="earnings")
        assert v.label == "real news"

    def test_binary_schema_rejects_unknown_label(self):
        with pytest.raises(ValidationError):
            BinaryClassification(label="maybe news")

    def test_binary_confidence_and_reason_optional(self):
        v = BinaryClassification(label="none news")
        assert v.confidence is None
        assert v.reason == ""

    def test_finegrained_has_16_categories(self):
        assert len(FINEGRAINED_CATEGORIES) == 16
        assert len(set(FINEGRAINED_CATEGORIES)) == 16

    def test_finegrained_schema_rejects_unknown_category(self):
        with pytest.raises(ValidationError):
            FinegrainedClassification(category="not-a-category")

    def test_finegrained_reason_optional(self):
        v = FinegrainedClassification(category="earnings-reporting")
        assert v.reason == ""


class TestActMapping:
    def test_binary_real_news_is_act(self):
        assert is_act_binary("real news") is True

    def test_binary_none_news_is_not_act(self):
        assert is_act_binary("none news") is False

    def test_news_subtypes_are_the_seven_from_the_spec(self):
        assert NEWS_SUBTYPES == frozenset({
            "earnings-reporting", "dividend-reporting",
            "merger/investment/funding", "legal-event", "MACRO-investment",
            "news-event", "news-report",
        })

    def test_unknown_binary_label_is_not_act(self):
        assert is_act_binary("garbage") is False

    def test_every_finegrained_category_maps(self):
        acts = {c for c in FINEGRAINED_CATEGORIES if is_act_finegrained(c)}
        assert acts == NEWS_SUBTYPES
        non_acts = set(FINEGRAINED_CATEGORIES) - acts
        assert non_acts == {
            "recap/review", "market speculation", "MACRO-political",
            "legal-call", "conference-PR", "marketing fluff", "book PR",
            "Other-filing-reporting", "other",
        }
