"""Offline tests for the experimental classification variants."""

import asyncio

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


class StubChain:
    """Async chain double recording invocations; returns canned verdicts in order."""

    def __init__(self, *verdicts):
        self._verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def ainvoke(self, inputs, config=None):
        self.calls.append(inputs)
        result = self._verdicts.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _run(coro):
    return asyncio.run(coro)


class TestVariantRunner:
    def test_two_pass_confirms_act_verdicts(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="real news"))
        confirm = StubChain(BinaryClassification(label="none news"))
        runner = VariantRunner(lite=lite, confirm=confirm,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "none news"   # confirm overturned
        assert confirmed is True
        assert len(lite.calls) == 1 and len(confirm.calls) == 1

    def test_two_pass_skips_confirm_for_non_act(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="none news"))
        confirm = StubChain()
        runner = VariantRunner(lite=lite, confirm=confirm,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "none news"
        assert confirmed is False
        assert confirm.calls == []

    def test_two_pass_keeps_lite_verdict_when_confirm_fails(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="real news", confidence=0.8))
        confirm = StubChain(RuntimeError("quota"))
        runner = VariantRunner(lite=lite, confirm=confirm,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "real news"
        assert confirmed is True

    def test_single_pass_lite_never_confirms(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="real news"))
        runner = VariantRunner(lite=lite, confirm=None,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "real news"
        assert confirmed is False

    def test_single_pass_flash_only(self):
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        flash = StubChain(BinaryClassification(label="real news"))
        runner = VariantRunner(lite=None, confirm=flash,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.label == "real news"
        assert confirmed is False
        assert len(flash.calls) == 1

    def test_inputs_truncated_like_production(self):
        from ticker_news.classification.chain import MAX_ARTICLE_CHARS
        from ticker_news.classification.variants import VariantRunner, is_act_binary

        lite = StubChain(BinaryClassification(label="none news"))
        runner = VariantRunner(lite=lite, confirm=None,
                               is_act=is_act_binary, label_of=lambda v: v.label)
        _run(runner.classify("  t  " * 200, "x" * 10_000))
        sent = lite.calls[0]
        assert len(sent["title"]) <= 300
        assert len(sent["body"]) == MAX_ARTICLE_CHARS

    def test_fine_grained_two_pass_uses_news_subtypes(self):
        from ticker_news.classification.variants import (
            VariantRunner, is_act_finegrained,
        )

        lite = StubChain(FinegrainedClassification(category="conference-PR"))
        confirm = StubChain()
        runner = VariantRunner(lite=lite, confirm=confirm,
                               is_act=is_act_finegrained,
                               label_of=lambda v: v.category)
        verdict, confirmed = _run(runner.classify("T", "B"))
        assert verdict.category == "conference-PR"
        assert confirmed is False
        assert confirm.calls == []


class TestMakeRunner:
    def test_rejects_unknown_variant_and_mode(self):
        from ticker_news.classification.variants import make_runner

        with pytest.raises(ValueError, match="variant"):
            make_runner("ternary", "lite")
        with pytest.raises(ValueError, match="mode"):
            make_runner("binary", "warp")
