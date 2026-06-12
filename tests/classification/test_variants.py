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


class TestClassifier:
    def _binary(self, *verdicts):
        from ticker_news.classification.variants import Classifier, is_act_binary

        chain = StubChain(*verdicts)
        clf = Classifier(
            chain=chain, variant="binary", model="gemini-2.5-flash-lite",
            label_of=lambda v: v.label,
            dataset_label_of=lambda v: "YES" if is_act_binary(v.label) else "NO",
        )
        return clf, chain

    def test_single_llm_call(self):
        clf, chain = self._binary(BinaryClassification(label="real news"))
        verdict = _run(clf.classify("T", "B"))
        assert verdict.label == "real news"
        assert len(chain.calls) == 1

    def test_inputs_truncated_like_production(self):
        from ticker_news.classification.chain import MAX_ARTICLE_CHARS

        clf, chain = self._binary(BinaryClassification(label="none news"))
        _run(clf.classify("  t  " * 200, "x" * 10_000))
        sent = chain.calls[0]
        assert len(sent["title"]) <= 300
        assert len(sent["body"]) == MAX_ARTICLE_CHARS

    def test_run_name_carries_variant(self):
        from ticker_news.classification.variants import Classifier

        seen = {}

        class CfgChain(StubChain):
            async def ainvoke(self, inputs, config=None):
                seen.update(config or {})
                return await super().ainvoke(inputs, config)

        chain = CfgChain(FinegrainedClassification(category="other"))
        clf = Classifier(chain=chain, variant="finegrained", model="m",
                         label_of=lambda v: v.category,
                         dataset_label_of=lambda v: v.category)
        _run(clf.classify("T", "B"))
        assert seen["run_name"] == "classify-finegrained"


class TestDatasetLabelMapping:
    def test_binary_maps_to_yes_no(self):
        from ticker_news.classification.variants import make_classifier

        clf = _make_offline(make_classifier, "binary")
        assert clf.dataset_label_of(BinaryClassification(label="real news")) == "YES"
        assert clf.dataset_label_of(BinaryClassification(label="none news")) == "NO"

    def test_finegrained_passes_category_through(self):
        from ticker_news.classification.variants import make_classifier

        clf = _make_offline(make_classifier, "finegrained")
        v = FinegrainedClassification(category="legal-call")
        assert clf.dataset_label_of(v) == "legal-call"
        assert clf.label_of(v) == "legal-call"


def _make_offline(make_classifier, variant):
    """make_classifier without a Gemini client: stub the chain builder."""
    from unittest.mock import patch

    target = ("ticker_news.classification.variants."
              f"build_{'binary' if variant == 'binary' else 'finegrained'}_classifier")
    with patch(target, return_value=StubChain()):
        return make_classifier(variant, "gemini-2.5-flash-lite")


class TestMakeClassifier:
    def test_rejects_unknown_variant(self):
        from ticker_news.classification.variants import make_classifier

        with pytest.raises(ValueError, match="variant"):
            make_classifier("ternary", "gemini-2.5-flash-lite")

    def test_records_variant_and_model(self):
        from ticker_news.classification.variants import make_classifier

        clf = _make_offline(make_classifier, "binary")
        assert clf.variant == "binary"
        assert clf.model == "gemini-2.5-flash-lite"

    def test_model_choices_map_cli_names(self):
        from ticker_news.classification.variants import MODEL_CHOICES
        from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE

        assert MODEL_CHOICES == {"lite": GEMINI_FLASH_LITE, "flash": GEMINI_FLASH}


class _FakeLLM:
    """Stub for gemini_chat: each builder step returns self until with_retry,
    which must return a real Runnable so `prompt | structured` composes."""

    def with_structured_output(self, schema):
        return self

    def with_config(self, **kw):
        return self

    def with_retry(self, **kw):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda x: x)


class TestBuildLinksPrompt:
    def _patch(self, monkeypatch, prompt_text, prompt_obj):
        from ticker_news.shared import prompts

        monkeypatch.setattr(
            prompts, "get_prompt_entry", lambda name, fb: (prompt_text, prompt_obj)
        )
        monkeypatch.setattr(
            "ticker_news.shared.llm.gemini_chat", lambda m, timeout_s: _FakeLLM()
        )

    def test_build_attaches_langfuse_prompt_metadata(self, monkeypatch):
        from ticker_news.classification import variants

        class FakePromptObj:
            prompt = variants.BINARY_PROMPT_TEMPLATE
            version = 9

        fake = FakePromptObj()
        self._patch(monkeypatch, fake.prompt, fake)
        chain = variants.build_binary_classifier("gemini-2.5-flash-lite")
        # chain is prompt | llm; the first runnable is the ChatPromptTemplate
        assert chain.first.metadata == {"langfuse_prompt": fake}

    def test_bad_remote_template_falls_back_without_link(self, monkeypatch):
        from ticker_news.classification import variants

        class FakePromptObj:
            prompt = "broken {only_title}"
            version = 9

        fake = FakePromptObj()
        self._patch(monkeypatch, fake.prompt, fake)
        chain = variants.build_binary_classifier("gemini-2.5-flash-lite")
        assert set(chain.first.input_variables) == {"title", "body"}
        assert not (chain.first.metadata or {}).get("langfuse_prompt")
