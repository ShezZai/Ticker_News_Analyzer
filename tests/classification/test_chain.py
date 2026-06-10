import pytest
from langchain_core.runnables import RunnableLambda

from ticker_news.classification.chain import MAX_ARTICLE_CHARS, classify_article
from ticker_news.classification.schemas import Classification


def _const(category, reason=""):
    return RunnableLambda(lambda _x: Classification(category=category, reason=reason))


def test_non_real_news_skips_confirmation():
    def explode(_x):
        raise AssertionError("confirmation must not run for non-real-news")

    result, confirmed = classify_article(
        "T", "body", lite=_const("marketing fluff", "ad"), confirm=RunnableLambda(explode)
    )
    assert result.category == "marketing fluff"
    assert confirmed is False


def test_real_news_goes_to_confirmation_and_can_be_overturned():
    result, confirmed = classify_article(
        "T", "body", lite=_const("real news"), confirm=_const("recap/review", "post-hoc")
    )
    assert result.category == "recap/review"
    assert confirmed is True


def test_confirmation_failure_keeps_lite_verdict():
    def boom(_x):
        raise RuntimeError("api down")

    result, confirmed = classify_article(
        "T", "body", lite=_const("real news", "from lite"), confirm=RunnableLambda(boom)
    )
    assert result.category == "real news"
    assert result.reason == "from lite"
    assert confirmed is True


def test_inputs_are_truncated():
    seen = {}

    def capture(x):
        seen.update(x)
        return Classification(category="other")

    classify_article("t" * 1000, "b" * 100_000, lite=RunnableLambda(capture))
    assert len(seen["title"]) == 300
    assert len(seen["body"]) == MAX_ARTICLE_CHARS


def test_invalid_category_rejected_by_schema():
    with pytest.raises(Exception):
        Classification(category="not a real category")
