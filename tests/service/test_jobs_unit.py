from ticker_news.service.jobs import (
    BACKOFF_CAP_S,
    BASE_BACKOFF_S,
    DONE,
    STAGES,
    backoff_delay,
    next_stage,
)


def test_stage_chain_order():
    assert STAGES == ["scrape", "embed", "classify", "tag", "insights"]


def test_next_stage_walks_the_chain_then_done():
    assert next_stage("scrape") == "embed"
    assert next_stage("insights") == DONE


def test_backoff_is_exponential_and_capped():
    assert backoff_delay(0) == BASE_BACKOFF_S
    assert backoff_delay(1) == BASE_BACKOFF_S * 2
    assert backoff_delay(50) == BACKOFF_CAP_S
