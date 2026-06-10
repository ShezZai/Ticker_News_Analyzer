import pytest

from ticker_news.service import stages
from ticker_news.service.jobs import Job


def _job(url="https://example.com/a"):
    return Job(article_url=url, stage="scrape", attempts=0,
               tickers=["NVDA"], published_utc=None, publisher="Benzinga")


class _Resources:
    """Only what scrape_stage touches."""
    fetcher = store = settings = limiter = robots = None


async def test_scrape_error_raises_stage_error(monkeypatch):
    async def fake_process_job(job, fetcher, store, settings, limiter, robots):
        return "error"

    monkeypatch.setattr(stages, "process_job", fake_process_job)
    with pytest.raises(stages.StageError):
        await stages.scrape_stage(_job(), _Resources())


async def test_scrape_ok_and_empty_pass_through(monkeypatch):
    for result in ("ok", "empty"):
        async def fake_process_job(job, *a, _r=result, **kw):
            return _r

        monkeypatch.setattr(stages, "process_job", fake_process_job)
        assert await stages.scrape_stage(_job(), _Resources()) == result


async def test_scrape_builds_article_job_from_queue_payload(monkeypatch):
    seen = {}

    async def fake_process_job(job, *a, **kw):
        seen["job"] = job
        return "ok"

    monkeypatch.setattr(stages, "process_job", fake_process_job)
    await stages.scrape_stage(_job(), _Resources())
    assert seen["job"].url == "https://example.com/a"
    assert seen["job"].tickers == ["NVDA"]
    assert seen["job"].publisher == "Benzinga"
