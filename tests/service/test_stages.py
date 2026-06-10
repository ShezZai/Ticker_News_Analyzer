import pytest

from ticker_news.service import stages
from ticker_news.service.jobs import Job


def _job(url="https://example.com/a"):
    return Job(article_url=url, stage="scrape", attempts=0,
               tickers=["NVDA"], published_utc=None, publisher="Benzinga")


class _FakeConn:
    """Stub DB connection: execute().fetchone() returns None."""

    def execute(self, sql, params):
        class _R:
            def fetchone(self):
                return None
        return _R()


class _FakeStore:
    """Default stub: exists_ok always returns False, conn returns no rows."""
    conn = _FakeConn()

    def exists_ok(self, url):
        return False


class _Resources:
    """Only what scrape_stage touches."""
    fetcher = settings = limiter = robots = None
    store = _FakeStore()


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


async def test_scrape_skips_already_ok(monkeypatch):
    class FakeStore:
        def exists_ok(self, url):
            return True

    called = {}

    async def fake_process_job(*a, **kw):
        called["yes"] = True
        return "ok"

    monkeypatch.setattr(stages, "process_job", fake_process_job)
    res = _Resources()
    res.store = FakeStore()
    assert await stages.scrape_stage(_job(), res) == "ok"
    assert "yes" not in called


async def test_scrape_robots_block_is_permanent(monkeypatch):
    class FakeConn:
        def execute(self, sql, params):
            class R:
                def fetchone(self):
                    return ("blocked_by_robots",)
            return R()

    class FakeStore:
        conn = FakeConn()

        def exists_ok(self, url):
            return False

    async def fake_process_job(*a, **kw):
        return "error"

    monkeypatch.setattr(stages, "process_job", fake_process_job)
    res = _Resources()
    res.store = FakeStore()
    with pytest.raises(stages.PermanentStageError):
        await stages.scrape_stage(_job(), res)
