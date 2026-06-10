import asyncio

import pytest

from ticker_news.service import worker
from ticker_news.service.jobs import DONE, Job


def _job(stage="scrape"):
    return Job(article_url="https://example.com/a", stage=stage, attempts=0,
               tickers=["NVDA"], published_utc=None, publisher=None)


class FakeQueue:
    """Records advance/fail calls instead of touching Postgres."""

    def __init__(self):
        self.advanced = []
        self.failed = []

    def advance(self, conn, url, to_stage):
        self.advanced.append(to_stage)

    def fail(self, conn, url, error, **kw):
        self.failed.append(error)


async def test_process_article_runs_stages_in_order_and_advances():
    ran = []

    async def scrape(job):
        ran.append("scrape")
        return "ok"

    def sync_stage(name):
        def _run(job):
            ran.append(name)
        return _run

    runners = {
        "scrape": scrape,
        "embed": sync_stage("embed"),
        "classify": sync_stage("classify"),
        "tag": sync_stage("tag"),
        "insights": sync_stage("insights"),
    }
    q = FakeQueue()
    await worker.process_article(_job(), runners, q, conn=None)
    assert ran == ["scrape", "embed", "classify", "tag", "insights"]
    assert q.advanced == ["embed", "classify", "tag", "insights", DONE]


async def test_process_article_resumes_mid_chain():
    ran = []
    runners = {
        "tag": lambda job: ran.append("tag"),
        "insights": lambda job: ran.append("insights"),
    }
    q = FakeQueue()
    await worker.process_article(_job(stage="tag"), runners, q, conn=None)
    assert ran == ["tag", "insights"]
    assert q.advanced == ["insights", DONE]


async def test_empty_scrape_short_circuits_to_done():
    async def scrape(job):
        return "empty"

    q = FakeQueue()
    await worker.process_article(_job(), {"scrape": scrape}, q, conn=None)
    assert q.advanced == [DONE]


async def test_stage_failure_calls_fail_not_advance():
    async def scrape(job):
        raise worker.StageError("boom")

    q = FakeQueue()
    await worker.process_article(_job(), {"scrape": scrape}, q, conn=None)
    assert q.advanced == []
    assert len(q.failed) == 1 and "boom" in q.failed[0]


async def test_permanent_failure_parks_immediately():
    async def scrape(job):
        raise worker.PermanentStageError("robots")

    q = FakeQueue()
    await worker.process_article(_job(), {"scrape": scrape}, q, conn=None)
    assert q.advanced == []
    assert len(q.failed) == 1
