import asyncio
from contextlib import contextmanager

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
        "sentiment": sync_stage("sentiment"),
    }
    q = FakeQueue()
    await worker.process_article(_job(), runners, q, conn=None)
    assert ran == ["scrape", "embed", "classify", "tag", "insights", "sentiment"]
    assert q.advanced == ["embed", "classify", "tag", "insights", "sentiment", DONE]


async def test_process_article_resumes_mid_chain():
    ran = []
    runners = {
        "tag": lambda job: ran.append("tag"),
        "insights": lambda job: ran.append("insights"),
        "sentiment": lambda job: ran.append("sentiment"),
    }
    q = FakeQueue()
    await worker.process_article(_job(stage="tag"), runners, q, conn=None)
    assert ran == ["tag", "insights", "sentiment"]
    assert q.advanced == ["insights", "sentiment", DONE]


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


class FakeRoot:
    """Captures root.update kwargs the way a Langfuse span would receive them."""

    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


def _fake_trace(monkeypatch, root):
    @contextmanager
    def fake_article_trace(url, *, ticker=None, entrypoint="service"):
        yield root

    monkeypatch.setattr(worker.obs, "article_trace", fake_article_trace)


async def test_root_output_carries_category_and_verdict(monkeypatch):
    root = FakeRoot()
    _fake_trace(monkeypatch, root)
    verdict = {"ticker": "NVDA", "action": "buy", "confidence": 0.9}
    runners = {
        "classify": lambda job: "real news",
        "tag": lambda job: None,
        "insights": lambda job: None,
        "sentiment": lambda job: verdict,
    }
    q = FakeQueue()
    assert await worker.process_article(_job(stage="classify"), runners, q, conn=None)
    (kw,) = root.updates
    assert kw["output"] == {
        "final_stage": DONE, "ok": True,
        "category": "real news", "verdict": verdict,
    }
    assert "metadata" in kw


async def test_root_metadata_carries_prompt_versions(monkeypatch):
    from ticker_news.shared import prompts as prompts_mod

    root = FakeRoot()
    _fake_trace(monkeypatch, root)
    monkeypatch.setitem(prompts_mod._seen_versions, "classify-article", 3)
    q = FakeQueue()
    await worker.process_article(
        _job(stage="sentiment"), {"sentiment": lambda job: None}, q, conn=None)
    (kw,) = root.updates
    assert kw["metadata"] == {"prompt_versions": {"classify-article": 3}}


async def test_root_error_output_keeps_partial_summary(monkeypatch):
    root = FakeRoot()
    _fake_trace(monkeypatch, root)

    def tag(job):
        raise worker.StageError("boom")

    runners = {"classify": lambda job: "real news", "tag": tag}
    q = FakeQueue()
    ok = await worker.process_article(_job(stage="classify"), runners, q, conn=None)
    assert ok is False
    (kw,) = root.updates
    assert kw["level"] == "ERROR"
    assert "boom" in kw["status_message"]
    assert kw["output"] == {"final_stage": "tag", "ok": False, "category": "real news"}
    assert "metadata" in kw


async def test_root_output_on_empty_scrape_early_return(monkeypatch):
    root = FakeRoot()
    _fake_trace(monkeypatch, root)

    async def scrape(job):
        return "empty"

    q = FakeQueue()
    await worker.process_article(_job(), {"scrape": scrape}, q, conn=None)
    (kw,) = root.updates
    assert kw["output"] == {"final_stage": DONE, "ok": True}
    assert "metadata" in kw


async def test_process_article_unchanged_under_disabled_observability(monkeypatch):
    # belt-and-braces: explicitly disabled, full chain still runs in order
    from ticker_news.shared.config import get_settings
    get_settings.cache_clear()
    from ticker_news.shared import observability as obs
    obs.client.cache_clear()
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    ran = []
    runners = {s: (lambda job, _s=s: ran.append(_s)) for s in
               ["embed", "classify", "tag", "insights", "sentiment"]}
    q = FakeQueue()
    await worker.process_article(_job(stage="embed"), runners, q, conn=None)
    assert ran == ["embed", "classify", "tag", "insights", "sentiment"]
