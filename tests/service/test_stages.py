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


async def test_scrape_persists_provider_sentiments(monkeypatch):
    writes = {}

    class FakeConn:
        def execute(self, sql, params):
            writes["params"] = params

            class R:
                def fetchone(self):
                    return None
            return R()

    class FakeStore:
        conn = FakeConn()

        def exists_ok(self, url):
            return True  # skip path must STILL persist sentiments

    res = _Resources()
    res.store = FakeStore()
    job = _job()
    job.source_meta = {"sentiments": {"NVDA": {"sentiment": "positive"}}}
    assert await stages.scrape_stage(job, res) == "ok"
    assert writes["params"][1] == "https://example.com/a"


# ---------------------------------------------------------------------------
# sentiment_stage offline tests
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._row


class _StubConn:
    """Returns queued rows in order; records rollbacks."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.rolled_back = 0

    def execute(self, sql, params=None):
        return _Row(self.rows.pop(0))

    def rollback(self):
        self.rolled_back += 1

    def commit(self):
        pass


async def test_sentiment_skips_non_real_news(monkeypatch):
    conn = _StubConn([(1, "T", "body", "marketing fluff", "NVDA", None, None)])
    called = {}
    monkeypatch.setattr(stages, "judge_article", lambda a: called.setdefault("x", True))
    stages.sentiment_stage(conn, "https://example.com/a")
    assert "x" not in called
    assert conn.rolled_back == 1


async def test_sentiment_skips_untagged(monkeypatch):
    conn = _StubConn([(1, "T", "body", "real news", None, None, None)])
    monkeypatch.setattr(stages, "judge_article", lambda a: (_ for _ in ()).throw(AssertionError))
    stages.sentiment_stage(conn, "https://example.com/a")
    assert conn.rolled_back == 1


async def test_sentiment_skips_empty_content(monkeypatch):
    conn = _StubConn([(1, "T", "   ", "real news", "NVDA", None, None)])
    monkeypatch.setattr(stages, "judge_article", lambda a: (_ for _ in ()).throw(AssertionError))
    stages.sentiment_stage(conn, "https://example.com/a")
    assert conn.rolled_back == 1


async def test_sentiment_judges_real_news(monkeypatch):
    from ticker_news.sentiment.schemas import Verdict

    conn = _StubConn([
        (1, "T", "body", "real news", "NVDA", None, {"NVDA": {"sentiment": "positive"}}),
    ])
    seen = {}
    monkeypatch.setattr(stages.sentiment_store, "has_verdict", lambda c, a, t: False)
    monkeypatch.setattr(stages, "similar_past_articles", lambda c, a, k=5: ["p1"])
    monkeypatch.setattr(
        stages, "judge_article",
        lambda article: (seen.update(article) or
                         (Verdict(action="hold", confidence=0.5, reasoning=""), [])),
    )
    saved = {}
    monkeypatch.setattr(
        stages.sentiment_store, "save_verdict",
        lambda c, aid, t, v, an, m: saved.update(aid=aid, ticker=t, action=v.action),
    )
    stages.sentiment_stage(conn, "https://example.com/a")
    assert seen["provider_sentiment"] == "positive"
    assert seen["precedents"] == ["p1"]
    assert saved == {"aid": 1, "ticker": "NVDA", "action": "hold"}
