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


async def test_sentiment_skips_non_actionable_legacy(monkeypatch):
    # is_act NULL -> legacy fallback to category != 'real news' -> skip
    conn = _StubConn([(1, "T", "body", "marketing fluff", None, "NVDA", None)])
    called = {}
    monkeypatch.setattr(stages, "judge_article", lambda a, config=None, model=None, summary_model=None: called.setdefault("x", True))
    assert stages.sentiment_stage(conn, "https://example.com/a") is None
    assert "x" not in called
    assert conn.rolled_back == 1


async def test_sentiment_skips_non_act_finegrained(monkeypatch):
    # is_act=False overrides even an actionable-looking finegrained category
    conn = _StubConn([(1, "T", "body", "recap/review", False, "NVDA", None)])
    monkeypatch.setattr(stages, "judge_article", lambda a, config=None, model=None, summary_model=None: (_ for _ in ()).throw(AssertionError))
    assert stages.sentiment_stage(conn, "https://example.com/a") is None
    assert conn.rolled_back == 1


async def test_sentiment_skips_untagged(monkeypatch):
    conn = _StubConn([(1, "T", "body", "real news", None, None, None)])
    monkeypatch.setattr(stages, "judge_article", lambda a, config=None, model=None, summary_model=None: (_ for _ in ()).throw(AssertionError))
    assert stages.sentiment_stage(conn, "https://example.com/a") is None
    assert conn.rolled_back == 1


async def test_sentiment_skips_empty_content(monkeypatch):
    conn = _StubConn([(1, "T", "   ", "real news", None, "NVDA", None)])
    monkeypatch.setattr(stages, "judge_article", lambda a, config=None, model=None, summary_model=None: (_ for _ in ()).throw(AssertionError))
    assert stages.sentiment_stage(conn, "https://example.com/a") is None
    assert conn.rolled_back == 1


async def test_sentiment_skips_existing_verdict(monkeypatch):
    conn = _StubConn([(1, "T", "body", "real news", None, "NVDA", None)])
    monkeypatch.setattr(stages.sentiment_store, "has_verdict", lambda c, a, t: True)
    monkeypatch.setattr(stages, "judge_article", lambda a, config=None, model=None, summary_model=None: (_ for _ in ()).throw(AssertionError))
    assert stages.sentiment_stage(conn, "https://example.com/a") is None
    assert conn.rolled_back == 1


async def test_sentiment_skips_no_insights(monkeypatch):
    # actionable + tagged + has content, but the insights stage extracted no
    # boxes -> skip the verdict, ending the pipeline for the item.
    conn = _StubConn([(1, "T", "body", "real news", None, "NVDA", None)])
    monkeypatch.setattr(stages.sentiment_store, "has_verdict", lambda c, a, t: False)
    monkeypatch.setattr(stages, "_has_insights", lambda c, a: False)
    monkeypatch.setattr(stages, "judge_article", lambda a, config=None, model=None, summary_model=None: (_ for _ in ()).throw(AssertionError))
    assert stages.sentiment_stage(conn, "https://example.com/a") is None
    assert conn.rolled_back == 1


async def test_sentiment_judges_act_finegrained(monkeypatch):
    # is_act=True on a finegrained category drives judging (new path)
    from ticker_news.sentiment.schemas import Verdict

    conn = _StubConn([
        (1, "T", "body", "news-event", True, "NVDA", None),
    ])
    seen = {}
    monkeypatch.setattr(stages.sentiment_store, "has_verdict", lambda c, a, t: False)
    monkeypatch.setattr(stages, "_has_insights", lambda c, a: True)
    monkeypatch.setattr(stages, "gather_precedents", lambda c, a, source=None: ["p1"])
    monkeypatch.setattr(
        stages, "judge_article",
        lambda article, config=None, model=None, summary_model=None: (seen.update(article) or
                                      (Verdict(action="hold", confidence=0.5, reasoning=""), [])),
    )
    saved = {}
    monkeypatch.setattr(
        stages.sentiment_store, "save_verdict",
        lambda c, aid, t, v, an, m: saved.update(aid=aid, ticker=t, action=v.action),
    )
    result = stages.sentiment_stage(conn, "https://example.com/a")
    assert seen["precedents"] == ["p1"]
    assert saved == {"aid": 1, "ticker": "NVDA", "action": "hold"}
    assert result == {"ticker": "NVDA", "action": "hold", "confidence": 0.5}


from datetime import datetime, timezone


class _InsightsCursor:
    """Cursor stub: no-ops GUC/savepoint statements, pops a result list per SELECT."""

    def __init__(self, per_box_results):
        self._results = list(per_box_results)
        self._last: list = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        low = sql.lower()
        if "set_config" in low or "savepoint" in low:
            return self  # ef_search / iterative_scan tuning - irrelevant to stub
        self._last = self._results.pop(0)
        return self

    def fetchall(self):
        return self._last


class _InsightsConn:
    """conn.execute serves published_utc + box embeddings; cursor serves hits."""

    def __init__(self, published, boxes, per_box_results):
        self._top = [(published,), boxes]
        self._cursor = _InsightsCursor(per_box_results)

    def execute(self, sql, params=None):
        return _Row(self._top.pop(0))

    def cursor(self):
        return self._cursor


def test_insights_similarity_groups_by_article():
    # rows: (insight_id, article_id, date, ticker, headline, insight, topic,
    #        label, sim) - unlabelled corpus -> None label
    published = datetime(2025, 6, 1, tzinfo=timezone.utc)
    conn = _InsightsConn(
        published,
        [([0.0],), ([0.1],)],               # two embedded boxes
        [
            # box 1 hits: two insights from article 100, one from article 200
            [(10, 100, "2025-01-01", "NVDA", "NVDA beats", "insight A", "topicA", None, 0.91),
             (11, 100, "2025-01-01", "NVDA", "NVDA beats", None, "topic\nB", None, 0.80),
             (12, 200, "2025-01-02", "AMD", "AMD news", "insight C", None, None, 0.85)],
            # box 2 hits: insight 10 again, higher sim -> wins the dedup
            [(10, 100, "2025-01-01", "NVDA", "NVDA beats", "insight A", "topicA", None, 0.95)],
        ],
    )
    lines = stages.insights_similarity(conn, 1)
    assert lines == [
        # article 100 ranks first (best insight sim .95), >1 excerpt -> nested
        "2025-01-01 [NVDA] NVDA beats\n    - insight A\n    - topic B",
        # article 200, single excerpt -> inlined after the em dash
        "2025-01-02 [AMD] AMD news — insight C",
    ]


def test_insights_similarity_tags_label_on_distilled():
    published = datetime(2025, 6, 1, tzinfo=timezone.utc)
    conn = _InsightsConn(
        published,
        [([0.0],)],                          # one embedded box
        [[
            (10, 100, "2025-01-01", "NVDA", "NVDA beats", "insight A", "t", "evidance-event", 0.95),
            (11, 100, "2025-01-01", "NVDA", "NVDA beats", "insight B", "t", "informative", 0.90),
        ]],
    )
    lines = stages.insights_similarity(
        conn, 1, table="public.distilled_article_insights", label_col="first_label"
    )
    assert lines == [
        "2025-01-01 [NVDA] NVDA beats\n"
        "    - [evidance-event] insight A\n"
        "    - [informative] insight B",
    ]


def test_insights_similarity_empty_when_no_boxes():
    conn = _InsightsConn(datetime(2025, 6, 1, tzinfo=timezone.utc), [], [])
    assert stages.insights_similarity(conn, 1) == []


def test_insights_similarity_empty_when_no_published():
    conn = _InsightsConn(None, [], [])
    assert stages.insights_similarity(conn, 1) == []


def test_own_article_insights_collapses_and_drops_empty():
    # rows: (insight, topic, label) - unlabelled -> None
    conn = _StubConn([[
        ("insight one", "topicX", None),
        (None, "topic\ntwo", None),   # insight None -> topic, newline collapsed
        ("   ", None, None),          # all-whitespace -> dropped
    ]])
    assert stages.own_article_insights(conn, 1) == ["insight one", "topic two"]


def test_own_article_insights_tags_label_on_distilled():
    conn = _StubConn([[
        ("insight one", "t", "informative"),
        ("insight two", "t", "evidance-event"),
    ]])
    out = stages.own_article_insights(
        conn, 1, table="public.distilled_article_insights", label_col="second_label"
    )
    assert out == ["[informative] insight one", "[evidance-event] insight two"]


def test_gather_precedents_dispatches_on_config(monkeypatch):
    seen = {}
    monkeypatch.setattr(stages, "insights_similarity",
                        lambda c, a, **kw: seen.update(kw) or ["insight-line"])
    monkeypatch.setattr(stages, "article_similarity",
                        lambda c, a, k=5, **kw: ["article-line"])

    class _S:
        precedent_source = "insights"
        precedent_insights_threshold = 0.7
        precedent_insights_limit = 40
        precedent_lookback_days = 90
        precedent_real_news_only = False

    monkeypatch.setattr(stages, "get_settings", lambda: _S())
    assert stages.gather_precedents(None, 1) == ["insight-line"]
    assert (seen["table"], seen["label_col"], seen["filter_drop"]) == (
        "public.article_insights", None, False)
    _S.precedent_source = "distilled-first"
    assert stages.gather_precedents(None, 1) == ["insight-line"]
    assert (seen["table"], seen["label_col"], seen["filter_drop"]) == (
        "public.distilled_article_insights", "first_label", False)
    _S.precedent_source = "distilled-second"
    assert stages.gather_precedents(None, 1) == ["insight-line"]
    assert (seen["table"], seen["label_col"], seen["filter_drop"]) == (
        "public.distilled_article_insights", "second_label", True)
    _S.precedent_source = "article"
    assert stages.gather_precedents(None, 1) == ["article-line"]


# ---------------------------------------------------------------------------
# classify_stage offline tests
# ---------------------------------------------------------------------------


async def test_classify_returns_finegrained_category_and_stores_is_act(monkeypatch):
    # Second queued row feeds the UPDATE execute (its cursor result is unused).
    conn = _StubConn([(1, "T", "body", None), None])
    seen = {}
    monkeypatch.setattr(
        stages, "classify_article_finegrained",
        lambda title, content, config=None, model=None: ("news-event", "solid", True),
    )
    # capture the UPDATE params to confirm is_act is persisted
    orig_execute = conn.execute

    def spy(sql, params=None):
        if sql.strip().startswith("UPDATE"):
            seen["params"] = params
        return orig_execute(sql, params)

    conn.execute = spy
    assert stages.classify_stage(conn, "https://example.com/a") == "news-event"
    assert seen["params"] == ("news-event", "solid", True, 1)


async def test_classify_skip_paths_return_none(monkeypatch):
    monkeypatch.setattr(
        stages, "classify_article_finegrained",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError),
    )
    already = _StubConn([(1, "T", "body", "news-event")])
    assert stages.classify_stage(already, "https://example.com/a") is None
    assert already.rolled_back == 1
    empty = _StubConn([(1, "T", "   ", None)])
    assert stages.classify_stage(empty, "https://example.com/a") is None
    assert empty.rolled_back == 1


# ---------------------------------------------------------------------------
# sentiment_stage: precedents span observability tests
# ---------------------------------------------------------------------------


from contextlib import contextmanager


def _fake_stage_span(record):
    """Return a stage_span replacement that records the span name and update kwargs."""

    @contextmanager
    def stage_span(name):
        record["span_name"] = name

        class Span:
            def update(self, **kw):
                record.update(kw)

        yield Span()

    return stage_span


def _stub_sentiment_conn():
    """Minimal _StubConn for the judging path: real news + NVDA ticker.

    is_act NULL exercises the legacy COALESCE fallback (category == 'real news').
    """
    return _StubConn([
        (1, "T", "body", "real news", None, "NVDA", None),
    ])


def _patch_sentiment_deps(monkeypatch, precedents, own_insights, mode):
    """Wire gather_precedents and own_article_insights stubs; patch has_verdict/judge."""
    from ticker_news.sentiment.schemas import Verdict

    monkeypatch.setattr(stages.sentiment_store, "has_verdict", lambda c, a, t: False)
    monkeypatch.setattr(stages, "_has_insights", lambda c, a: True)
    monkeypatch.setattr(
        stages, "gather_precedents",
        lambda c, a, source=None: precedents,
    )
    monkeypatch.setattr(
        stages, "own_article_insights",
        lambda c, a, **kw: own_insights,
    )

    class _S:
        precedent_source = mode
        precedent_insights_threshold = 0.75
        precedent_insights_limit = 20
        precedent_lookback_days = 90
        precedent_real_news_only = False
        sentiment_verdict_model = "gemini-2.5-flash-lite"

    monkeypatch.setattr(stages, "get_settings", lambda: _S())
    monkeypatch.setattr(
        stages, "judge_article",
        lambda article, config=None, model=None, summary_model=None: (
            Verdict(action="buy", confidence=0.8, reasoning=""), []
        ),
    )
    monkeypatch.setattr(
        stages.sentiment_store, "save_verdict",
        lambda *a, **kw: None,
    )


def test_sentiment_span_records_explicit_precedent_source(monkeypatch):
    """An explicit precedent_source override beats the settings default."""
    record = {}
    monkeypatch.setattr(stages.obs, "stage_span", _fake_stage_span(record))
    # settings default deliberately differs from the override so a regression
    # that ignores the explicit argument cannot pass this test
    _patch_sentiment_deps(monkeypatch, ["p1", "p2"], ["oi1"], "article")
    conn = _stub_sentiment_conn()
    stages.sentiment_stage(conn, "https://example.com/a", precedent_source="distilled-second")
    assert record["span_name"] == "precedents"
    assert record["metadata"]["precedent_source"] == "distilled-second"
    assert record["output"] == {"n_precedents": 2, "n_own_insights": 1}


def test_sentiment_span_records_default_config_mode(monkeypatch):
    """When no explicit override, span metadata uses the settings default ('article')."""
    record = {}
    monkeypatch.setattr(stages.obs, "stage_span", _fake_stage_span(record))
    _patch_sentiment_deps(monkeypatch, [], [], "article")
    conn = _stub_sentiment_conn()
    stages.sentiment_stage(conn, "https://example.com/a")  # no precedent_source arg
    assert record["span_name"] == "precedents"
    assert record["metadata"]["precedent_source"] == "article"
    assert record["output"] == {"n_precedents": 0, "n_own_insights": 0}


def test_sentiment_span_records_threshold_and_limit(monkeypatch):
    """Span metadata carries threshold and limit from settings."""
    record = {}
    monkeypatch.setattr(stages.obs, "stage_span", _fake_stage_span(record))
    _patch_sentiment_deps(monkeypatch, ["p1"], [], "insights")
    conn = _stub_sentiment_conn()
    stages.sentiment_stage(conn, "https://example.com/a", precedent_source="insights")
    assert record["metadata"]["threshold"] == 0.75
    assert record["metadata"]["limit"] == 20
