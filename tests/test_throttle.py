"""Tests for adaptive 429/throttle handling (circuit breaker + labeling)."""
import asyncio

from scraper.config import Settings
from scraper.fetch import _parse_retry_after, is_throttled
from scraper.models import ArticleJob, RawPage
from scraper.pipeline import DomainLimiter, process_job


# --- is_throttled / Retry-After parsing -----------------------------------

def _raw(status):
    return RawPage(url="u", final_url="u", status=status, html="", method="http")


def test_is_throttled_only_for_429_503():
    assert is_throttled(_raw(429)) is True
    assert is_throttled(_raw(503)) is True
    assert is_throttled(_raw(403)) is False
    assert is_throttled(_raw(200)) is False
    assert is_throttled(None) is False


def test_parse_retry_after():
    assert _parse_retry_after("30") == 30.0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("Wed, 21 Oct 2025 07:28:00 GMT") is None  # date form ignored


# --- adaptive DomainLimiter (the circuit breaker) -------------------------

def test_penalize_grows_and_reward_decays():
    lim = DomainLimiter(per_domain=2, delay=1.0, backoff_factor=2.0, max_delay=10.0)
    assert lim.current_delay("fool.com") == 1.0
    lim.penalize("fool.com")
    assert lim.current_delay("fool.com") == 2.0
    lim.penalize("fool.com")
    assert lim.current_delay("fool.com") == 4.0
    lim.reward("fool.com")
    assert lim.current_delay("fool.com") == 2.0


def test_penalize_caps_at_max_delay():
    lim = DomainLimiter(per_domain=2, delay=4.0, backoff_factor=2.0, max_delay=6.0)
    lim.penalize("fool.com")
    assert lim.current_delay("fool.com") == 6.0


def test_penalize_honors_retry_after():
    lim = DomainLimiter(per_domain=2, delay=1.0, max_delay=100.0)
    lim.penalize("fool.com", retry_after=30.0)
    assert lim.current_delay("fool.com") == 30.0


def test_penalize_uses_per_domain_override_base():
    lim = DomainLimiter(
        per_domain=2, delay=1.0, backoff_factor=2.0,
        overrides={"fool.com": {"delay": 3.0, "max_delay": 60.0}},
    )
    assert lim.current_delay("fool.com") == 3.0
    lim.penalize("fool.com")
    assert lim.current_delay("fool.com") == 6.0
    assert lim.current_delay("investing.com") == 1.0  # global base elsewhere


# --- process_job: throttle is labeled retryable and never escalates -------

class FakeFetcher:
    def __init__(self, http=None, browser=None):
        self._http, self._browser = http, browser
        self.browser_calls = 0

    async def http_get(self, url):
        return self._http

    async def browser_get(self, url):
        self.browser_calls += 1
        return self._browser


class FakeStore:
    def __init__(self):
        self.saved = []

    def exists_ok(self, url):
        return False

    def save(self, **kw):
        self.saved.append(kw)


def _fast(**kw):
    # http_backoff_base=0 keeps retry sleeps ~0 so the suite stays fast.
    return Settings(respect_robots=False, http_max_retries=2, http_backoff_base=0.0, **kw)


def test_process_job_rate_limited_skips_browser_and_labels_error(monkeypatch):
    monkeypatch.setattr("scraper.pipeline.extract", lambda raw, min_words: None)
    job = ArticleJob(url="https://fool.com/a", tickers=[], published_utc=None, publisher="Fool")
    throttled = RawPage(url="u", final_url="https://fool.com/a", status=429,
                        html="x" * 2000, method="http")
    fetcher = FakeFetcher(http=throttled, browser=None)
    store = FakeStore()
    limiter = DomainLimiter(per_domain=2, delay=0.0, backoff_factor=2.0, max_delay=30.0)

    res = asyncio.run(process_job(job, fetcher, store, _fast(), limiter, robots=None))

    assert res == "error"
    assert store.saved[0]["status"] == "error"
    assert store.saved[0]["error"] == "rate_limited"
    assert store.saved[0]["http_status"] == 429
    assert fetcher.browser_calls == 0               # never escalate on a throttle
    assert limiter.current_delay("fool.com") > 0    # breaker engaged
