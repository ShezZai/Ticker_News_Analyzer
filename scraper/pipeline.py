import asyncio
import contextlib
import random
import time

from .config import Settings
from .csv_source import read_jobs
from .extract.extractor import extract
from .fetch import Fetcher, http_looks_bad, is_throttled
from .models import ArticleJob, RawPage
from .robots import RobotsCache
from .store.db import Store
from .urls import canonicalize_url, domain_of


class DomainLimiter:
    """Caps concurrency per domain, enforces a min delay between requests, and
    adapts to rate limiting.

    On a 429/503 a domain is *penalized*: its effective delay grows
    exponentially (up to a cap) and a cooldown pauses every worker on that
    domain, not just the URL that hit the limit. A clean response *rewards*
    the domain, decaying the delay back toward its baseline. Per-domain
    ``overrides`` set a slower baseline (concurrency / delay / max_delay) for
    sites known to throttle.
    """

    def __init__(self, per_domain: int, delay: float, *,
                 overrides: dict | None = None, max_delay: float = 30.0,
                 backoff_factor: float = 2.0, jitter: float = 0.3):
        self.per_domain = per_domain
        self.delay = delay
        self.overrides = overrides or {}
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._last: dict[str, float] = {}
        self._current_delay: dict[str, float] = {}   # domain -> live effective delay
        self._paused_until: dict[str, float] = {}     # domain -> monotonic deadline

    def _base_delay(self, domain: str) -> float:
        return self.overrides.get(domain, {}).get("delay", self.delay)

    def _max_delay(self, domain: str) -> float:
        return self.overrides.get(domain, {}).get("max_delay", self.max_delay)

    def _per_domain(self, domain: str) -> int:
        return self.overrides.get(domain, {}).get("per_domain", self.per_domain)

    def current_delay(self, domain: str) -> float:
        return self._current_delay.get(domain, self._base_delay(domain))

    def penalize(self, domain: str, retry_after: float | None = None) -> bool:
        """Record a throttle hit. Returns True if the domain newly entered/extended
        cooldown (useful for one-line logging). Grows the delay and pauses the domain."""
        cur = self.current_delay(domain)
        if retry_after is not None:
            new = min(max(cur, retry_after), self._max_delay(domain))
            cooldown = retry_after
        else:
            grown = max(cur, self._base_delay(domain)) * self.backoff_factor
            new = min(grown, self._max_delay(domain))
            if new <= 0:                      # baseline delay was 0
                new = 1.0
            cooldown = new
        self._current_delay[domain] = new
        pause = cooldown * (1 + random.uniform(0, self.jitter))
        self._paused_until[domain] = time.monotonic() + pause
        return True

    def reward(self, domain: str) -> None:
        """Record a clean response; decay the delay toward the baseline."""
        cur = self._current_delay.get(domain)
        if cur is None:
            return
        base = self._base_delay(domain)
        new = max(base, cur / self.backoff_factor)
        if new <= base:
            self._current_delay.pop(domain, None)
        else:
            self._current_delay[domain] = new

    def _sem(self, domain: str) -> asyncio.Semaphore:
        if domain not in self._sems:
            self._sems[domain] = asyncio.Semaphore(self._per_domain(domain))
        return self._sems[domain]

    @contextlib.asynccontextmanager
    async def slot(self, domain: str):
        sem = self._sem(domain)
        await sem.acquire()
        try:
            # Honor a domain-wide cooldown set by penalize().
            pause = self._paused_until.get(domain, 0.0) - time.monotonic()
            if pause > 0:
                await asyncio.sleep(pause)
            delay = self.current_delay(domain)
            if delay > 0:
                wait = delay - (time.monotonic() - self._last.get(domain, 0.0))
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last[domain] = time.monotonic()
            yield
        finally:
            sem.release()


async def _http_with_retries(fetcher, url: str, settings: Settings,
                             limiter: "DomainLimiter | None" = None,
                             domain: str | None = None) -> RawPage | None:
    """HTTP GET with exponential backoff + jitter. Engages the domain circuit
    breaker on a throttle (429/503) and rewards it on a clean response."""
    raw = None
    delay = settings.http_backoff_base
    for attempt in range(settings.http_max_retries):
        raw = await fetcher.http_get(url)
        if raw is not None and raw.status != 429 and raw.status < 500:
            if limiter is not None and domain is not None:
                limiter.reward(domain)
            return raw
        if limiter is not None and domain is not None and is_throttled(raw):
            limiter.penalize(domain, raw.retry_after)
        if attempt < settings.http_max_retries - 1:  # don't sleep after the last try
            base = raw.retry_after if (raw and raw.retry_after) else delay
            await asyncio.sleep(base * (1 + random.uniform(0, settings.http_backoff_jitter)))
            delay = min(delay * 2, settings.http_backoff_max)
    return raw


async def process_job(job: ArticleJob, fetcher, store, settings: Settings,
                      limiter: DomainLimiter, robots) -> str:
    domain = domain_of(job.url)

    if settings.respect_robots and robots is not None:
        if not await asyncio.to_thread(robots.allowed, job.url):
            await asyncio.to_thread(
                store.save, url=job.url, url_canonical=canonicalize_url(job.url),
                source_domain=domain, publisher=job.publisher, tickers=job.tickers,
                published_utc=job.published_utc, title=None, author=None, content=None,
                raw_html=None, lang=None, word_count=0, fetch_method=None, http_status=None,
                status="error", error="blocked_by_robots",
            )
            return "error"

    async with limiter.slot(domain):
        raw = await _http_with_retries(fetcher, job.url, settings, limiter, domain)

    used_browser = False
    art = None if (raw is None or http_looks_bad(raw)) else extract(raw, settings.min_words)

    # Don't escalate a pure throttle to the browser: it shares the same
    # rate-limited IP, so it can't help and only burns time.
    needs_more = raw is None or http_looks_bad(raw) or art is None or art.is_weak(settings.min_words)
    if needs_more and not is_throttled(raw):
        async with limiter.slot(domain):
            raw2 = await fetcher.browser_get(job.url)
        # Only adopt the browser result if it's actually usable; otherwise keep
        # the original HTTP response so we don't store a worse page (e.g. a
        # playwright 403 challenge clobbering a real 200 body).
        if raw2 is not None and not http_looks_bad(raw2):
            used_browser = True
            raw = raw2
            art = extract(raw2, settings.min_words)

    if raw is None:
        status, error, art = "error", "fetch_failed", None
    elif is_throttled(raw) and (art is None or not art.text.strip()):
        # A throttle that never cleared: retryable, not a dead-end "empty".
        status, error, art = "error", "rate_limited", None
    elif art is None or not art.text.strip():
        status, error = "empty", None
    else:
        status, error = "ok", None

    fetch_method = ("playwright" if used_browser else (raw.method if raw else None))

    await asyncio.to_thread(
        store.save, url=job.url, url_canonical=canonicalize_url(job.url),
        source_domain=domain, publisher=job.publisher, tickers=job.tickers,
        published_utc=job.published_utc,
        title=(art.title if art else None), author=(art.author if art else None),
        content=(art.text if art else None), raw_html=(raw.html if raw else None),
        lang=(art.lang if art else None), word_count=(art.word_count if art else 0),
        fetch_method=fetch_method, http_status=(raw.status if raw else None),
        status=status, error=error,
    )
    return status


async def run(csv_path: str | None, settings: Settings, limit: int | None = None,
              retry_errors: bool = False, retry_failed: bool = False) -> dict[str, int]:
    store = Store(settings.db_dsn)
    store.init_schema()
    fetcher = Fetcher(settings)
    robots = RobotsCache(settings.user_agent) if settings.respect_robots else None
    limiter = DomainLimiter(
        settings.per_domain, settings.domain_delay_s,
        overrides=settings.domain_overrides, max_delay=settings.http_backoff_max,
        backoff_factor=settings.backoff_factor,
    )
    # Source jobs from the DB's rate_limited backlog, or from the CSV.
    jobs = store.iter_failed_jobs() if retry_failed else read_jobs(csv_path)

    counts = {"ok": 0, "empty": 0, "error": 0, "skip": 0}
    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.concurrency * 4)

    async def worker():
        while True:
            job = await queue.get()
            try:
                if job is None:
                    return
                if not retry_errors and await asyncio.to_thread(store.exists_ok, job.url):
                    counts["skip"] += 1
                else:
                    res = await process_job(job, fetcher, store, settings, limiter, robots)
                    counts[res] = counts.get(res, 0) + 1
                done = sum(counts.values())
                if done % 50 == 0:
                    print(f"[progress] {done} done :: {counts}")
            except Exception as exc:  # never let one bad URL kill a worker
                counts["error"] = counts.get("error", 0) + 1
                print(f"[worker error] {getattr(job, 'url', '?')}: {exc!r}")
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(settings.concurrency)]

    try:
        produced = 0
        for job in jobs:
            if limit is not None and produced >= limit:
                break
            produced += 1
            await queue.put(job)
        for _ in workers:
            await queue.put(None)

        await queue.join()
        await asyncio.gather(*workers)
    finally:
        # Always release the browser subprocess, httpx client, and DB connection,
        # even on Ctrl+C or an unexpected error mid-run.
        for w in workers:
            w.cancel()
        await fetcher.aclose()
        store.close()
    print(f"[done] {counts}")
    return counts
