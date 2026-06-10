import psycopg
import pytest

from ticker_news.ingestion.feed import FeedItem
from ticker_news.service import jobs

pytestmark = pytest.mark.db

# Reuse the guarded news_test connection logic from the scraping tests.
from tests.scraping.conftest import TEST_DSN, _connect_test_db


@pytest.fixture
def conn():
    try:
        c = _connect_test_db()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d`")
    jobs.ensure_schema(c)
    c.execute("TRUNCATE pipeline_jobs")
    c.commit()
    yield c
    c.execute("TRUNCATE pipeline_jobs")
    c.commit()
    c.close()


def _item(url="https://example.com/a"):
    return FeedItem(url=url, tickers=["NVDA"], publisher="Benzinga")


def test_enqueue_is_idempotent(conn):
    assert jobs.enqueue(conn, _item()) is True
    assert jobs.enqueue(conn, _item()) is False
    assert jobs.counts(conn) == {"pending": 1}


def test_claim_marks_running_and_returns_payload(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    assert job.article_url == "https://example.com/a"
    assert job.stage == "scrape"
    assert job.tickers == ["NVDA"]
    assert jobs.claim(conn) is None  # nothing else pending


def test_advance_to_done(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    jobs.advance(conn, job.article_url, jobs.DONE)
    assert jobs.counts(conn) == {"done": 1}
    assert jobs.queue_drained(conn) is True


def test_fail_backs_off_then_parks_as_failed(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    jobs.fail(conn, job.article_url, "boom")
    assert jobs.counts(conn) == {"pending": 1}
    # backed-off job is not claimable right now
    assert jobs.claim(conn) is None
    for _ in range(jobs.MAX_ATTEMPTS - 1):
        jobs.fail(conn, job.article_url, "boom again")
    assert jobs.counts(conn) == {"failed": 1}


def test_requeue_failed_resets(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    for _ in range(jobs.MAX_ATTEMPTS):
        jobs.fail(conn, job.article_url, "boom")
    assert jobs.requeue_failed(conn) == 1
    assert jobs.claim(conn) is not None


def test_recover_orphans(conn):
    jobs.enqueue(conn, _item())
    jobs.claim(conn)
    assert jobs.recover_orphans(conn) == 1
    assert jobs.claim(conn) is not None


def test_fail_permanent_parks_immediately(conn):
    jobs.enqueue(conn, _item())
    job = jobs.claim(conn)
    jobs.fail(conn, job.article_url, "robots", permanent=True)
    assert jobs.counts(conn) == {"failed": 1}
