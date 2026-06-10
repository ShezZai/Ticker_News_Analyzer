import psycopg
import pytest

from ticker_news.sentiment import store
from ticker_news.sentiment.schemas import Verdict

pytestmark = pytest.mark.db

from tests.scraping.conftest import _connect_test_db


@pytest.fixture
def conn():
    try:
        c = _connect_test_db()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    # articles table must exist for the FK
    from ticker_news.scraping.store.db import Store
    from tests.scraping.conftest import TEST_DSN

    s = Store(TEST_DSN)
    s.init_schema()
    s.close()
    store.ensure_schema(c)
    c.execute("TRUNCATE article_sentiment, articles RESTART IDENTITY CASCADE")
    c.commit()
    yield c
    c.execute("TRUNCATE article_sentiment, articles RESTART IDENTITY CASCADE")
    c.commit()
    c.close()


def _seed_article(conn) -> int:
    row = conn.execute(
        "INSERT INTO articles (url, source_domain, status) "
        "VALUES ('https://example.com/s', 'example.com', 'ok') RETURNING id"
    ).fetchone()
    conn.commit()
    return row[0]


def test_save_and_has_verdict_roundtrip(conn):
    aid = _seed_article(conn)
    v = Verdict(action="buy", confidence=0.7, reasoning="r")
    assert store.has_verdict(conn, aid, "NVDA") is False
    store.save_verdict(conn, aid, "NVDA", v, [{"role": "x", "analysis": "y"}], "m")
    assert store.has_verdict(conn, aid, "NVDA") is True
    # idempotent: second save is a no-op, not an error
    store.save_verdict(conn, aid, "NVDA", v, [], "m")
