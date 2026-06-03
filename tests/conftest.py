import os
import pytest
import psycopg

DSN = os.environ.get("SCRAPER_DB_DSN", "postgresql://scraper:scraper@localhost:5432/news")


@pytest.fixture
def store():
    from scraper.store.db import Store
    try:
        s = Store(DSN)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d`")
    s.init_schema()
    s.conn.execute("TRUNCATE articles")
    yield s
    s.conn.execute("TRUNCATE articles")
    s.close()
