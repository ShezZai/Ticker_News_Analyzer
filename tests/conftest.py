import os

import psycopg
import pytest

# DB tests run against a DEDICATED test database, NEVER the populated `news` DB.
# The `store` fixture TRUNCATEs `articles`; pointing that at the real database
# destroys data. Default to a separate `news_test` DB and, as a belt-and-braces
# safety net, refuse to truncate anything whose name doesn't end in `_test`.
DSN = os.environ.get(
    "SCRAPER_TEST_DB_DSN", "postgresql://scraper:scraper@localhost:5432/news_test"
)


def _current_db(conn) -> str:
    return conn.execute("SELECT current_database()").fetchone()[0]


@pytest.fixture
def store():
    from scraper.store.db import Store
    try:
        s = Store(DSN)
    except psycopg.OperationalError:
        pytest.skip(
            "Test Postgres not reachable. Run `docker compose up -d` and create the "
            "test DB:  docker exec news_pg createdb -U scraper news_test"
        )
    if not _current_db(s.conn).endswith("_test"):
        s.close()
        pytest.skip(
            "Refusing to run db tests against a non-test database "
            "(name must end with '_test'); set SCRAPER_TEST_DB_DSN."
        )
    s.init_schema()
    s.conn.execute("TRUNCATE articles")
    yield s
    s.conn.execute("TRUNCATE articles")
    s.close()
