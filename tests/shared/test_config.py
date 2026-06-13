from ticker_news.shared.config import AppSettings, get_settings


def test_default_database_url_matches_docker_compose(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SCRAPER_DB_DSN", raising=False)
    s = AppSettings(_env_file=None)
    assert s.database_url == "postgresql://scraper:scraper@localhost:5432/news"


def test_database_url_env_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@dbhost:5432/other")
    s = AppSettings(_env_file=None)
    assert s.database_url == "postgresql://u:p@dbhost:5432/other"


def test_scraper_knobs_read_legacy_env_names(monkeypatch):
    monkeypatch.setenv("SCRAPER_CONCURRENCY", "3")
    monkeypatch.setenv("SCRAPER_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("SCRAPER_DOMAIN_DELAY", "2.5")
    monkeypatch.setenv("SCRAPER_UA", "TestBot/1.0")
    s = AppSettings(_env_file=None)
    assert s.scraper_concurrency == 3
    assert s.scraper_respect_robots is False
    assert s.scraper_domain_delay_s == 2.5
    assert s.scraper_user_agent == "TestBot/1.0"


def test_api_keys_default_to_none(monkeypatch):
    for var in ("MASSIVE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = AppSettings(_env_file=None)
    assert s.massive_api_key is None
    assert s.openai_api_key is None
    assert s.google_api_key is None


def test_scraper_db_dsn_fallback_during_migration(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SCRAPER_DB_DSN", "postgresql://legacy:x@oldhost:5432/news")
    s = AppSettings(_env_file=None)
    assert s.database_url == "postgresql://legacy:x@oldhost:5432/news"


def test_database_url_wins_over_scraper_db_dsn(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://new:x@newhost:5432/news")
    monkeypatch.setenv("SCRAPER_DB_DSN", "postgresql://legacy:x@oldhost:5432/news")
    s = AppSettings(_env_file=None)
    assert s.database_url == "postgresql://new:x@newhost:5432/news"


def test_langfuse_disabled_by_default(monkeypatch):
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = AppSettings(_env_file=None)
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host == "https://cloud.langfuse.com"
