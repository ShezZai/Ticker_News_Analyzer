from ticker_news.shared.config import AppSettings


def test_default_database_url_matches_docker_compose(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
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
    s = AppSettings(_env_file=None)
    assert s.scraper_concurrency == 3
    assert s.scraper_respect_robots is False
    assert s.scraper_domain_delay_s == 2.5


def test_api_keys_default_to_none(monkeypatch):
    for var in ("MASSIVE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = AppSettings(_env_file=None)
    assert s.massive_api_key is None
    assert s.openai_api_key is None
    assert s.google_api_key is None
