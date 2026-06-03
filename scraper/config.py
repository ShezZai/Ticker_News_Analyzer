import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_dsn: str = os.environ.get(
        "SCRAPER_DB_DSN", "postgresql://scraper:scraper@localhost:5432/news"
    )
    concurrency: int = int(os.environ.get("SCRAPER_CONCURRENCY", "8"))
    per_domain: int = int(os.environ.get("SCRAPER_PER_DOMAIN", "2"))
    domain_delay_s: float = float(os.environ.get("SCRAPER_DOMAIN_DELAY", "1.0"))
    http_timeout_s: float = float(os.environ.get("SCRAPER_HTTP_TIMEOUT", "20"))
    min_words: int = int(os.environ.get("SCRAPER_MIN_WORDS", "120"))
    respect_robots: bool = os.environ.get("SCRAPER_RESPECT_ROBOTS", "1") != "0"
    user_agent: str = os.environ.get(
        "SCRAPER_UA",
        "Mozilla/5.0 (compatible; AITickerNewsBot/0.1; research project)",
    )
