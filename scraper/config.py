import json
import os
from dataclasses import dataclass, field


def _domain_overrides() -> dict:
    """Per-domain pacing overrides: {domain: {per_domain, delay, max_delay}}.

    Sourced from the SCRAPER_DOMAIN_OVERRIDES env var (JSON). The built-in
    default slows fool.com, which rate-limits (HTTP 429) aggressively under load.
    """
    raw = os.environ.get("SCRAPER_DOMAIN_OVERRIDES")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {"fool.com": {"per_domain": 1, "delay": 3.0, "max_delay": 60.0}}


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
    # Adaptive throttle / backoff handling.
    http_max_retries: int = int(os.environ.get("SCRAPER_HTTP_MAX_RETRIES", "5"))
    http_backoff_base: float = float(os.environ.get("SCRAPER_HTTP_BACKOFF_BASE", "1.0"))
    http_backoff_max: float = float(os.environ.get("SCRAPER_HTTP_BACKOFF_MAX", "30.0"))
    http_backoff_jitter: float = float(os.environ.get("SCRAPER_HTTP_BACKOFF_JITTER", "0.3"))
    backoff_factor: float = float(os.environ.get("SCRAPER_BACKOFF_FACTOR", "2.0"))
    domain_overrides: dict = field(default_factory=_domain_overrides)
