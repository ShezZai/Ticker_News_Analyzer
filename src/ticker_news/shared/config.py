from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Single source of configuration for the whole package.

    One database (DATABASE_URL) for every stage — the scraper's old
    SCRAPER_DB_DSN and the analysis scripts' NEWS_DB_DSN both die here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="postgresql://scraper:scraper@localhost:5432/news",
        # SCRAPER_DB_DSN is honored as a fallback for the migration window
        # (legacy scraper convention); remove the alias in the final cleanup phase.
        validation_alias=AliasChoices("DATABASE_URL", "SCRAPER_DB_DSN"),
    )

    massive_api_key: str | None = Field(default=None, validation_alias="MASSIVE_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    # Scraper knobs — legacy SCRAPER_* env names preserved on purpose.
    scraper_concurrency: int = Field(default=8, validation_alias="SCRAPER_CONCURRENCY")
    scraper_per_domain: int = Field(default=2, validation_alias="SCRAPER_PER_DOMAIN")
    scraper_domain_delay_s: float = Field(default=1.0, validation_alias="SCRAPER_DOMAIN_DELAY")
    scraper_http_timeout_s: float = Field(default=20.0, validation_alias="SCRAPER_HTTP_TIMEOUT")
    scraper_min_words: int = Field(default=120, validation_alias="SCRAPER_MIN_WORDS")
    scraper_respect_robots: bool = Field(default=True, validation_alias="SCRAPER_RESPECT_ROBOTS")
    scraper_user_agent: str = Field(
        default="Mozilla/5.0 (compatible; AITickerNewsBot/0.1; research project)",
        validation_alias="SCRAPER_UA",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
