# Pipeline v2 — Plan 1: Package Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the repo into an installable `src/`-layout package (`ticker_news`), unify configuration into one pydantic-settings object with a single `DATABASE_URL`, move the scraper into `ticker_news.scraping`, and ship a `ticker-news` typer CLI with a working `scrape` command — all existing tests green throughout.

**Architecture:** This is phase 1 of the spec at `docs/superpowers/specs/2026-06-10-pipeline-v2-design.md`. It creates the package skeleton every later phase builds on. The scraper package moves nearly intact (its internal imports are all relative, so only test imports and `run_scrape.py` need rewriting). The old `scraper/cli.py` stays temporarily so `run_scrape.py` keeps working mid-migration; both die in the final phase. Also fixes a latent footgun: the test `store` fixture currently TRUNCATEs the **real** `news` database by default — it moves to a `news_test` database with a guard.

**Tech Stack:** Python 3.11+, setuptools src-layout packaging, pydantic-settings 2.x, typer, psycopg 3, pytest (asyncio_mode=auto).

**Branch:** all work on `refactor/pipeline-v2` (already created). Later phases (stage migration to LangChain, service/queue, sentiment graph, Langfuse, research port + cleanup) get their own plans written against the code state this plan produces.

---

### Task 1: src-layout packaging and editable install

**Files:**
- Modify: `pyproject.toml`
- Create: `src/ticker_news/__init__.py`

- [ ] **Step 1: Capture the green baseline**

Run: `pytest -m "not db and not integration" -q`
Expected: all pass (this is the baseline every later step must preserve). If anything fails here, STOP and report — do not start the refactor on a red baseline.

- [ ] **Step 2: Replace `pyproject.toml` with packaging metadata**

The current file holds only pytest config — keep that section verbatim and add packaging. Full new content:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "ticker-news-analyzer"
version = "0.2.0"
description = "AI-compute-sector stock news pipeline: collect, scrape, embed, classify, enrich, judge sentiment."
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "trafilatura>=1.12",
    "playwright",
    "psycopg[binary]>=3.2",
    "pgvector",
    "pydantic-settings>=2.6",
    "python-dotenv",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
ticker-news = "ticker_news.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"ticker_news.scraping.store" = ["*.sql"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "db: tests that require a running Postgres (docker compose up -d)",
    "integration: tests that hit the live network",
]
```

Notes for the engineer:
- `dependencies` lists only what the *package* needs after this task (scraper + shared config + CLI). `requirements.txt` stays untouched — the legacy `scripts/` still install from it until the final phase deletes them. Do not try to reconcile the two now.
- `ticker_news.cli:app` doesn't exist until Task 4; the console script entry is declared now so one editable install covers the whole plan. Installing with a dangling entry point is fine — it only fails if *invoked*.
- The `package-data` block ships `schema.sql`, which `scraper/store/db.py` loads relative to its own file. It matters after the Task 3 move.

- [ ] **Step 3: Create the package root**

Create `src/ticker_news/__init__.py`:

```python
__version__ = "0.2.0"
```

- [ ] **Step 4: Editable-install and verify import**

Run: `pip install -e ".[dev]"`
Expected: `Successfully installed ticker-news-analyzer-0.2.0` (plus any missing deps).

Run: `python -c "import ticker_news; print(ticker_news.__version__)"`
Expected: `0.2.0`

- [ ] **Step 5: Verify baseline still green**

Run: `pytest -m "not db and not integration" -q`
Expected: same pass count as Step 1. (Tests still import the old top-level `scraper` package — untouched so far.)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ticker_news/__init__.py
git commit -m "build: src-layout packaging for ticker_news, editable install"
```

---

### Task 2: unified settings in `shared/config.py` (TDD)

One pydantic-settings object, one `DATABASE_URL`. The scraper's `SCRAPER_*` env vars keep working — they map onto fields here, and Task 3 rewires the scraper's dataclass defaults to read from this object. This ends the `SCRAPER_DB_DSN` vs `NEWS_DB_DSN` split for all *new* code (legacy `scripts/` keep their own env reads until their migration phase).

**Files:**
- Create: `tests/shared/__init__.py`
- Create: `tests/shared/test_config.py`
- Create: `src/ticker_news/shared/__init__.py`
- Create: `src/ticker_news/shared/config.py`

- [ ] **Step 1: Write the failing tests**

Create empty `tests/shared/__init__.py`, then `tests/shared/test_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/shared/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ticker_news.shared'`

- [ ] **Step 3: Implement settings**

Create empty `src/ticker_news/shared/__init__.py`, then `src/ticker_news/shared/config.py`:

```python
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Single source of configuration for the whole package.

    One database (DATABASE_URL) for every stage — the scraper's old
    SCRAPER_DB_DSN and the analysis scripts' NEWS_DB_DSN both die here.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="postgresql://scraper:scraper@localhost:5432/news",
        validation_alias="DATABASE_URL",
    )

    massive_api_key: str | None = Field(default=None, validation_alias="MASSIVE_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

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
```

Note: `_env_file=None` in the tests disables `.env` loading so a developer's local `.env` can't break the suite. `get_settings()` is cached — production code calls it; tests construct `AppSettings` directly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/shared/test_config.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tests/shared src/ticker_news/shared
git commit -m "feat: unified AppSettings with single DATABASE_URL"
```

---

### Task 3: move scraper to `ticker_news.scraping`, retarget tests at `news_test`

The scraper's internal imports are all relative (`from .config import Settings` etc.), so the package body moves unchanged. Only three things need edits: the dataclass defaults in `config.py` (now sourced from `AppSettings`), the test imports, and `run_scrape.py`. The `store` fixture also moves and gets pointed at a dedicated `news_test` database with a guard — it currently TRUNCATEs the real `news` table by default, which has destroyed production data before.

**Files:**
- Move: `scraper/` → `src/ticker_news/scraping/` (whole tree: `__init__.py`, `cli.py`, `config.py`, `csv_source.py`, `fetch.py`, `models.py`, `pipeline.py`, `robots.py`, `urls.py`, `extract/`, `store/` incl. `schema.sql`)
- Modify: `src/ticker_news/scraping/config.py`
- Move: all scraper tests `tests/test_*.py` → `tests/scraping/` (`test_cli.py`, `test_csv_source.py`, `test_extractor.py`, `test_extractor_real.py`, `test_fetch.py`, `test_models.py`, `test_pipeline.py`, `test_robots.py`, `test_store.py`, `test_urls.py`) — `tests/test_load_ticker_overview.py` stays put (it tests `scripts/enrichment`, moves in a later phase)
- Move: `tests/fixtures/` → `tests/scraping/fixtures/` (used by extractor tests)
- Move+rewrite: `tests/conftest.py` → `tests/scraping/conftest.py`
- Create: `tests/scraping/__init__.py`
- Modify: `run_scrape.py`

- [ ] **Step 1: Move the package and test trees with git mv**

```bash
git mv scraper src/ticker_news/scraping
mkdir tests/scraping
git mv tests/test_cli.py tests/test_csv_source.py tests/test_extractor.py tests/test_extractor_real.py tests/test_fetch.py tests/test_models.py tests/test_pipeline.py tests/test_robots.py tests/test_store.py tests/test_urls.py tests/scraping/
git mv tests/fixtures tests/scraping/fixtures
git mv tests/conftest.py tests/scraping/conftest.py
```

Create empty `tests/scraping/__init__.py`.

Check: if any extractor test builds fixture paths as `Path(__file__).parent / "fixtures"`, the move keeps them working (fixtures moved alongside). If a test uses `tests/fixtures` literally, fix it to `Path(__file__).parent / "fixtures"`.

- [ ] **Step 2: Rewrite test imports**

In every file under `tests/scraping/`, replace the module prefix `scraper.` → `ticker_news.scraping.`. Exact occurrences (from grep, line 1-6 of each file):

| File | Old | New |
|---|---|---|
| `tests/scraping/test_urls.py:1` | `from scraper.urls import domain_of, canonicalize_url` | `from ticker_news.scraping.urls import domain_of, canonicalize_url` |
| `tests/scraping/test_robots.py:1` | `from scraper.robots import RobotsCache` | `from ticker_news.scraping.robots import RobotsCache` |
| `tests/scraping/test_csv_source.py:2` | `from scraper.csv_source import read_jobs` | `from ticker_news.scraping.csv_source import read_jobs` |
| `tests/scraping/test_cli.py:1` | `from scraper.cli import build_settings, parse_args` | `from ticker_news.scraping.cli import build_settings, parse_args` |
| `tests/scraping/test_pipeline.py:4-6` | `from scraper.config import Settings` / `from scraper.models import ArticleJob, RawPage, Article` / `from scraper.pipeline import DomainLimiter, process_job` | same with `ticker_news.scraping.` prefix |
| `tests/scraping/test_models.py:2` | `from scraper.models import ArticleJob, RawPage, Article` | `from ticker_news.scraping.models import ...` |
| `tests/scraping/test_fetch.py:1-2` | `from scraper.fetch import http_looks_bad` / `from scraper.models import RawPage` | same with prefix |
| `tests/scraping/test_extractor_real.py:4-5` | `from scraper.models import RawPage` / `from scraper.extract.extractor import extract` | same with prefix |
| `tests/scraping/test_extractor.py:1-2` | `from scraper.models import RawPage` / `from scraper.extract.extractor import extract, register, SITE_OVERRIDES` | same with prefix |

Also check `tests/scraping/test_store.py` for `from scraper.` imports (not in the grep above because it may import only via the fixture) and rewrite if present.

- [ ] **Step 3: Rewrite the conftest — `news_test` with a prod-name guard**

Replace the entire content of `tests/scraping/conftest.py`:

```python
import os

import psycopg
import pytest

TEST_DSN = os.environ.get(
    "TICKER_NEWS_TEST_DSN", "postgresql://scraper:scraper@localhost:5432/news_test"
)
ADMIN_DSN = "postgresql://scraper:scraper@localhost:5432/news"


def _connect_test_db():
    """Connect to news_test, creating the database on first run."""
    if "news_test" not in TEST_DSN:
        raise RuntimeError(
            f"Refusing to run db tests against {TEST_DSN!r}: the test database "
            "name must contain 'news_test' (this fixture TRUNCATEs tables)."
        )
    try:
        return psycopg.connect(TEST_DSN)
    except psycopg.OperationalError as exc:
        if "news_test" not in str(exc):
            pytest.skip("Postgres not reachable; run `docker compose up -d`")
        admin = psycopg.connect(ADMIN_DSN, autocommit=True)
        admin.execute("CREATE DATABASE news_test")
        admin.close()
        return psycopg.connect(TEST_DSN)


@pytest.fixture
def store():
    from ticker_news.scraping.store.db import Store

    try:
        conn = _connect_test_db()
        conn.close()
        s = Store(TEST_DSN)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d`")
    s.init_schema()
    s.conn.execute("TRUNCATE articles")
    yield s
    s.conn.execute("TRUNCATE articles")
    s.close()
```

Why this shape: the old fixture defaulted to the real `news` DB and TRUNCATEd it (it has wiped production data before — see project memory). The guard makes pointing this fixture at a non-test database an error, not an accident. First run auto-creates `news_test`; `Store.init_schema()` then applies `schema.sql` (including `CREATE EXTENSION vector` — the docker `scraper` user is the superuser, so this works).

- [ ] **Step 4: Rewire scraper config defaults to AppSettings**

Replace the entire content of `src/ticker_news/scraping/config.py`:

```python
from dataclasses import dataclass, field

from ticker_news.shared.config import get_settings


def _s():
    return get_settings()


@dataclass(frozen=True)
class Settings:
    """Frozen per-run scraper settings.

    Defaults come from the unified AppSettings (DATABASE_URL, SCRAPER_* env
    vars); tests and the CLI override individual fields via dataclasses.replace.
    """

    db_dsn: str = field(default_factory=lambda: _s().database_url)
    concurrency: int = field(default_factory=lambda: _s().scraper_concurrency)
    per_domain: int = field(default_factory=lambda: _s().scraper_per_domain)
    domain_delay_s: float = field(default_factory=lambda: _s().scraper_domain_delay_s)
    http_timeout_s: float = field(default_factory=lambda: _s().scraper_http_timeout_s)
    min_words: int = field(default_factory=lambda: _s().scraper_min_words)
    respect_robots: bool = field(default_factory=lambda: _s().scraper_respect_robots)
    user_agent: str = field(default_factory=lambda: _s().scraper_user_agent)
```

The public interface (`Settings()`, frozen, `dataclasses.replace`) is unchanged — `pipeline.py`, `cli.py`, and `tests/scraping/test_pipeline.py` keep working untouched. What changes is where defaults come from: `DATABASE_URL` instead of `SCRAPER_DB_DSN`. **Behavioral note:** the old TCP default DSN is preserved via `AppSettings.database_url`'s default, so a no-env run behaves identically.

- [ ] **Step 5: Update run_scrape.py**

Replace content of `run_scrape.py`:

```python
from ticker_news.scraping.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the offline suite**

Run: `pytest -m "not db and not integration" -q`
Expected: same pass count as the Task 1 baseline. Typical failure mode: a missed `scraper.` import — the error names the file; fix the prefix and rerun.

- [ ] **Step 7: Run the db-marked tests against news_test (if docker is up)**

Run: `docker compose up -d` then `pytest -m db -q`
Expected: pass (first run creates `news_test` and applies the schema). If docker can't run on this machine, note it and move on — the offline suite is the gate.

Sanity check after: `python -c "import psycopg; print(psycopg.connect('postgresql://scraper:scraper@localhost:5432/news').execute('SELECT count(*) FROM articles').fetchone())"` — the real `news` count must be whatever it was before (db tests no longer touch it).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: move scraper into ticker_news.scraping; db tests isolated to news_test"
```

---

### Task 4: `ticker-news` typer CLI with the scrape command (TDD)

The typer app becomes the package's single entry point; this task ships `scrape` (mirroring the old argparse CLI exactly). Later phases add `serve`, `backfill`, `search`, etc. The old `ticker_news/scraping/cli.py` stays for `run_scrape.py` until the cleanup phase.

**Files:**
- Create: `tests/test_root_cli.py`
- Create: `src/ticker_news/cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_root_cli.py`:

```python
from typer.testing import CliRunner

import ticker_news.cli as cli

runner = CliRunner()


def test_help_lists_scrape_command():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "scrape" in result.output


def test_scrape_passes_args_and_settings(monkeypatch):
    captured = {}

    async def fake_run(csv, settings, *, limit=None, retry_errors=False):
        captured.update(csv=csv, settings=settings, limit=limit, retry_errors=retry_errors)

    monkeypatch.setattr(cli, "pipeline_run", fake_run)
    result = runner.invoke(
        cli.app,
        ["scrape", "--csv", "x.csv", "--ignore-robots", "--concurrency", "4",
         "--limit", "10", "--retry-errors"],
    )
    assert result.exit_code == 0, result.output
    assert captured["csv"] == "x.csv"
    assert captured["limit"] == 10
    assert captured["retry_errors"] is True
    assert captured["settings"].respect_robots is False
    assert captured["settings"].concurrency == 4


def test_scrape_defaults_respect_robots(monkeypatch):
    captured = {}

    async def fake_run(csv, settings, *, limit=None, retry_errors=False):
        captured["settings"] = settings

    monkeypatch.setattr(cli, "pipeline_run", fake_run)
    result = runner.invoke(cli.app, ["scrape", "--csv", "x.csv"])
    assert result.exit_code == 0, result.output
    assert captured["settings"].respect_robots is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_root_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ticker_news.cli'`

- [ ] **Step 3: Implement the CLI**

Create `src/ticker_news/cli.py`:

```python
import asyncio
from dataclasses import replace

import typer

from ticker_news.scraping.config import Settings
from ticker_news.scraping.pipeline import run as pipeline_run

app = typer.Typer(help="Ticker News Analyzer pipeline.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Ticker News Analyzer — collect, scrape, embed, classify, and judge stock news."""
    # Forces typer to treat this as a multi-command app even while `scrape`
    # is the only command, so the interface stays `ticker-news scrape ...`.


@app.command()
def scrape(
    csv: str = typer.Option(..., help="Path to the articles CSV."),
    limit: int | None = typer.Option(None, help="Process at most N rows."),
    retry_errors: bool = typer.Option(False, "--retry-errors", help="Re-process URLs already stored with an error."),
    ignore_robots: bool = typer.Option(False, "--ignore-robots", help="Skip robots.txt checks."),
    concurrency: int | None = typer.Option(None, help="Worker count override."),
) -> None:
    """Scrape article bodies for every URL in the CSV into the articles table."""
    settings = Settings()
    if ignore_robots:
        settings = replace(settings, respect_robots=False)
    if concurrency:
        settings = replace(settings, concurrency=concurrency)
    asyncio.run(pipeline_run(csv, settings, limit=limit, retry_errors=retry_errors))
```

Note: `monkeypatch.setattr(cli, "pipeline_run", ...)` in the tests works because `scrape` references the module-global `pipeline_run` at call time.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_root_cli.py -q`
Expected: 3 passed

- [ ] **Step 5: Verify the console script end-to-end**

Run: `ticker-news --help`
Expected: usage text listing the `scrape` command. (If `command not found`, re-run `pip install -e ".[dev]"` — the entry point was declared in Task 1.)

- [ ] **Step 6: Commit**

```bash
git add tests/test_root_cli.py src/ticker_news/cli.py
git commit -m "feat: ticker-news typer CLI with scrape command"
```

---

### Task 5: full verification sweep

**Files:** none modified — verification only.

- [ ] **Step 1: Full offline suite**

Run: `pytest -m "not db and not integration" -q`
Expected: everything green — old count + 4 (config) + 3 (CLI).

- [ ] **Step 2: db suite (if docker available)**

Run: `pytest -m db -q`
Expected: green against `news_test`.

- [ ] **Step 3: Smoke the real entry points**

Run: `ticker-news --help` and `python run_scrape.py --help`
Expected: both print usage (proving the new CLI and the legacy path both work mid-migration).

- [ ] **Step 4: Push the branch**

```bash
git push -u origin refactor/pipeline-v2
```

(The PR to `main` is opened at the end of the full migration, or earlier as a stacked/draft PR if the user wants incremental review — ask before opening one.)

---

## Out of scope for Plan 1 (later plans)

- Plan 2: embedding / classification / enrichment stages move into the package as LangChain structured-output chains (`langchain` 1.3.x pins land there — note `requirements.txt`'s old `<1` pin comment is vestigial: nothing imports langchain today, grep-verified).
- Plan 3: `pipeline_jobs` table, worker service, `NewsFeedSource` protocol + Massive poller + CSV backfill.
- Plan 4: sentiment LangGraph (Send fan-out analysts).
- Plan 5: Langfuse compose services + tracing wiring.
- Plan 6: `research/` port, deletion of `scripts/`, `run_scrape.py`, `scraping/cli.py`, CLAUDE.md update, final PR.
