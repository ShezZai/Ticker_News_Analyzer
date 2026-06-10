# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline for collecting AI-compute-sector stock news, scraping full article text, embedding it, and running semantic search / event analysis against price action. Everything centers on a single Postgres (pgvector) table: **`public.articles`**. Each stage reads from and/or writes to that table — understanding the table is the fastest way to understand the system.

## Architecture: the pipeline stages

Data flows left to right; every stage past step 2 talks to the same `articles` table and is independently re-runnable / resumable.

1. **Collect news metadata** — `scripts/data_getting_parsing/ticker_news.py` (`fetch_news_csv`) pulls news + per-ticker sentiment from the **Massive.com** REST API into a CSV (`ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name`). `run_universe.py` drives it over the ~118-ticker AI-compute universe defined in `ai_compute_us_market_universe_consolidated_segments_min5.csv`.
2. **Scrape article bodies** — `scraper/` package (entry: `run_scrape.py`) reads that CSV and upserts full articles into `articles`. See "The scraper" below.
3. **Embed** — `scripts/embedding/embed_articles.py` adds an `embedding vector(1536)` column (OpenAI `text-embedding-3-small`; vectors come back unit-normalized so cosine == inner product), then builds an HNSW cosine index. Incremental: only rows with `embedding IS NULL` unless `--reembed`.
4. **Classify** — `scripts/classify/classify_articles.py` assigns each article a content `category` via a two-pass Gemini pipeline (gemini-2.5-flash-lite first; "real news" verdicts re-confirmed with gemini-2.5-flash). `reclassify_real_news.py` is a one-off re-run over existing "real news" rows.
5. **Enrich/tag** — `scripts/enrichment/tag_segments.py` derives `primary_ticker`, `primary_segment`, `more_tickers[]`, `more_segments[]` from each row's `tickers[]`. `load_ticker_data.py` loads the ticker→segment/company lookup into `public.ticker_data`; `load_ticker_overview.py` adds Yahoo Finance company descriptions to `public.ticker_overview`. `extract_insights.py` chunks each article into "insight boxes" with Gemini and embeds them into `public.article_insights`.
6. **Search/analyze** — `scripts/search/search_articles.py` embeds a query with the *same* model and does ANN search over `articles.embedding`, with filters for tickers, segment, domain, and date. `search_articles_by_insights.py` does the same at *insight* granularity over `public.article_insights`. `scripts/checks_backtesting/ticker_candles.py` renders intraday OHLC charts (Massive API) to relate articles to price moves.
7. **Sentiment & backtesting** — `scripts/search/insight_sentiment.py` (buy/sell/hold for a ticker at an article's publication moment, Gemini-judged), `followup_sentiment.py` (second-pass refinement of a `--top-2` run), `backtest_top2.py` (backtests top-2 sentiment on real-news articles against prices). The `scripts/ticker_scan/` suite scans for big intraday-range days (`scan_ranges.py`), attaches that day's articles (`attach_articles.py`), computes buy-the-news returns per catalyst article (`catalyst_returns.py`), and renders charts (`render_all_tickers.py`, `render_bombs.py`, `render_catalyst_bombs.py`).

## Critical gotchas

- **Two different DB connection conventions for the same database.** The scraper reads `SCRAPER_DB_DSN` (default `postgresql://scraper:scraper@localhost:5432/news` — TCP, matches `docker-compose.yml`). Every analysis script under `scripts/` reads `NEWS_DB_DSN` / `DATABASE_URL` (default `dbname=news` — local-socket peer auth). Both must resolve to the **same `news` database**. When running the scraper against the docker DB but analysis scripts against a local socket, set the env vars explicitly or they will silently hit different databases.
- **Both search tools import `embed_articles.py`** (`search_articles.py` and `search_articles_by_insights.py` do `from embed_articles import embed_query, get_conn, ...`) but live in *different* `scripts/` subfolders. Running either requires the embedding folder on `PYTHONPATH` (e.g. `PYTHONPATH=scripts/embedding python scripts/search/search_articles.py ...`). Query and stored vectors are produced by the same code path on purpose — don't fork the model config.
- **pgvector must be enabled once as a superuser**: `CREATE EXTENSION vector;` in the `news` DB. The scraper's `schema.sql` does this and creates `articles`; `embed_articles.py` only *checks* for the extension and fails loudly if missing.
- **The universe CSV is not committed** (`*.csv` is gitignored). `run_universe.py` and `load_ticker_data.py` expect `ai_compute_us_market_universe_consolidated_segments_min5.csv` to be present locally.
- **Secrets via `.env`** (loaded with `python-dotenv`): `MASSIVE_API_KEY` for all Massive.com calls (news, candles, ticker_scan prices); `OPENAI_API_KEY` for embeddings (`embed_articles.py`, `extract_insights.py`); `GOOGLE_API_KEY` for all Gemini calls (classify, insights, sentiment); DB DSN env vars as above.

## The scraper (`scraper/`)

Async, CSV-driven, resumable. `pipeline.run()` fans CSV rows out to a worker pool (`SCRAPER_CONCURRENCY`) behind a `DomainLimiter` (per-domain concurrency + min delay). Per URL it is **HTTP-first**: `Fetcher.http_get` (httpx); only if the response `http_looks_bad` (bad status, too short, or a Cloudflare/JS-challenge marker) does it fall back to a lazily-launched, shared Playwright Chromium (`browser_get`). Text extraction is in `scraper/extract/` (trafilatura + per-domain overrides in `extract/overrides/`). `store.Store` uses an **autocommit connection — one row per transaction**, so a run is safe to kill and resume; already-`ok` URLs are skipped unless `--retry-errors`. `robots.txt` is honored unless `--ignore-robots`. Settings come from `SCRAPER_*` env vars (see `scraper/config.py`).

## Knowledge graph (`graphify-out/`)

A pre-built knowledge graph of this repo lives in `graphify-out/` (520 nodes, 974 edges, 29 communities). For architecture questions, query it instead of grepping — answers cost ~2k tokens vs ~35k for raw reads:

```powershell
graphify query "<question>"          # BFS context from graphify-out/graph.json
```

Structural facts it surfaced:
- **Core abstractions (most-connected nodes):** `RawPage`, `Settings`, `ArticleJob`, `DomainLimiter`, `Article`, `process_job()`, `Fetcher`, `RobotsCache` — the scraper's data-flow spine. Changes to these ripple widest.
- **`embed_query()` is the bridge** between embedding and both search tools (the PYTHONPATH gotcha above).
- **Market-time handling (`ZoneInfo`) is duplicated** across `ticker_candles.py`, `scan_ranges.py`, `catalyst_returns.py`, and the sentiment scripts — no shared market-calendar module; keep their premarket/after-hours conventions in sync manually.
- Tests stub the pipeline via `FakeFetcher` / `FakeStore` / `FakeRobots` in `tests/test_pipeline.py`, all driven by the same `Settings` object as production.

## Commands

Setup (Windows / PowerShell):
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium   # required by the scraper's browser fallback
```
On macOS/Linux, `./setup_venv.sh` does the venv + pip steps (still run `playwright install chromium` after).

Database:
```powershell
docker compose up -d           # pgvector/pgvector:pg16, db=news, user/pass scraper, :5432
```
The scraper applies `scraper/store/schema.sql` automatically on its first run.

Run the pipeline:
```powershell
python run_scrape.py --csv <news.csv>            # scrape (--limit, --retry-errors, --ignore-robots, --concurrency)
python scripts/embedding/embed_articles.py       # embed (--limit, --reembed, --batch-size, --chunk-pool, --no-index)
python scripts/classify/classify_articles.py     # categorize articles (two-pass Gemini)
python scripts/enrichment/tag_segments.py        # tag tickers/segments
python scripts/enrichment/extract_insights.py    # chunk articles into embedded insight boxes
$env:PYTHONPATH="scripts/embedding"; python scripts/search/search_articles.py "nvidia data center demand" --k 5 --ticker NVDA
$env:PYTHONPATH="scripts/embedding"; python scripts/search/search_articles_by_insights.py "guidance cut" --k 10
```

Tests (pytest config in `pyproject.toml`, `asyncio_mode=auto`):
```powershell
pytest                                  # full suite
pytest -m "not db and not integration"  # offline only — skips tests needing Postgres or live network
pytest tests/test_urls.py               # one file
pytest tests/test_pipeline.py::test_name -q   # one test
```
The `db` marker = needs a running Postgres; `integration` = hits the live network.

Use Context7 for best practices with LangFuse and LangGraph