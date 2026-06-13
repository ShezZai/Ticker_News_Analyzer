# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline for AI-compute-sector stock news: collect metadata → scrape full text → embed (pgvector) → classify (Gemini) → tag tickers/segments → extract insights → judge sentiment (LangGraph analyst panel). One installable package, `src/ticker_news/`, organized by domain (screaming architecture): `ingestion/`, `scraping/`, `embedding/`, `classification/`, `enrichment/`, `sentiment/`, `service/`, `research/`, `shared/`. Single CLI entry point: `ticker-news` (Typer app in `src/ticker_news/cli.py`).

Core Postgres tables (one `news` database): `articles` (scraped text + `embedding vector(1536)`), `pipeline_jobs` (the work queue), `article_insights` (embedded insight boxes), `article_sentiment` (verdicts, PK article_id+ticker), `ticker_data` / `ticker_overview` (universe reference data).

## Architecture

Two ways to run the same stage chain (`scrape → embed → classify → tag → insights → sentiment`, defined in `service/jobs.py:STAGES`):

- **Continuous service** — `ticker-news serve` polls the Massive REST API (`ingestion/massive_rest.py`) and processes articles end to end; `ticker-news backfill --csv ...` does the same in drain mode (exits when feed exhausted + queue empty). Both go through `service/worker.serve()`: a feed task enqueues into `pipeline_jobs`, an async worker pool drives each article through the remaining stages (`service/stages.py` adapters), advancing `stage` after each step so a crash resumes mid-article.
- **Per-stage batch CLIs** — `ticker-news embed|classify|tag|insights|sentiment` etc. for corpus-wide, resumable work (each picks up only pending rows unless `--reprocess`/`--reembed`).

**Queue design** (`service/jobs.py`): one row per article URL; workers claim with `FOR UPDATE SKIP LOCKED`; failures retry with exponential backoff (30s base, 1h cap, 5 attempts) then park as `failed` (requeue via `ticker-news jobs retry`); `PermanentStageError` parks immediately. `NOTIFY pipeline_jobs` wakes the service instantly on enqueue; polling is the fallback. `jobs.recover_orphans()` resets `running` rows at startup (single-service assumption).

**Sentiment** (`sentiment/graph.py`): LangGraph orchestrator — three fixed-role analysts (`fundamentals`, `market_context`, `historical_precedent`) fan out in parallel via the Send API (gemini-2.5-flash-lite), then a synthesis judge (gemini-2.5-flash, structured output) produces a buy/sell/hold `Verdict`. No supervisor; roles are static. Stored in `article_sentiment`.

**Feed port** (`ingestion/feed.py`): `NewsFeedSource` protocol (`stream() -> AsyncIterator[FeedItem]`). A future real-time provider (websocket etc.) is one new file implementing it; nothing downstream changes. Current impls: `MassiveRestSource` (live polling), `CsvBackfillSource` (drain).

**Research** (`research/`): on-demand search/scan/backtest/chart tools under `ticker-news research` plus `search` / `search-insights` — read the DB, never write pipeline tables.

## Commands

Setup:
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[dev,charts]"
playwright install chromium     # scraper's browser fallback
docker compose up -d            # pgvector/pgvector:pg16, db=news, user/pass scraper, :5432
```
If the `news_pg` container already exists from a prior run, `docker start news_pg` instead. On macOS/Linux, `./setup_venv.sh` does venv + pip. Scraper schema (`scraping/store/schema.sql`) applies automatically on first run.

| Command | Does |
|---|---|
| `ticker-news serve` | Live pipeline: poll Massive feed, process end to end (`--workers`, `--tickers`, `--lookback-hours`) |
| `ticker-news backfill --csv F` | Enqueue a news CSV, process to completion, exit (drain mode) |
| `ticker-news fetch-news --start D` | Fetch news metadata + provider sentiment from Massive into a CSV |
| `ticker-news scrape --csv F` | Scrape article bodies from a CSV into `articles` (`--retry-errors`, `--ignore-robots`) |
| `ticker-news embed` | Embed rows missing an embedding; builds HNSW index (`--reembed`, `--no-index`) |
| `ticker-news classify` | Two-pass Gemini categorization (`--reprocess`, `--ids`, `--workers`) |
| `ticker-news tag` | Derive primary/secondary tickers + segments from `tickers[]` |
| `ticker-news insights` | Extract + embed insight boxes (`--embed-only`, `--fix-quotes`) |
| `ticker-news sentiment` | Batch analyst-panel verdicts for real-news articles missing one |
| `ticker-news load-universe` / `load-overviews` | Populate `ticker_data` / `ticker_overview` |
| `ticker-news search` / `search-insights` | pgvector ANN search (`--like ID`, `--ticker`, `--segment`, `--since`, `--ef-search`) |
| `ticker-news research chart\|scan-ranges\|attach-articles\|catalyst-returns\|backtest\|render-*` | Charts, big-mover scans, buy-the-news + verdict backtests |
| `ticker-news jobs status` / `jobs retry [--url U]` | Queue counts; requeue failed jobs |
| `ticker-news eval pipeline --ids N[,..]` | E2E eval: re-run articles through every stage, score verdict vs realized price move as a Langfuse experiment (`--dataset` for run-over-run comparison, `--dsn` for the shared DB, `--ids-file` for an id-per-line file, `--skip-stages embed[,..]\|all` to reuse stable stage outputs, `--precedent-source` to A/B the sentiment precedent retrieval) |
| `ticker-news eval classify` | Single-pass classifier experiments: binary vs `140-articles-act-no-act`, finegrained vs `140-articles-categories`, **finegrained-act** (finegrained classifier, category collapsed to ACT via `is_act_finegrained`, scored vs `140-articles-act-no-act` for YES-class precision/recall/F1 directly comparable to binary); scores accuracy + time/cost per run, plus binary YES-class precision/recall/F1 and finegrained per-category macro precision/recall/F1 + support-weighted F1 (`--variant binary\|finegrained\|finegrained-act\|both`; `both` = binary+finegrained, finegrained-act is opt-in; `--model lite\|flash`, `--ids`, `--concurrency`) |
| `ticker-news prompts push` | Upsert in-repo prompts to Langfuse with the `production` label |

## Configuration

All config flows through `shared/config.py` (`AppSettings`, pydantic-settings, reads `.env`):

- `DATABASE_URL` — the single DSN for every stage (default `postgresql://scraper:scraper@localhost:5432/news`). `SCRAPER_DB_DSN` is accepted as a legacy fallback alias only.
- `MASSIVE_API_KEY` — news feed, candles, all price data. `OPENAI_API_KEY` — embeddings. `GOOGLE_API_KEY` — all Gemini calls (classify, insights, sentiment).
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` (default `https://cloud.langfuse.com`) — keys absent ⇒ tracing strictly disabled, every helper no-ops.
- `SCRAPER_*` knobs (concurrency, per-domain limits, timeouts, UA) keep their legacy names — see `AppSettings`.
- `SENTIMENT_PRECEDENT_SOURCE` — retrieval mode for the historical-precedent analyst: `article` (default; top-5 similar articles) | `insights` (per-insight-box ANN) | `distilled-first` / `distilled-second` (distilled boxes tagged by first/second label; `-second` drops `DROP`-labelled). Plus `SENTIMENT_PRECEDENT_INSIGHTS_THRESHOLD` (0.7) / `..._LIMIT` (40). Per-run override: `eval pipeline --precedent-source`. See `docs/precedent-source-options.md`.

## Observability & prompts

- **One Langfuse trace per article** (`shared/observability.article_trace`), trace id deterministically seeded from the URL — re-runs and batch re-judges land in the same trace; `entrypoint` metadata (`service`/`batch`) distinguishes runs.
- **Stable observation names are the eval contract** — do not rename: `process-article` (root), `scrape`, `embed`, `classify`, `tag`, `insights`, `sentiment` (with `precedents` child carrying the effective `precedent_source` + retrieval counts), `analyst:<role>`, `synthesize`. Root output carries `category` + `verdict`; metadata carries `prompt_versions` + `entrypoint`. Evals: `eval pipeline` (E2E verdict vs price move) and `eval classify` (single-pass prompt experiments) both run as Langfuse experiments.
- **Prompts** (`shared/prompts.py`): in-repo fallback templates are the source of truth; `ticker-news prompts push` publishes them to Langfuse under the `production` label. `get_prompt()` prefers the Langfuse copy, falls back silently. Chains are `lru_cache`d — restart the process to pick up Langfuse prompt edits.
- **Prompt↔generation linking**: `get_prompt_entry()` returns `(text, prompt_object)`; attach the object via `prompt.metadata = {"langfuse_prompt": obj}` (see `classification/variants._build`) so generations carry a `promptId`. Never use `get_langchain_prompt()` — it mangles the escaped `{{...}}` JSON braces in our f-string templates. Eval chains are rebuilt per run (no cache), so Langfuse prompt edits apply immediately there.
- **Eval datasets in Langfuse**: `140-articles-act-no-act` (expected `{label: YES|NO}`, 42/98) and `140-articles-categories` (expected `{label: <FinegrainedCategory>}`); items hold only `{article_id}` — bodies are read from the DB at run time, never stored in the dataset.
- **Accepted gap**: embedding costs are not traced (OpenAI embeddings bypass the LangChain callback handler). `eval classify` computes run cost from the in-repo price table `classify_eval.GEMINI_PRICES_USD_PER_1M` — update it if Google reprices.

## Testing

```powershell
pytest                                  # full suite
pytest -m "not db and not integration"  # offline only
```
Markers (pyproject.toml): `db` = needs Postgres, `integration` = live network.

- **`db` tests target `news_test` ONLY.** `tests/scraping/conftest.py` connects via `TICKER_NEWS_TEST_DSN` (default `postgresql://scraper:scraper@localhost:5432/news_test`), auto-creates the database, and **refuses to run unless the db name contains `news_test`** — the fixture TRUNCATEs `articles`. NEVER point it at the real `news` db; a past incident wiped the articles table.
- `tests/conftest.py` neutralizes `.env` (`AppSettings.model_config["env_file"] = None`) and deletes shell `LANGFUSE_*` vars per test, so tests never read developer secrets or export traces.

## Critical gotchas

- **pgvector extension**: `CREATE EXTENSION vector;` must run once in the db (the scraper's `schema.sql` does it; `embedding/pipeline.py` only checks and fails loudly).
- **HNSW two-query rule**: for similar-to-article search, fetch the seed embedding first, then run a separate `ORDER BY embedding <=> %s` query — a join-embedded form silently defeats the HNSW index (seq scan). See `research/search.py`.
- **Embeddings are unit-normalized** (text-embedding-3-small, 1536 dims; cosine == inner product). Query and stored vectors must come from the same code path — `embedding/embedder.py`. Don't fork the model/truncation config.
- **Sync psycopg connections are not shareable** across threads/concurrent `to_thread` calls — each service worker owns its own connection + scraper `Store` (`service/worker.py`). Follow that pattern for any new concurrency.
- **Langfuse `CallbackHandler` instances are not thread-safe** — `obs.chain_config()` builds a fresh handler per invocation; do that inside thread-pool workers, never share one handler.
- **langfuse `run_experiment` silently drops errored items** from `item_results` before run-level evaluators execute, shrinking metric denominators. `evals/classify_eval._with_missing_items` re-injects them as `output=None` ("errored = wrong") — wrap run evaluators the same way in any new experiment.
- **The universe CSV is not committed** (`*.csv` gitignored). `load-universe` expects `ai_compute_us_market_universe_consolidated_segments_min5.csv` at repo root; `research scan-ranges` reads `public.ticker_data` instead, so run `load-universe` first to populate it.
- **`charts` extra required** for `research chart` / `render-*` (pandas/matplotlib/mplfinance are lazy-imported; missing ⇒ clear error telling you to `pip install -e ".[charts]"`).
- **Scraper behavior**: HTTP-first (httpx), Playwright Chromium fallback only on bad responses; per-domain rate limiting; robots.txt honored unless `--ignore-robots`; autocommit store (one row per transaction) makes runs kill-safe and resumable.

The knowledge graph in `graphify-out/` is stale (pre-refactor `scripts/` layout) — do not trust it until regenerated.

Use Context7 for best practices with LangFuse and LangGraph
