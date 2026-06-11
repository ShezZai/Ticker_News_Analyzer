# Ticker News Analyzer

An end-to-end pipeline for **AI-compute-sector stock news**. It collects article metadata,
scrapes full text, embeds it into pgvector, classifies it with Gemini, tags tickers/segments,
extracts decision-useful insight boxes, and judges buy/sell/hold sentiment with a LangGraph
analyst panel — all traced per-article in Langfuse.

One installable package (`src/ticker_news/`), one Postgres database (`news`), one CLI
entry point (`ticker-news`).

---

## High-level architecture

The pipeline is a fixed stage chain — `scrape → embed → classify → tag → insights → sentiment`
(defined in `service/jobs.py:STAGES`). The same chain runs two ways:

```
                         ┌─────────────────────────────────────────────────────────┐
   News feed             │                     STAGE CHAIN                          │
 (Massive REST /         │  scrape → embed → classify → tag → insights → sentiment  │
  CSV backfill)          └─────────────────────────────────────────────────────────┘
        │                          │        │         │       │        │
        ▼                          ▼        ▼         ▼       ▼        ▼
  pipeline_jobs  ──claim──►   articles(+embedding vector)   article_insights   article_sentiment
   (work queue)              ticker_data / ticker_overview (universe reference data)
```

- **Continuous service** (`ticker-news serve`): polls the Massive REST feed, enqueues one row
  per article URL into `pipeline_jobs`, and an async worker pool drives each article through the
  remaining stages — advancing `stage` after each step so a crash resumes mid-article.
  `ticker-news backfill --csv ...` is the same machinery in **drain mode** (exits when the feed
  is exhausted and the queue is empty).
- **Per-stage batch CLIs** (`embed`, `classify`, `tag`, `insights`, `sentiment`, …): corpus-wide,
  resumable work. Each picks up only pending rows unless told to reprocess.

**Domain layout** (screaming architecture):

| Domain | Responsibility |
|---|---|
| `ingestion/` | Feed sources — Massive REST poller, CSV backfill (`NewsFeedSource` port) |
| `scraping/` | HTTP-first fetch (httpx) with Playwright fallback, per-domain rate limiting, body extraction |
| `embedding/` | `text-embedding-3-small` (1536-dim, unit-normalized) + HNSW index |
| `classification/` | Two-pass Gemini categorization (lite → flash confirm) |
| `enrichment/` | Ticker/segment tagging, insight-box extraction, reference data |
| `sentiment/` | LangGraph analyst panel + synthesis judge → buy/sell/hold verdict |
| `service/` | Job queue (`pipeline_jobs`), worker pool, stage adapters |
| `research/` | On-demand search, big-mover scans, charts, backtests (read-only) |
| `shared/` | Config, DB connection, LLM factory, prompts, Langfuse observability |

**Sentiment graph** (`sentiment/graph.py`): three fixed-role analysts (`fundamentals`,
`market_context`, `historical_precedent`) fan out in parallel via the LangGraph Send API
(gemini-2.5-flash-lite), then a synthesis judge (gemini-2.5-flash, structured output) produces
the `Verdict`.

Core tables (one `news` database): `articles` (scraped text + `embedding vector(1536)`),
`pipeline_jobs` (work queue), `article_insights`, `article_sentiment` (PK `article_id`+`ticker`),
`ticker_data` / `ticker_overview` (universe reference data).

---

## Prerequisites

- **Python 3.11+**
- **Docker** (for the pgvector Postgres) — or your own Postgres 16 with the `vector` extension
- API keys:
  - `MASSIVE_API_KEY` — news feed, candles, all price data
  - `OPENAI_API_KEY` — embeddings
  - `GOOGLE_API_KEY` — all Gemini calls (classify, insights, sentiment)
  - *(optional)* Langfuse keys — tracing; absent ⇒ all observability helpers no-op

---

## Setup

```powershell
# 1. Virtual environment + install
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,charts]"

# 2. Scraper browser fallback
playwright install chromium

# 3. Database (pgvector/pgvector:pg16, db=news, user/pass scraper, :5432)
docker compose up -d
```

On macOS/Linux, `./setup_venv.sh` does the venv + pip steps.
If the `news_pg` container already exists from a prior run, use `docker start news_pg` instead of
`docker compose up -d`.

The pgvector extension and the scraper schema (`scraping/store/schema.sql`) are applied
automatically on first run.

### `.env`

Create a `.env` at the repo root (loaded by `shared/config.py`):

```dotenv
# Database — the single DSN for every stage
DATABASE_URL=postgresql://scraper:scraper@localhost:5432/news

# API keys
MASSIVE_API_KEY=your-massive-key
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Langfuse (optional — see "Observability" below)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

All keys default to `None` and Langfuse is **disabled** unless both keys are present.
Scraper knobs (`SCRAPER_CONCURRENCY`, `SCRAPER_PER_DOMAIN`, `SCRAPER_DOMAIN_DELAY`,
`SCRAPER_HTTP_TIMEOUT`, `SCRAPER_MIN_WORDS`, `SCRAPER_RESPECT_ROBOTS`, `SCRAPER_UA`) keep their
legacy names — see `AppSettings` in `shared/config.py`.

### Observability — Langfuse Cloud (free tier)

Tracing is optional but recommended; the free **Hobby** tier is enough for development.

1. Sign up at **https://cloud.langfuse.com** (free tier, no card required).
2. Create an **Organization** → **Project**.
3. In **Project Settings → API Keys**, click **Create new API keys**. Copy the **Public key**
   (`pk-lf-…`) and **Secret key** (`sk-lf-…`).
4. Put them in `.env` as `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`. Set `LANGFUSE_HOST` to the
   region shown in the dashboard (`https://cloud.langfuse.com` for US, `https://eu.cloud.langfuse.com`
   for EU). Default is US.
5. *(Optional)* Publish the in-repo prompts to Langfuse under the `production` label so you can
   edit them in the UI:
   ```powershell
   ticker-news prompts push
   ```

With keys set, you get **one trace per article** (trace id deterministically seeded from the URL,
so re-runs land in the same trace). Without keys, every observability helper degrades to a no-op —
nothing is exported and no developer secrets are read. **Restart the process** to pick up prompt
edits made in the Langfuse UI (prompt chains are `lru_cache`d).

---

## Usage

### Run the whole pipeline

```powershell
# Live: poll the Massive feed and process end to end
ticker-news serve --workers 4 --tickers NVDA,AMD --lookback-hours 24

# Drain a CSV to completion, then exit
ticker-news backfill --csv news.csv

# Fetch news metadata (+ provider sentiment) from Massive into a CSV first
ticker-news fetch-news --start 2025-01-01
```

### Run a single stage (corpus-wide, resumable)

```powershell
ticker-news scrape --csv news.csv      # scrape bodies into `articles`
ticker-news embed                      # embed rows missing an embedding; build HNSW index
ticker-news classify                   # two-pass Gemini categorization
ticker-news tag                        # primary/secondary tickers + segments
ticker-news insights                   # extract + embed insight boxes
ticker-news sentiment                  # analyst-panel verdicts for real-news articles
```

### CLI reference

| Command | What it does |
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
| `ticker-news jobs status` / `jobs retry [--url U]` | Queue counts; requeue failed jobs |
| `ticker-news prompts push` | Upsert in-repo prompts to Langfuse with the `production` label |

Every command supports `--help`. Start with `ticker-news --help` for the full tree.

### Research tools (read-only)

These read the DB and never write pipeline tables. Charting requires the `charts` extra
(`pip install -e ".[charts]"`).

```powershell
ticker-news research scan-ranges --threshold 5         # flag big intraday-range days
ticker-news research attach-articles SCAN.csv          # attach same-day articles to movers
ticker-news research catalyst-returns --start 2025-02-01  # buy-the-news returns per catalyst
ticker-news research backtest --start 2025-02-01       # backtest analyst-panel verdicts vs returns
ticker-news research chart NVDA 2025-03-10             # marked intraday candle chart
ticker-news research render-bombs|render-catalysts|render-all   # batch chart rendering
```

> The universe CSV is **not committed** (`*.csv` is gitignored). `load-universe` expects
> `ai_compute_us_market_universe_consolidated_segments_min5.csv` at the repo root and must run
> before `research scan-ranges` (which reads `public.ticker_data`).

---

## Testing

```powershell
pytest                                  # full suite
pytest -m "not db and not integration"  # offline only (no Postgres, no network)
```

Markers (`pyproject.toml`):
- `db` — needs a running Postgres
- `integration` — hits the live network

> ⚠️ **`db` tests target `news_test` ONLY.** The fixture connects via `TICKER_NEWS_TEST_DSN`
> (default `postgresql://scraper:scraper@localhost:5432/news_test`), auto-creates that database,
> and **refuses to run unless the db name contains `news_test`** — it TRUNCATEs `articles`.
> Never point it at the real `news` database.

Tests also neutralize `.env` and strip shell `LANGFUSE_*` vars per test, so they never read
developer secrets or export traces.

---

## Gotchas

- **pgvector extension** must exist once in the db (`CREATE EXTENSION vector;`) — the scraper's
  `schema.sql` does it; `embedding/pipeline.py` only checks and fails loudly.
- **HNSW two-query rule**: for similar-to-article search, fetch the seed embedding first, then run
  a separate `ORDER BY embedding <=> %s` query — a join-embedded form silently defeats the index.
- **Embeddings are unit-normalized** (cosine == inner product). Query and stored vectors must come
  from the same code path (`embedding/embedder.py`).
- **Sync psycopg connections aren't shareable** across threads — each service worker owns its own
  connection + scraper `Store`.
- The `charts` extra (pandas/matplotlib/mplfinance) is **lazy-imported**; missing it gives a clear
  error telling you to install it.

See `CLAUDE.md` for the full architecture notes, queue design, and the Langfuse eval contract.
