# Pipeline v2: clean durable pipeline, live-feed-ready, LangChain/LangGraph + Langfuse

**Date:** 2026-06-10
**Status:** Approved design, pending implementation
**Branch:** `refactor/pipeline-v2` → PR to `main`

## Goal

Consolidate the current collection of stage scripts into a single clean Python package with a
screaming-architecture layout, run it as a continuous service ready for a live news feed (provider
not yet chosen — websocket or polling, abstracted behind an interface), rebuild LLM stages on
LangChain/LangGraph, and add per-article Langfuse observability. Evals are the next milestone after
this; the design must be eval-ready.

## Decisions (settled with the user)

| Question | Decision |
|---|---|
| Runtime shape | Continuous service: feed adapter enqueues, async worker pool processes each article through all stages. Batch CLI remains for backfill. |
| Pipeline scope | Service covers collect → scrape → embed → classify → tag → insights → **sentiment**. Search CLIs, backtests, ticker_scan stay as on-demand `research/` tools. |
| Live feed | Provider unknown. Build a `NewsFeedSource` protocol now; ship a Massive REST poller + CSV backfill source; the real-time source plugs in later as one new file. |
| LLM stack | LangChain 1.3.x + LangGraph 1.0.x + langchain-google-genai 4.2.x + langchain-openai 1.1.x. Exact versions pinned. |
| Sentiment | LangGraph `StateGraph` orchestrator: `Send` fan-out to fixed-role analyst sub-agents in parallel, synthesizer produces structured verdict. Not a supervisor pattern (no dynamic routing needed). |
| Observability | Langfuse SDK 4.x, one trace per article spanning all stages (non-LLM spans + nested LLM generations). |
| Langfuse hosting | Self-hosted via Docker Compose (v3+ architecture: web, worker, Postgres, ClickHouse, Redis, MinIO). Cloud fallback is an `.env` change only. |
| Migration | Clean break: old `scripts/` and root `run_*.py` deleted once CLI covers them. One-offs (e.g. `reclassify_real_news.py`) archived via git history. No compatibility shims. |
| Process | All work on `refactor/pipeline-v2`, PR to `main`. |

## Package layout

Installable package (`pip install -e .`), `src/` layout, console entry point `ticker-news`.

```
src/ticker_news/
  ingestion/
    feed.py            # NewsFeedSource protocol: async iterator of FeedItem(url, tickers, published_utc, source_meta)
    massive_rest.py    # polling adapter over Massive news API (from scripts/data_getting_parsing/ticker_news.py)
    csv_backfill.py    # CSV-driven backfill source (current CSV path)
    universe.py        # ~118-ticker AI-compute universe loader (from run_universe.py)
  scraping/            # scraper/ package moves here nearly intact (fetcher, extract/, robots, limiter, store)
  embedding/           # from scripts/embedding/embed_articles.py — embedder (OpenAI client via langchain-openai)
                       #   + store ops; embed_query() importable normally (PYTHONPATH hack dies)
  classification/      # two-pass Gemini classify as plain LangChain structured-output chains
  enrichment/
    tagging.py         # from tag_segments.py
    insights.py        # from extract_insights.py (LangChain structured output)
    reference_data.py  # from load_ticker_data.py + load_ticker_overview.py
  sentiment/
    graph.py           # LangGraph StateGraph (see Sentiment section)
    analysts.py        # role registry: role name → prompt (fallback copies of Langfuse prompts)
    schemas.py         # Verdict, AnalystReport pydantic models
  research/            # on-demand tools, imports from the package:
    search.py          #   search_articles.py + search_articles_by_insights.py
    backtest.py        #   backtest_top2.py, followup_sentiment.py
    candles.py         #   checks_backtesting/ticker_candles.py
    ticker_scan/       #   scan_ranges, attach_articles, catalyst_returns, render_*
  service/
    worker.py          # async worker pool: claim job → run stage chain → advance state
    jobs.py            # pipeline_jobs access: enqueue, claim (FOR UPDATE SKIP LOCKED), complete, fail, retry
  shared/
    config.py          # ONE pydantic-settings object, ONE DATABASE_URL (ends SCRAPER_DB_DSN vs NEWS_DB_DSN split)
    db.py              # asyncpg/psycopg pool helpers
    llm.py             # model factory: Gemini + OpenAI chat/embedding models with retry, fallbacks,
                       #   shared InMemoryRateLimiter, Langfuse CallbackHandler — single choke point
    observability.py   # Langfuse client setup, @observe helpers, trace-id seeding
    market_time.py     # market-calendar/ZoneInfo logic currently duplicated across 4 files
  cli.py               # typer: serve | backfill | search | backtest | scan | jobs | load-reference
```

`service/` and `shared/` are the only technical-sounding folders; everything else screams the
domain: ingestion, scraping, classification, enrichment, sentiment, research.

## Service runtime: Postgres-backed queue

New table (additive migration, no destructive schema changes):

```sql
CREATE TABLE pipeline_jobs (
  article_url   text PRIMARY KEY,
  stage         text NOT NULL DEFAULT 'scrape',   -- scrape|embed|classify|tag|insights|sentiment|done
  status        text NOT NULL DEFAULT 'pending',  -- pending|running|failed|done
  attempts      int  NOT NULL DEFAULT 0,
  last_error    text,
  enqueued_at   timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);
```

- `ticker-news serve` runs the feed source task + an async worker pool.
- Workers claim with `FOR UPDATE SKIP LOCKED`; each drives one article through the remaining
  stages, advancing `stage` after each step — a crash resumes mid-article.
- Pickup: short-interval polling + `LISTEN/NOTIFY` wake on enqueue.
- Failures: `attempts` increments with capped exponential backoff; over-cap jobs park as `failed`,
  inspectable/requeueable via `ticker-news jobs retry`.
- Concurrency: article-level `asyncio.Semaphore`; analyst fan-out capped via LangGraph
  `max_concurrency`; provider quotas guarded by the shared rate limiter in `shared/llm.py`.
- Backfill (`ticker-news backfill --csv ...`) enqueues through the same path; idempotent: stages
  skip work already present (embedding not NULL, category set, etc.), preserving current
  resumability semantics.

## Feed port

```python
class FeedItem(BaseModel):
    url: str
    tickers: list[str]
    published_utc: datetime
    source_meta: dict  # provider sentiment, publisher, etc.

class NewsFeedSource(Protocol):
    def stream(self) -> AsyncIterator[FeedItem]: ...
```

The service consumes any implementation and enqueues. Ships now:
- `MassiveRestSource` — polls the Massive news endpoint per universe ticker on an interval,
  dedupes against the DB. Makes the pipeline genuinely live today.
- `CsvBackfillSource` — wraps the existing CSV flow.

When the real-time provider is chosen (websocket or polling), it is one new file implementing
`NewsFeedSource`. Nothing downstream changes.

## LLM stack and stage design

Pinned: `langchain` 1.3.x, `langgraph` 1.0.x, `langfuse` 4.x, `langchain-google-genai` 4.2.x,
`langchain-openai` 1.1.x. Both ecosystems shipped breaking rewrites recently (LangChain 1.0
2025-10, Langfuse v4 2026-03); pin exact versions, upgrade deliberately.

**Simple stages (classification, insights):** plain LangChain —
`prompt | model.with_structured_output(Schema)`. No graph machinery. Pydantic schemas with enums
(content category, buy/sell/hold). Models built only by `shared/llm.py` with `.with_retry()`,
`.with_fallbacks()` (gemini-2.5-flash → flash-lite), and a shared `InMemoryRateLimiter`.

**Sentiment (LangGraph orchestrator-worker, `Send` fan-out):**

- State: `article`, `analyses: Annotated[list, operator.add]` (reducer makes parallel writes
  safe), `verdict: Verdict`.
- Orchestrator edge fans out via `Send` to one parameterized `analyst` node — one parallel
  invocation per fixed role. Initial roster (pluggable; one dict entry + one prompt to add):
  - `fundamentals` — impact on the company's fundamentals
  - `market_context` — sector/market backdrop at publication time
  - `historical_precedent` — pgvector search for similar past articles + what followed
- `synthesize` node consumes all analyst reports → `Verdict(action: buy|sell|hold,
  confidence: float [0..1], reasoning: str)` via structured output.
- Why not a supervisor: supervisors are for dynamic LLM routing; these analysts are fixed-role,
  always all run, on the same input. `Send` fan-out is cheaper (no routing calls), deterministic,
  parallel by construction, and each analyst is independently traceable/evaluable.
- Analysts are single LLM calls (historical_precedent may be a 2-node subgraph for the vector
  search). No `create_agent`/`create_react_agent` (the latter is deprecated in LangGraph 1.x).

## Langfuse observability (SDK v4 APIs)

- One trace per article: worker wraps `process_article` with `@observe(name="process-article")`;
  ticker/URL/tags set via `propagate_attributes()` (v4 replacement for `update_current_trace`;
  metadata values must be strings ≤200 chars — known SDK bug drops non-strings).
- Non-LLM stages (scrape, embed, DB writes) get explicit spans via
  `start_as_current_observation(as_type="span", name=...)`.
- All chain/graph invokes receive `langfuse.langchain.CallbackHandler` (new import path) in
  `config={"callbacks": [...]}` — LLM generations nest under the current trace automatically.
- Trace IDs seeded from article URL (`Langfuse.create_trace_id(seed=url)`) so re-runs correlate.
  `flush()`/`shutdown()` on worker exit.
- Stable observation names per stage: `process-article`, `scrape`, `embed`, `classify`, `tag`,
  `insights`, `analyst:<role>`, `synthesize`.
- **Prompt management:** prompts live in Langfuse Prompts (versioned, `production`/`staging`
  labels, linked to traces), fetched with client-side caching; committed in-repo fallback copies
  so the service boots if Langfuse is down.
- **Docker:** extend project `docker-compose.yml` from the official Langfuse compose file —
  six services since v3: `langfuse-web` (:3000), `langfuse-worker`, Postgres, ClickHouse,
  Redis/Valkey, MinIO. Needs ~8–16 GB RAM headroom. Required env: `SALT`, `ENCRYPTION_KEY`,
  `NEXTAUTH_URL`, ClickHouse/Redis/S3 creds. Swapping to Langfuse Cloud later = `.env` change only.

## Eval-readiness (next milestone, designed in now)

- Stable trace/observation naming (above) — Langfuse dataset extraction and UI LLM-as-judge
  evaluators filter on names.
- Root trace input/output explicitly set (article ref in, verdict JSON out) → production traces
  convert directly to dataset items (`create_dataset_item(..., source_trace_id=...)`).
- Structured output everywhere → exact-match and judge scorers get parseable data.
- Metadata discipline: ticker, category, model on every trace → slice experiment results.
- Ground-truth loop: backtest logic writes realized returns back as **scores on production
  traces**, yielding an outcome-labeled eval dataset for free.
- Experiments later via Langfuse datasets + `item.run(...)` runners; prompt A/B via Langfuse
  prompt versions linked to experiment runs.

## Config and testing

- `shared/config.py`: one pydantic-settings object; `DATABASE_URL` (single DSN), API keys
  (`MASSIVE_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `LANGFUSE_*`), scraper knobs
  (current `SCRAPER_*` vars fold in), service knobs (worker count, poll interval, retry caps).
- Tests mirror the package (`tests/scraping/`, `tests/service/`, `tests/sentiment/`...).
  Existing `FakeFetcher`/`FakeStore`/`FakeRobots` pattern carries over.
- New fakes: `FakeFeedSource` (scripted FeedItems) and LangChain `GenericFakeChatModel` so
  chains and the sentiment graph run offline.
- `db`-marked tests hit `news_test` only (never the real `news` DB — standing rule).
- One end-to-end integration test: fake feed → worker → all stages with fake LLMs against test DB.

## Migration plan (incremental, tests green at every step)

1. Scaffold `pyproject.toml` packaging, `src/ticker_news/`, `shared/config.py`, CLI skeleton.
2. Move `scraper/` → `ticker_news/scraping/`; rewrite imports; tests move and pass.
3. Move stages one at a time (embedding → classification → enrichment), converting LLM calls to
   LangChain structured-output chains; each move lands with its tests before the next starts.
4. Build `service/` (jobs table migration, worker loop) + `ingestion/` (protocol, Massive poller,
   CSV backfill).
5. Build `sentiment/` LangGraph graph.
6. Wire Langfuse (compose services, `shared/observability.py`, callbacks, prompts).
7. Port `research/` tools to package imports.
8. Delete `scripts/`, root `run_*.py`, old `scraper/` once `ticker-news` CLI covers every old
   command. Update CLAUDE.md.

DB schema is only added to; old scripts keep working mid-migration. PR from
`refactor/pipeline-v2` to `main` at the end (or stacked PRs per milestone if review size warrants).

## Out of scope (this effort)

- Choosing/integrating the real-time feed provider (interface ships, provider lands later).
- Implementing evals/experiments (designed for, not built).
- Kubernetes/production Langfuse deployment (compose is the dev/single-machine path).
- Re-running historical backfills or re-embedding existing rows.
