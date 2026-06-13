# E2E Pipeline Eval (Langfuse Experiment) — Design

**Date:** 2026-06-11
**Status:** Approved

## Goal

A repeatable eval that runs a single article through the whole post-scrape
pipeline (`embed → classify → tag → insights → sentiment`) against the shared
team database, then scores the produced verdict against what the stock
actually did: entry price at publication time vs. the close of that trading
day, fetched from the Massive API. Results land in Langfuse as experiment
runs with per-item scores, so prompt/model changes can be compared run-over-run.

## Context

- The shared DB (`sharedproject`, reached via cloudflared tunnel at
  `localhost:15432`) holds ~20.5k scraped articles (Nov 2024 – May 2026),
  already embedded/classified/tagged, with ~90k insight rows and a 117-ticker
  `ticker_data` universe. It has **no** `article_sentiment` or `pipeline_jobs`
  tables, and its `articles` table predates the `insights_extracted_at` /
  `provider_sentiments` columns.
- All four DB roles (robert, pyapp, shay, appuser) are superusers — write
  access verified 2026-06-11 with a rolled-back UPDATE/DELETE.
- Langfuse SDK in the project is v4 (`langfuse>=4,<5`) — the experiment
  runner (`langfuse.run_experiment` / `dataset.run_experiment`) is current.
- `research/ticker_scan.py` already implements the price rule
  (`load_prices` + `simulate`: buy at first tradeable minute at/after the
  headline, sell at that day's regular close; at/after 16:00 ET rolls to the
  next session). `research/backtest.py` uses the same primitives.

## Decisions (made with user)

1. **Form:** Langfuse experiment (not a pytest gate). A wrong verdict is a
   score, not a broken build.
2. **Target DB:** directly against the shared DB. Verdicts accumulate there
   for the whole team; the eval creates/heals schema additively.
3. **Scoring:** directional agreement — buy+up = 1, sell+down = 1, wrong
   direction = 0; hold / skipped / missing-price = `None` (excluded) with an
   explanatory comment. Raw signed move recorded as a second score.
4. **Pipeline scope:** force re-run of every post-scrape stage for the eval
   article (reset derived fields first), using the unmodified production
   stage adapters from `service/stages.py`. Stored scrape content is the input.

## Components

### 1. `src/ticker_news/evals/pipeline_eval.py` (new package `evals/`)

- `build_items(conn, ids) -> list[dict]` — loads `{article_id, url, ticker,
  published_utc, title}` for the requested articles; rejects articles whose
  `status != 'ok'` or that lack `published_utc`/content.
- `reset_article(conn, article_id)` — clears `embedding`, `category`,
  `category_reason`, `primary_ticker`, `primary_segment`, `more_tickers`,
  `more_segments`, `insights_extracted_at`; deletes the article's
  `article_insights` and `article_sentiment` rows. Single transaction.
- `ensure_eval_schema(conn)` — additive healing: `ADD COLUMN IF NOT EXISTS
  insights_extracted_at timestamptz`, `ADD COLUMN IF NOT EXISTS
  provider_sentiments jsonb`, `sentiment.store.ensure_schema` (creates
  `article_sentiment`).
- `run_pipeline_task` — the experiment task. Per item: open a fresh psycopg
  connection (sync connections are not shareable across workers), reset the
  article, then call `embed_stage`, `classify_stage`, `tag_stage` (with a
  `TagContext` loaded once per run from shared `ticker_data`),
  `insights_stage`, `sentiment_stage` in order. Returns
  `{action, confidence, category, ticker}` (action `None` if sentiment
  skipped, with the reason).
- `directional_agreement_evaluator` — fetches prices via
  `ticker_scan.load_prices`/`simulate` for `(ticker, published_utc)`;
  returns two `Evaluation`s:
  - `directional_agreement`: 1.0 / 0.0 / `None` per the scoring decision,
    comment explains the case (`buy, +2.3% -> 1`, `hold -> excluded`,
    `no verdict (category=recap/review)`, `no price data (<skip_reason>)`).
  - `price_move_pct`: signed percent move entry→exit (or `None` if no data).
- `avg_directional_agreement` — run-level evaluator: mean over non-`None`
  item scores, plus a count comment (`scored 4/5 items`).
- `run_eval(ids, dataset_name=None, dsn=None)` — orchestrator. Local-data
  mode by default (`langfuse.run_experiment`); with `dataset_name`, upserts
  the items into a Langfuse dataset (`create_dataset_item` keyed on
  article id) and uses `dataset.run_experiment` so runs are comparable in
  the Langfuse UI. `max_concurrency` small (2) — each task makes ~7 LLM calls.

### 2. CLI: `ticker-news eval pipeline`

New `eval` sub-app in `cli.py`:

```
ticker-news eval pipeline --ids 20491[,20512,...]
    [--dataset pipeline-e2e]   # also upsert items + record a dataset run
    [--dsn postgresql://...]   # override DATABASE_URL for the shared DB
```

Prints `result.format()` at the end. Requires `LANGFUSE_*` keys (hard error,
not silent no-op — an eval without Langfuse is pointless), `MASSIVE_API_KEY`,
`GOOGLE_API_KEY`, `OPENAI_API_KEY`.

### 3. Tracing

The experiment runner creates its own trace per item. The task does **not**
use `obs.article_trace` — its deterministic per-URL trace id would collide
across eval runs and break dataset-run linkage. Stage spans keep the
contract names (`embed`, `classify`, `tag`, `insights`, `sentiment`,
`analyst:<role>`, `synthesize`) via the existing `obs.stage_span` /
`obs.chain_config` helpers, which nest under the active OTel context.

## Error handling

- Stage raises → experiment runner isolates the failure to that item; the
  item shows as errored in Langfuse, others continue.
- Article re-classifies away from `real news` → sentiment skips →
  `directional_agreement = None` with comment (classification is
  non-deterministic; this is signal, not failure).
- Massive price data missing/holiday/halted → `None` + `skip_reason` comment.
- Tunnel down / DSN wrong → fail fast before any reset happens (connect +
  `ensure_eval_schema` run before the first task).

## Testing

- Offline unit tests (`tests/evals/test_pipeline_eval.py`, no markers):
  scoring matrix (buy/sell/hold × up/down/missing → value+comment),
  `build_items` filtering, `reset_article` SQL shape (mocked conn).
- The eval command itself is the integration path; no `integration`-marked
  pytest wrapper for now.
- Never touches `news_test` fixtures; the eval is not part of `pytest -m db`.

## Out of scope (YAGNI)

- Magnitude-weighted scoring (raw move is recorded; weighting can be derived
  later).
- Live re-scrape of article URLs (old links die; scrape stage is exercised
  by `backfill`, not this eval).
- CI/CD experiment gates (`langfuse/experiment-action`) — possible later on
  top of the same dataset.
- Multi-ticker verdicts (eval scores the primary ticker only, matching
  `sentiment_stage`).
