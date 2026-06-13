# LangGraph & Langfuse Infrastructure

A plain-language guide to the two pieces of "AI infrastructure" in this project:
**LangGraph** runs the sentiment analyst panel, and **Langfuse** observes, versions,
and evaluates everything. Code references are relative to `src/ticker_news/`.

---

## 1. The big picture

Every article moves through the same stage chain:

```
scrape → embed → classify → tag → insights → sentiment
```

The first five stages are plain functions (`service/stages.py`). The last one,
**sentiment**, is where LangGraph comes in: a small graph that fans out three
LLM "analysts" in parallel and then synthesizes their takes into one
buy/sell/hold verdict.

Langfuse wraps the *whole* chain: one trace per article, a span per stage, a
nested generation per LLM call, versioned prompts, and an experiment runner
that scores verdicts against real price moves.

```
                ┌────────────────────────────── Langfuse trace: process-article ─┐
                │                                                                │
 article URL ──▶ scrape ─▶ embed ─▶ classify ─▶ tag ─▶ insights ─▶ sentiment     │
                │                                                  │             │
                │                                          ┌───────┴────────┐    │
                │                                          │ LangGraph      │    │
                │                                          │  ┌ fundamentals│    │
                │                                fan-out ──┼──┼ market_ctx  │    │
                │                                          │  └ hist_prec   │    │
                │                                          │       │        │    │
                │                                          │   synthesize   │    │
                │                                          │       │        │    │
                │                                          │    Verdict     │    │
                │                                          └────────────────┘    │
                └────────────────────────────────────────────────────────────────┘
```

---

## 2. LangGraph: the sentiment analyst panel

**Files:** `sentiment/graph.py`, `sentiment/analysts.py`, `sentiment/schemas.py`

### The graph shape

```
START ──(Send fan-out)──▶ analyst ×3 (parallel) ──▶ synthesize ──▶ END
```

There is **no supervisor** and no dynamic routing — the three roles are fixed
and always all run. The graph exists purely for the parallel fan-out and the
state merge.

- **Fan-out** (`graph.py: fan_out`): a conditional edge from `START` emits one
  `Send("analyst", {...})` task per role. LangGraph runs all Sends of a
  superstep concurrently on its background executor, so the three analysts run
  in parallel even though the node functions are sync.

- **State** (`SentimentState`): the key trick is the reducer on `analyses`:

  ```python
  analyses: Annotated[list[dict], operator.add]
  ```

  Three parallel analysts each return `{"analyses": [{"role", "analysis"}]}`;
  `operator.add` tells LangGraph to *concatenate* the lists instead of
  overwriting, so after the fan-out the state holds all three analyses.

- **Synthesize**: reads the accumulated analyses from state, renders the
  `synthesize-verdict` prompt, and calls a judge model with
  `.with_structured_output(Verdict)` so the answer is a validated pydantic
  object, not free text.

### The three analysts

Each gets the same article block (ticker, headline, body capped at 24k chars,
publish time, third-party provider sentiment) plus a role instruction
(`analysts.py: ANALYST_PROMPTS`):

| Role | Question it answers | Model |
|---|---|---|
| `fundamentals` | Does this change revenue / margins / guidance / competitive position? | gemini-2.5-flash-lite |
| `market_context` | What's already priced in? Does the headline over/understate the substance? | gemini-2.5-flash-lite |
| `historical_precedent` | Is this a recurring news pattern or genuinely new? (gets RAG context, see below) | gemini-2.5-flash-lite |

The `historical_precedent` analyst is the only retrieval-augmented one: before
the graph runs, `service/stages.py: similar_past_articles()` fetches the 5
cosine-nearest **earlier** articles via pgvector (HNSW index, two-query pattern
so the index is actually used) and injects them as a precedent list. The
time filter (`published_utc <` the article's own date) prevents lookahead.

### The judge

The **synthesize** node uses gemini-2.5-flash (one tier up from the analysts)
and must produce a `Verdict`:

```python
class Verdict(BaseModel):
    action: Literal["buy", "sell", "hold"]
    confidence: float        # in [0, 1]
    reasoning: str
```

Both LLMs have a 90s timeout and 3 retries with exponential jitter. The
compiled graph and its clients are `lru_cache`d per process.

### Where verdicts go

`sentiment_stage()` only judges articles that are `category = 'real news'`
with a tagged `primary_ticker` and non-empty content, and skips if a verdict
already exists. The verdict plus the three raw analyst texts are stored in
`article_sentiment` (PK `article_id + ticker`, insert is `ON CONFLICT DO
NOTHING` — re-judging requires an explicit delete, which `ticker-news
sentiment --reprocess` does).

---

## 3. Langfuse: observability, prompts, evals

**Files:** `shared/observability.py`, `shared/prompts.py`, `evals/pipeline_eval.py`

### Kill switch first

If `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are absent, **every helper
no-ops**: no traces, no prompt fetches, no crashes. The pipeline runs fully
offline from Langfuse. Tests rely on this — `tests/conftest.py` strips the
keys so test runs never export traces.

### Tracing: one trace per article

`observability.article_trace(url)` opens the root span and — the important
design choice — derives the **trace id deterministically from the URL**
(`Langfuse.create_trace_id(seed=url)`). Consequences:

- Re-running an article (service retry, batch re-judge, eval) lands in the
  **same trace**, so the full history of one article lives in one place.
- The `entrypoint` metadata (`service` / `batch` / `eval`) tells the runs apart.

Inside the root, each stage wraps itself in `stage_span(name)`, and LLM calls
nest automatically via the LangChain `CallbackHandler` that
`obs.chain_config()` injects per invocation (a fresh handler each time —
the handler is not thread-safe, never share one).

**The observation names are a contract.** Evals select observations by name,
so these must never be renamed:

```
process-article (root)
├── scrape, embed, classify, tag, insights
└── sentiment
    ├── analyst:fundamentals
    ├── analyst:market_context
    ├── analyst:historical_precedent
    └── synthesize
```

The root output carries `category` + `verdict`; trace metadata carries
`prompt_versions` (see below) and `entrypoint`.

Known gap: OpenAI embedding calls bypass the LangChain callback, so embedding
cost is not traced.

### Prompt management: in-repo source of truth, Langfuse for versions

`shared/prompts.py` implements a simple two-tier scheme:

1. **In-repo templates are the source of truth** — the literals in
   `sentiment/analysts.py`, `classification/chain.py`, `enrichment/insights.py`.
2. `ticker-news prompts push` upserts them all to Langfuse under the
   **`production`** label (names: `analyst-fundamentals`, `analyst-market_context`,
   `analyst-historical_precedent`, `synthesize-verdict`, `classify-article`,
   `extract-insights`).
3. At runtime, `get_prompt(name, fallback)` prefers the Langfuse copy and
   silently falls back to the in-repo text on any failure — the service boots
   fine with Langfuse down.

This means you can edit a prompt in the Langfuse UI and A/B it **without a
deploy**. Two safety nets back that up:

- `safe_format()` catches a remote template with typo'd placeholders and falls
  back to the in-repo version instead of crashing every article.
- Every prompt version actually fetched is recorded (`versions_seen()`) and
  attached to the trace metadata as `prompt_versions`, so each trace says
  exactly which prompt versions produced it.

Caching caveat: the sentiment renderers fetch per call (Langfuse's client-side
~60s TTL cache absorbs the cost), but chains that bake `get_prompt()` into an
`lru_cache`d constructor need a **process restart** to pick up prompt edits.

### Evals: the pipeline experiment

`ticker-news eval pipeline --ids N[,..]` (`evals/pipeline_eval.py`) is an
end-to-end Langfuse **experiment**:

1. **Task** — per article: wipe every derived field (embedding, category,
   tags, insights, verdict; scraped content stays), then re-run the real stage
   chain `embed → classify → tag → insights → sentiment` and return the verdict.
2. **Item evaluators** —
   - `directional_agreement`: buy + price up = 1, sell + price down = 1, wrong
     direction = 0. The realized move comes from Massive minute bars: entry at
     the first tradeable bar after publication, exit at that day's close.
   - `price_move_pct`: the raw entry→close move, recorded even when unscored.
   - Unscorable items (hold verdict, no verdict, no price data) become
     categorical `*_skip` scores rather than silently vanishing.
3. **Run evaluator** — `avg_directional_agreement` across all scorable items.

With `--dataset NAME` the article ids are upserted as Langfuse dataset items
(deterministic ids ⇒ idempotent) and the experiment runs over the **whole
dataset**, so the dataset grows into a regression suite and runs are
comparable in the Langfuse UI run-over-run. Because trace ids are
URL-seeded, eval re-runs also land in each article's existing trace.

---

## 4. Flow walkthrough (one article, end to end)

1. The feed (or a backfill CSV) enqueues a URL into `pipeline_jobs`; a worker
   claims it.
2. `article_trace(url)` opens (or re-enters) the article's deterministic trace.
3. Stages run in order, each inside its `stage_span`; after each, the job row's
   `stage` advances so a crash resumes mid-article.
4. At `sentiment`, the stage gathers the article + provider sentiment +
   pgvector precedents, then `judge_article()` invokes the LangGraph graph:
   three analysts in parallel → `operator.add` merges their analyses →
   `synthesize` returns a structured `Verdict`.
5. Every Gemini call appears in the trace under its stable name
   (`analyst:<role>`, `synthesize`) with the prompt versions in metadata.
6. The verdict + raw analyses are saved to `article_sentiment`; the root span
   gets the final output; `obs.flush()` ships the trace.
7. Later, `ticker-news eval pipeline` can replay the article and score the
   verdict against what the stock actually did — in the same trace.
