# Sentiment precedent-source options

The sentiment stage's **historical-precedent analyst** (one of the three fixed
roles in `sentiment/graph.py`) is grounded on prior corpus items so it can judge
how *distinctive vs. recurring* a piece of news is. How those priors are
retrieved is selectable via the `precedent_source` setting.

- **Config:** `SENTIMENT_PRECEDENT_SOURCE` (`shared/config.py`, default `article`)
- **Per-run override:** `ticker-news eval pipeline --precedent-source <value>`
- **Code:** `service/stages.py` — `gather_precedents()` dispatches on the value;
  `article_similarity()` / `insights_similarity()` do the retrieval;
  `own_article_insights()` supplies the target's own insights.

All modes share the same **precedent discipline**: sources must be earlier-published
(`published_utc < target`) and within the lookback window (`published_utc >= target
- precedent_lookback_days`), and the target's own rows are excluded — no look-ahead
leakage, since the verdict is later scored against the realized price move. The
`category = 'real news'` restriction is **optional** (off by default; see
`precedent_real_news_only` below).

The lookback window applies to **all** modes and is controlled by
`SENTIMENT_PRECEDENT_LOOKBACK_DAYS` (default `90`; `<= 0` means unlimited).

By default, precedents are **not** restricted by category — an insight is an
insight regardless of its source article's category, so recap/analysis/etc.
articles are eligible. Set `SENTIMENT_PRECEDENT_REAL_NEWS_ONLY=true` to restrict
candidates to `category = 'real news'` (the older, stricter behavior).

---

## The four options at a glance

| Option | Corpus searched | Query vector(s) | Result shape | Labels | DROP filter | "Own insights" section |
|---|---|---|---|---|---|---|
| `article` | `articles.embedding` | target article body embedding | top-`k` nearest **articles** (headlines, `k=5`) | — | — | no |
| `insights` | `article_insights` | each of the target's insight boxes | top-`limit` boxes, grouped by source article | — | — | yes (untagged) |
| `distilled-first` | `distilled_article_insights` | each of the target's distilled boxes | top-`limit` boxes, grouped by source article | `first_label` | no | yes (tagged) |
| `distilled-second` | `distilled_article_insights` | each of the target's distilled boxes | top-`limit` boxes, grouped by source article | `second_label` | excludes `DROP` | yes (tagged) |

Knobs shared by the three insight modes (`shared/config.py`):
`SENTIMENT_PRECEDENT_INSIGHTS_THRESHOLD` (similarity floor, default `0.7`) and
`SENTIMENT_PRECEDENT_INSIGHTS_LIMIT` (max boxes, default `60`). Two more apply to
**all four** modes (incl. `article`): `SENTIMENT_PRECEDENT_LOOKBACK_DAYS`
(default `90`) and `SENTIMENT_PRECEDENT_REAL_NEWS_ONLY` (default `false` — all
categories eligible).

---

## 1. `article` (default, legacy)

Top-k cosine-nearest **prior articles** to the target article's body embedding.

- One ANN query over `articles.embedding` (HNSW), `LIMIT 5`.
- The analyst sees five `date [ticker] headline` lines — titles only, no excerpts.
- No "THIS ARTICLE'S INSIGHTS" section.

Cheapest and oldest behavior; good coarse "have we seen this kind of story
before" signal, but no insight-level granularity.

## 2. `insights`

Insight-box level retrieval over `public.article_insights`.

- For **each** embedded insight box of the target article, one ANN query finds
  earlier insight boxes with cosine similarity `> threshold`.
- Hits are unioned across the target's boxes (dedup on insight id, max similarity
  wins), capped at `limit`, then **grouped by source article**: each entry is one
  prior article whose matching insight excerpt(s) are nested beneath it (a single
  match is inlined after an em dash).
- Adds a **THIS ARTICLE'S INSIGHTS** section: the target's own distilled insights,
  so the analyst can compare overlap directly.
- `article_insights` has no classification columns, so excerpts are **untagged**.

Finer-grained than `article`: matches on the actual claims, not the whole story.

## 3. `distilled-first`

Identical flow to `insights` but over `public.distilled_article_insights`, with
each excerpt **tagged by its `first_label`** classification.

- Same per-box ANN search, grouping, and own-insights section.
- Each excerpt is prefixed `[<first_label>]` (e.g. `[evidance-event]`,
  `[informative]`).
- `first_label` has no `DROP` values in the data, so nothing is filtered.

## 4. `distilled-second`

Same corpus as `distilled-first`, but **tagged by `second_label`** and with
**`DROP`-labelled insights excluded** (`WHERE second_label IS DISTINCT FROM 'DROP'`,
which keeps NULL/unlabelled rows).

- Excerpts prefixed `[<second_label>]`.
- Low-signal `DROP` boxes are dropped from both the precedents and the
  own-insights section, so it surfaces fewer, higher-signal items than
  `distilled-first`.

Because `first_label` and `second_label` are two independent classification
passes, the **same** insight can be tagged differently between the two distilled
modes (e.g. a "planning 1nm chips" insight may be `[evidance-event]` under
`first` but `[informative]` under `second`).

---

## The classification labels (distilled modes only)

The split is about **certainty / closure**, not topic:

- **`evidance-event`** — it definitively HAPPENED or was reported: a settled,
  quotable fact (reported earnings/guidance/margins, completed M&A, a signed deal
  with terms, a suit filed, a recall, layoffs, a regulatory ruling, a macro
  release, a disclosed position, a real price move on news). Bullish or bearish,
  but settled.
- **`informative`** — it was ANNOUNCED but isn't settled yet: real but
  soft/open/unverified (an MoU/LoI/proposed deal with no terms, a product launch,
  a funding round, a planned/future action, a self-reported milestone not
  independently confirmed). In motion, but terms/close/materiality unsettled.
- **`DROP`** — low-signal; excluded in `distilled-second`.

The analyst prompt carries this legend **only in the distilled modes** (where
excerpts are actually tagged) and is told to weight settled `evidance-event`
overlaps more heavily than soft `informative` ones. Under `article` / `insights`
the legend is omitted, so the analyst never sees a key for tags it won't
encounter.

---

## Usage

```bash
# pick the flow for an eval run (recorded as metadata.precedent_source)
ticker-news eval pipeline --ids 20310 --precedent-source distilled-second

# four-way A/B over one dataset, holding upstream stages fixed
for src in article insights distilled-first distilled-second; do
  ticker-news eval pipeline --dataset prec-ab --run-name "$src" \
    --precedent-source "$src" --skip-stages all
done
```

Set the default for the live service via `.env`:

```
SENTIMENT_PRECEDENT_SOURCE=distilled-second
SENTIMENT_PRECEDENT_REAL_NEWS_ONLY=false
SENTIMENT_PRECEDENT_LOOKBACK_DAYS=90
SENTIMENT_PRECEDENT_INSIGHTS_LIMIT=60
```

A worked example of the four rendered prompts for one article lives at
`docs/retreival_way_four_prompts_example.txt`.

### Checking retrieval quality

`ticker_news.evals.cluster_eval` measures recall against a labelled cluster CSV
(`docs/article_clusters_revisited.csv`): it queries backward from each cluster's
`later` article and reports how many earlier cluster members each mode surfaces.

```bash
python -m ticker_news.evals.cluster_eval docs/article_clusters_revisited.csv
```

It honors `real_news_only`; with it off, all clusters are testable (the expected
set is each cluster's any-category members within the window).

## Caveats

- **Distilled coverage:** `distilled_article_insights` covers ~15.8k articles. An
  article with no distilled boxes yields empty precedents under the distilled
  modes (graceful — the analyst is told "(none found)"), and — importantly — a
  *query* article with no distilled boxes retrieves nothing in distilled mode
  (the cluster check's DeepSeek miss is this, not a ranking failure).
- **Prompt publishing:** the in-repo prompt template is the source of truth; the
  Langfuse `production` copy is used at runtime only after `ticker-news prompts
  push` (and a process restart, since prompt chains are `lru_cache`d).
