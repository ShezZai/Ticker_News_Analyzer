# Retrieval Validation

How we measure whether the RAG retrieval surfaces the **right** prior coverage for an
event — quantified as **precision, recall, and accuracy** at two granularities
(**article** and **insight**), against a hand-labelled ground truth.

- **Ground truth:** `article_clusters.csv` (repo root).
- **Script:** `scripts/validate_retreival/validate_retrieval.py`.
- **What's tested:** the same no-lookahead, insight-level retrieval the
  [sentiment flow](../flows.md#3-sentiment-querying-flow) relies on, plus the
  whole-article retriever for comparison.

---

## 1. The ground-truth CSV

Each row is one real-world **event** (cluster). Columns:

| Column | Meaning |
|---|---|
| `cluster theme` | The event name (e.g. *"Nvidia Q4 FY2025 earnings & reaction"*). |
| `article ids` | Every article in the cluster. |
| `news_articles_ids` | The "news" subset of the cluster. |
| `later_news_article_ids` | The most-recent news article(s) of the event — the **query seed**. We anchor retrieval on the **latest-published** of these. |
| `all_less_than_3_months_before_later` | Every cluster article published in the **3 months before** the seed → the **article-level ground truth**: what retrieval, run from the seed with a 3-month no-lookahead window, *should* surface. |
| `news_less_than_3_months_before_later` | The news subset of the above (used when `--ground-truth news`). |

Clusters with no `later_news_article_ids` (e.g. the securities-class-action lists) have no
seed and are **skipped** — there is nothing to query *from*.

> The `_less_than_3_months_before_later` columns are built for a **3-month** window. Run
> the validator with `--months-before 3` (the default) so the retrieval window matches the
> labelled set.

---

## 2. Methodology

For each usable cluster:

1. **Seed** = the latest-published article in `later_news_article_ids`.
2. **Window** = `[seed_pub − months_before, seed_pub)` via `_seed_window(...)` — the exact
   same no-lookahead bounds used in production (`--exclusive` anchors strictly on the
   seed's timestamp, so nothing at/after the seed leaks in).
3. **Relevant set G** = the chosen ground-truth column, **minus the seed/later ids
   themselves**.
4. **Run retrieval** anchored on the seed, over the window, at both levels:

   | Level | Retriever | Retrieved set R |
   |---|---|---|
   | **Insight** | `search_by_insights(seed, k, …)` — top-`k` similar insights per seed insight, excluding the seed's own | the matched **insight ids** |
   | **Article — insight-overlap** | the distinct **source articles** of those insight hits (optionally capped/ranked by `related_articles`) | the matched **article ids** |
   | **Article — whole-article** | `similar_to(seed, k=article_k, …)` over `articles.embedding` | the matched **article ids** |

5. **Insight-level ground truth** = every embedded insight belonging to a ground-truth
   article ("*from the should-be-retrieved articles*").

---

## 3. Metric definitions

For a level with relevant set **G**, retrieved set **R**, and a retrievable universe
**U** (every item that *could* be returned — i.e. in the window and carrying the needed
embedding/insight):

```
TP = |R ∩ G|        FP = |R \ G|        FN = |(G∩U) \ R|        TN = |U| − |R ∪ (G∩U)|

precision  = TP / |R|
recall     = TP / |G|          (raw — vs the full labelled ground truth)
recall*    = TP / |G ∩ U|      (achievable — vs the ground truth that is even retrievable)
accuracy   = (TP + TN) / |U|
F1         = 2·precision·recall / (precision + recall)
```

- **`recall` vs `recall*`** — some labelled articles can't be retrieved at all (no
  embedding, or no extracted insights, or just outside the window). `recall` is the honest
  end-to-end number; `recall*` isolates the **ranking** quality by only counting ground
  truth that the index could have returned. A large gap means coverage (ingestion/insight
  extraction), not ranking, is the bottleneck.
- **`accuracy`** is reported for completeness but is **dominated by true negatives** (the
  window holds hundreds–thousands of irrelevant articles), so it sits near 99% for every
  cluster. **Precision, recall, and F1 are the discriminating metrics.**

### Universes (per cluster, computed from the DB over the window)

| Level | U = |
|---|---|
| Article — insight-overlap | articles in-window with ≥1 embedded insight |
| Article — whole-article | embedded articles in-window |
| Insight | embedded insights in-window |

### Aggregation

Reported two ways across clusters:

- **Micro** — pool `TP/FP/FN/TN` over all clusters, then compute the metrics. Weights
  events by size; the headline number.
- **Macro** — average the per-cluster metrics. Weights every event equally, so a tiny
  cluster counts as much as a huge one.

---

## 4. Running it

```bash
export NEWS_DB_DSN="postgresql://…@localhost:15432/sharedproject"   # content-rich DB
# (also needs OPENAI_API_KEY for the query embedding)

python scripts/validate_retreival/validate_retrieval.py article_clusters.csv
```

### Options

| Flag | Default | Effect |
|---|---|---|
| `-k, --k` | `5` | top-`k` similar insights **per seed insight** (the insight-level operating point). |
| `--article-k` | `50` | flat top-`k` for the whole-article retriever. |
| `--min-similarity` | `0.7` | cosine floor; raise for precision, lower for recall. |
| `--months-before` | `3` | no-lookahead window length (keep at 3 to match the CSV). |
| `--exclusive / --no-exclusive` | on | strict pre-seed window on the exact timestamp. |
| `--ground-truth {all,news}` | `all` | which CSV column is the relevant set. |
| `--top-articles N` | all | cap the insight-overlap article set to the top-N consolidated by `related_articles`. |
| `--json PATH` | — | machine-readable report (per-cluster + micro/macro aggregates). |
| `--csv-out PATH` | — | one row per (cluster × level). |

### Output

A table per level (insight-overlap article, whole-article, insight), one line per cluster
plus **MICRO** and **MACRO** summary rows:

```
  event                                           |G|  ret   TP   prec  recall   rec*    acc     F1
  Intel names Lip-Bu Tan CEO (Mar 2025)             6   10    4  40.0%  66.7%  80.0%  99.7%  50.0%
  …
  MICRO (pooled TP/FP/FN)                         338  173   39  22.5%  11.5%  12.4%  98.8%  15.3%
  MACRO (mean of clusters)                                       26.2%  20.2%  24.7%  98.6%  21.5%
```

---

## 5. Reading the results & tuning

- **Precision↔recall is an operating point.** `-k` and `--min-similarity` move it: a higher
  `k` / lower floor lifts recall and drops precision. Sweep them to draw the curve; the
  defaults (`k=5`, `min_sim=0.7`) are a deliberately tight, high-precision point, so raw
  recall reads low.
- **Watch the `recall` vs `recall*` gap.** When they're close, almost all relevant articles
  *were* retrievable and the number reflects ranking. When `recall*` ≫ `recall`, the misses
  are coverage gaps — those ground-truth articles lack embeddings or insights, or fall
  outside the window — fix ingestion, not the ranker.
- **Insight-level recall is naturally lower than article-level.** An event can have 500+
  labelled insights; matching a handful of them is often enough to surface the right
  articles (that's why article-level recall on the same event is much higher). Judge the
  insight level on **precision** (are the matched insights actually on-topic?) and on
  whether enough are found to drive the article consolidation.
- **Whole-article vs insight-overlap.** Whole-article tends to higher recall (one vector
  per doc casts a wide net) but lower precision; insight-overlap is tighter and is what the
  sentiment pipeline uses. Comparing the two columns shows the trade you're making.

## 6. Caveats

- **Accuracy is near-constant (~99%)** by construction — ignore it for ranking comparisons.
- **Ground truth is human-labelled** and event-scoped; an article that's topically relevant
  but not tagged into the cluster counts as a false positive, so precision is a *lower*
  bound on real usefulness.
- **One seed per cluster.** Only the latest `later_news_article_ids` article is used as the
  query; results describe "retrieve the event's history from its most recent article."
- **Window must match the CSV.** Changing `--months-before` away from 3 desyncs the
  retrieval window from the labelled `_less_than_3_months_before_later` sets.
