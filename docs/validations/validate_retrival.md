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

---

## 7. Results

Full run over every cluster that has a `later_news_article_ids` seed.

```
command : python scripts/validate_retreival/validate_retrieval.py \
            docs/validations/article_clusters.csv
params  : k=5  article_k=50  min_sim=0.7  months_before=3  exclusive=True  ground_truth=all
DB      : sharedproject (content-rich)
clusters: 14 evaluated · 6 skipped (securities-class-action lists — no news seed)
```

**Seeds used** (latest `later_news_article_ids` per cluster):

| Cluster | Seed | Seed published (UTC) |
|---|---|---|
| DeepSeek R1 launch & AI-stock selloff (Jan 2025) | a#3002 | 2025-02-06 09:42 |
| Stargate $500B AI infrastructure project (Jan 2025) | a#2919 | 2025-02-03 20:19 |
| Nvidia Q3 FY2025 earnings & reaction (Nov 2024) | a#765 | 2024-11-25 12:05 |
| Nvidia Q4 FY2025 earnings & reaction (Feb 2025) | a#3847 | 2025-03-03 17:32 |
| Nvidia Q1 FY2026 earnings (late May 2025) | a#6240 | 2025-05-29 18:08 |
| TSMC Arizona fab yields milestone | a#19665 | 2026-05-11 14:09 |
| Trump semiconductor / chip tariffs | a#8179 | 2025-08-07 11:29 |
| Trump 'reciprocal' tariffs market selloff (Apr 2025) | a#4862 | 2025-04-11 00:06 |
| Intel CEO Gelsinger departure (Dec 2024) | a#1489 | 2024-12-19 12:30 |
| Intel names Lip-Bu Tan CEO (Mar 2025) | a#4335 | 2025-03-19 09:24 |
| Nvidia H20 / China AI-chip export curbs | a#20122 | 2026-05-20 16:15 |
| Broadcom (AVGO) earnings & AI revenue outlook | a#4021 | 2025-03-07 17:27 |
| Palantir (PLTR) earnings & rally (Feb & May 2025) | a#5608 | 2025-05-06 18:10 |
| CoreWeave IPO (Mar-Apr 2025) | a#4690 | 2025-04-04 13:19 |

Skipped (no seed): Iris Energy (IREN), Wolfspeed (WOLF), ASML Holding, BigBear.ai (BBAI),
SoundHound AI (SOUN), GitLab (GTLB) — all securities-class-action lists.

### 7.1 Article level — insight-overlap retrieval

The pipeline's real article retriever: the distinct source articles of the matched
insights.

| Cluster | \|G\| | ret | TP | prec | recall | rec\* | acc | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DeepSeek R1 launch & AI-stock selloff | 112 | 12 | 6 | 50.0% | 5.4% | 5.6% | 95.8% | 9.7% |
| Stargate $500B AI infrastructure | 28 | 18 | 4 | 22.2% | 14.3% | 14.3% | 98.5% | 17.4% |
| Nvidia Q3 FY2025 earnings | 40 | 7 | 4 | 57.1% | 10.0% | 10.5% | 94.2% | 17.0% |
| Nvidia Q4 FY2025 earnings | 35 | 8 | 2 | 25.0% | 5.7% | 6.2% | 98.6% | 9.3% |
| Nvidia Q1 FY2026 earnings | 27 | 19 | 7 | 36.8% | 25.9% | 28.0% | 98.7% | 30.4% |
| TSMC Arizona fab yields | 3 | 29 | 0 | 0.0% | 0.0% | 0.0% | 99.2% | n/a |
| Trump semiconductor / chip tariffs | 9 | 10 | 0 | 0.0% | 0.0% | 0.0% | 99.2% | n/a |
| Trump 'reciprocal' tariffs selloff | 34 | 8 | 1 | 12.5% | 2.9% | 3.4% | 98.6% | 4.8% |
| Intel CEO Gelsinger departure | 9 | 2 | 1 | 50.0% | 11.1% | 11.1% | 99.3% | 18.2% |
| Intel names Lip-Bu Tan CEO | 6 | 10 | 4 | 40.0% | 66.7% | 80.0% | 99.7% | 50.0% |
| Nvidia H20 / China export curbs | 3 | 15 | 2 | 13.3% | 66.7% | 66.7% | 99.6% | 22.2% |
| Broadcom (AVGO) earnings & AI outlook | 16 | 8 | 3 | 37.5% | 18.8% | 20.0% | 99.4% | 25.0% |
| Palantir (PLTR) earnings & rally | 9 | 22 | 5 | 22.7% | 55.6% | 100.0% | 99.3% | 32.3% |
| CoreWeave IPO | 7 | 5 | 0 | 0.0% | 0.0% | 0.0% | 99.5% | n/a |
| **MICRO** (pooled) | **338** | **173** | **39** | **22.5%** | **11.5%** | **12.4%** | **98.8%** | **15.3%** |
| **MACRO** (cluster mean) | | | | **26.2%** | **20.2%** | **24.7%** | **98.6%** | **21.5%** |

### 7.2 Article level — whole-article retrieval

`similar_to(seed)` over `articles.embedding` (one vector per document, `article_k=50`).

| Cluster | \|G\| | ret | TP | prec | recall | rec\* | acc | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DeepSeek R1 launch & AI-stock selloff | 112 | 50 | 29 | 58.0% | 25.9% | 25.9% | 96.2% | 35.8% |
| Stargate $500B AI infrastructure | 28 | 50 | 4 | 8.0% | 14.3% | 14.3% | 97.5% | 10.3% |
| Nvidia Q3 FY2025 earnings | 40 | 25 | 11 | 44.0% | 27.5% | 27.5% | 93.7% | 33.8% |
| Nvidia Q4 FY2025 earnings | 35 | 50 | 11 | 22.0% | 31.4% | 31.4% | 97.8% | 25.9% |
| Nvidia Q1 FY2026 earnings | 27 | 50 | 8 | 16.0% | 29.6% | 29.6% | 97.5% | 20.8% |
| TSMC Arizona fab yields | 3 | 50 | 1 | 2.0% | 33.3% | 33.3% | 98.7% | 3.8% |
| Trump semiconductor / chip tariffs | 9 | 50 | 1 | 2.0% | 11.1% | 11.1% | 97.8% | 3.4% |
| Trump 'reciprocal' tariffs selloff | 34 | 50 | 0 | 0.0% | 0.0% | 0.0% | 96.9% | n/a |
| Intel CEO Gelsinger departure | 9 | 17 | 7 | 41.2% | 77.8% | 77.8% | 99.1% | 53.8% |
| Intel names Lip-Bu Tan CEO | 6 | 50 | 4 | 8.0% | 66.7% | 66.7% | 98.3% | 14.3% |
| Nvidia H20 / China export curbs | 3 | 50 | 2 | 4.0% | 66.7% | 66.7% | 98.8% | 7.5% |
| Broadcom (AVGO) earnings & AI outlook | 16 | 50 | 11 | 22.0% | 68.8% | 68.8% | 98.5% | 33.3% |
| Palantir (PLTR) earnings & rally | 9 | 50 | 4 | 8.0% | 44.4% | 66.7% | 98.1% | 13.6% |
| CoreWeave IPO | 7 | 10 | 0 | 0.0% | 0.0% | 0.0% | 99.4% | n/a |
| **MICRO** (pooled) | **338** | **602** | **93** | **15.4%** | **27.5%** | **27.8%** | **98.0%** | **19.8%** |
| **MACRO** (cluster mean) | | | | **16.8%** | **35.5%** | **37.1%** | **97.7%** | **21.4%** |

### 7.3 Insight level — `search_by_insights`

Relevant set = every embedded insight of the should-be-retrieved articles.

| Cluster | \|G\| | ret | TP | prec | recall | rec\* | acc | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DeepSeek R1 launch & AI-stock selloff | 544 | 18 | 7 | 38.9% | 1.3% | 1.3% | 95.3% | 2.5% |
| Stargate $500B AI infrastructure | 134 | 22 | 4 | 18.2% | 3.0% | 3.0% | 98.7% | 5.1% |
| Nvidia Q3 FY2025 earnings | 193 | 10 | 5 | 50.0% | 2.6% | 2.6% | 93.5% | 4.9% |
| Nvidia Q4 FY2025 earnings | 156 | 10 | 2 | 20.0% | 1.3% | 1.3% | 98.6% | 2.4% |
| Nvidia Q1 FY2026 earnings | 118 | 27 | 14 | 51.9% | 11.9% | 11.9% | 98.9% | 19.3% |
| TSMC Arizona fab yields | 20 | 39 | 0 | 0.0% | 0.0% | 0.0% | 99.7% | n/a |
| Trump semiconductor / chip tariffs | 44 | 20 | 0 | 0.0% | 0.0% | 0.0% | 99.4% | n/a |
| Trump 'reciprocal' tariffs selloff | 136 | 12 | 2 | 16.7% | 1.5% | 1.5% | 98.7% | 2.7% |
| Intel CEO Gelsinger departure | 42 | 7 | 1 | 14.3% | 2.4% | 2.4% | 99.2% | 4.1% |
| Intel names Lip-Bu Tan CEO | 23 | 18 | 8 | 44.4% | 34.8% | 34.8% | 99.8% | 39.0% |
| Nvidia H20 / China export curbs | 13 | 19 | 2 | 10.5% | 15.4% | 15.4% | 99.8% | 12.5% |
| Broadcom (AVGO) earnings & AI outlook | 73 | 15 | 5 | 33.3% | 6.8% | 6.8% | 99.3% | 11.4% |
| Palantir (PLTR) earnings & rally | 17 | 26 | 8 | 30.8% | 47.1% | 47.1% | 99.7% | 37.2% |
| CoreWeave IPO | 31 | 8 | 0 | 0.0% | 0.0% | 0.0% | 99.7% | n/a |
| **MICRO** (pooled) | **1544** | **251** | **58** | **23.1%** | **3.8%** | **3.8%** | **98.9%** | **6.5%** |
| **MACRO** (cluster mean) | | | | **23.5%** | **9.1%** | **9.1%** | **98.6%** | **12.8%** |

### 7.4 Summary

| Level | Micro precision | Micro recall | Micro recall\* | Micro F1 |
|---|--:|--:|--:|--:|
| Article — insight-overlap | 22.5% | 11.5% | 12.4% | 15.3% |
| Article — whole-article | 15.4% | 27.5% | 27.8% | 19.8% |
| Insight — `search_by_insights` | 23.1% | 3.8% | 3.8% | 6.5% |

**Takeaways at the default tight operating point (`k=5`, `min_sim=0.7`):**

1. **Whole-article casts the widest net.** It returns ~600 articles for 93 hits → highest
   recall (27.5%) but lowest precision (15.4%). Insight-overlap is the opposite: far fewer,
   cleaner returns (173 for 39 hits, 22.5% precision) — the right bias for feeding a
   sentiment prompt where noise hurts.
2. **`recall ≈ recall*` almost everywhere**, so the misses are **ranking/threshold**, not
   coverage — the relevant articles *were* indexed and in-window, the tight cut-off just
   didn't reach them. The one exception is Palantir (insight-overlap `recall 55.6%` vs
   `rec* 100%`): every *retrievable* relevant article was found; the rest of G had no
   insights to match on.
3. **Event shape drives the score.** Tight, well-defined events (Intel Lip-Bu Tan CEO,
   Nvidia H20 curbs, Palantir) score well; broad market-wide narratives (TSMC Arizona,
   Trump tariffs) score ~0 at this threshold — their cluster articles are only loosely
   similar to the seed and fall below `0.7` cosine.
4. **Insight-level recall is low by design** (an event has 100–500 labelled insights;
   matching a handful is enough to surface the right articles). Read it on **precision**
   (23%) — i.e. *are the matched insights on-topic* — which is in line with the
   article-level number.

> These are a single high-precision operating point. Raising `-k` and lowering
> `--min-similarity` trades precision for recall; re-run the sweep to draw the curve before
> tuning production retrieval. Machine-readable dumps: `--json` / `--csv-out`.
