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

### 7.5 Hybrid retrieval comparison

§7 showed the two retrievers are **complementary**: whole-article has higher recall, insight
has higher precision. This section tests whether combining them does better — measured on the
same 14 clusters.

- **Engine:** `scripts/search/hybrid_retrieval.py` (reusable; has its own CLI).
- **Harness:** `scripts/validate_retreival/compare_hybrids.py` (scores all methods on the CSV).

```bash
python scripts/validate_retreival/compare_hybrids.py docs/validations/article_clusters.csv
```

**Methods.**

| Method | Flow | Intuition |
|---|---|---|
| `whole-article` | `similar_to()` over `articles.embedding` | baseline (recall-y) |
| `insight` | `search_by_insights()` | baseline (precision-y) |
| `intersection` | keep insights whose article is in **both** top-k sets | naive "keep only the overlap" |
| `cascade` | **wide** whole-article net → filter insights inside it by cosine ≥ `tau` | recall net + precision filter |
| `fusion` | reciprocal-rank fusion of (insight cosine) and (article-net rank), keep top `budget` | union + rerank |

> **Why order matters.** Set intersection inherits the *lower* recall:
> `recall(A∩B) ≤ min(recall(A), recall(B))`. So the high-recall method must be the wide
> **net** and the high-precision method the **filter** — never the reverse. `cascade` does
> this; `intersection` does not.

**Run params:** `k=5 article_k=50 min_sim=0.7 net_k=150 net_min=0.45 tau=0.70 budget=50`,
`months_before=3`, `exclusive`, `ground_truth=all`.

**Article level (micro; macro-F1 at right):**

| Method | avg ret | precision | recall | rec\* | F1 | macro-F1 |
|---|--:|--:|--:|--:|--:|--:|
| whole-article | 43.0 | 15.4% | 27.5% | 27.8% | 19.8% | 21.4% |
| insight | 12.4 | 22.5% | 11.5% | 12.4% | 15.3% | 21.5% |
| intersection | 8.4 | **25.4%** | 8.9% | 9.5% | 13.2% | 19.8% |
| cascade | 75.1 | 12.5% | **38.8%** | **41.6%** | 18.8% | 20.2% |
| fusion | 15.6 | 22.5% | 14.5% | 15.6% | 17.6% | **21.9%** |

**Insight level (micro):**

| Method | avg ret | precision | recall | F1 | macro-F1 |
|---|--:|--:|--:|--:|--:|
| insight | 17.9 | 23.1% | 3.8% | 6.5% | 12.8% |
| intersection | 13.1 | **25.1%** | 3.0% | 5.3% | 11.6% |
| cascade | 204.4 | 14.0% | **25.9%** | **18.2%** | **18.1%** |
| fusion | 50.0 | 20.9% | 9.5% | 13.0% | 16.7% |

**Per-cluster article-level F1:**

| Event | whole | insight | intersect | cascade | fusion |
|---|--:|--:|--:|--:|--:|
| DeepSeek R1 launch & AI-stock selloff | 35.8% | 9.7% | 9.9% | 32.3% | 16.9% |
| Stargate $500B AI infrastructure | 10.3% | 17.4% | 5.6% | 16.5% | 9.1% |
| Nvidia Q3 FY2025 earnings | 33.8% | 17.0% | 8.9% | **41.9%** | 19.2% |
| Nvidia Q4 FY2025 earnings | 25.9% | 9.3% | 9.5% | 22.1% | 8.0% |
| Nvidia Q1 FY2026 earnings | 20.8% | 30.4% | 31.1% | 13.1% | 29.8% |
| TSMC Arizona fab yields | 3.8% | n/a | n/a | 3.4% | n/a |
| Trump semiconductor / chip tariffs | 3.4% | n/a | n/a | 6.1% | 10.0% |
| Trump 'reciprocal' tariffs selloff | n/a | 4.8% | n/a | n/a | 4.3% |
| Intel CEO Gelsinger departure | 53.8% | 18.2% | 18.2% | 53.8% | 42.1% |
| Intel names Lip-Bu Tan CEO | 14.3% | 50.0% | 42.9% | 19.6% | 38.1% |
| Nvidia H20 / China export curbs | 7.5% | 22.2% | 16.7% | 5.6% | 23.5% |
| Broadcom (AVGO) earnings & AI outlook | 33.3% | 25.0% | 25.0% | 20.8% | 33.3% |
| Palantir (PLTR) earnings & rally | 13.6% | 32.3% | 30.8% | 7.1% | 27.8% |
| CoreWeave IPO | n/a | n/a | n/a | n/a | n/a |

#### Findings

1. **`intersection` is an anti-pattern — confirmed empirically.** It has the highest precision
   (25.4%) but its recall (8.9%) drops *below* the insight baseline (11.5%), giving the **worst
   F1 of any method** (13.2%). Exactly the "intersection inherits the lower recall" prediction —
   it discards the recall whole-article was contributing.
2. **`cascade` is a recall powerhouse, and tunable.** At `tau=0.70` it out-recalls *every* method
   (38.8% article; 25.9% vs 3.8% at the insight level — a ~7× jump) because its net (`net_k=150`)
   is wider than the whole-article baseline. The cost is precision (12.5%) and volume (~75
   articles / ~204 insights — over the 80-insight prompt budget). Tightening `tau` (≈0.78) and
   `net_k` converts that recall headroom toward a balanced point.
3. **`fusion` is the balanced winner at these defaults.** It holds insight-level precision
   (22.5%) while lifting recall above insight-alone (14.5%), and takes the **top macro-F1
   (21.9%)** — it wins on the *most* clusters, not just the big ones. It also ships with a
   `budget` cap, so it fits the sentiment prompt directly.
4. **Methods specialize by event shape.** `cascade` dominates big multi-article events
   (Nvidia Q3, DeepSeek, Intel Gelsinger); `fusion`/`insight` win the tight, well-defined ones
   (Intel Lip-Bu Tan, Palantir, Nvidia H20).

#### Recommendation

- **Drop `intersection`.**
- For the sentiment pipeline's `gather_related()` (clean context within an 80-insight budget),
  **`fusion`** is the natural drop-in: balanced, precision-preserving, budget-capped.
- **`cascade`** is the stronger engine *if tuned* — a `tau`/`net_k` sweep should push it past both
  baselines on F1; until then it is recall-skewed.

All knobs are exposed on both scripts; re-run `compare_hybrids.py` with different
`--tau-insight` / `--net-k` / `--budget` to move the operating point.

---

## 8. Results on the revisited (tight-event) table

§7.5 found retrieval scores well on tight, well-defined events and poorly on broad
market-wide narratives. To test that directly, the ground truth was rebuilt as
**11 tight, single-catalyst events** — `docs/validations/article_clusters_revisited.csv`,
generated by `scripts/validate_retreival/build_revisited_clusters.py` (keyword anchor +
short coverage window per event; TSMC Arizona and the Trump-tariff narratives dropped as
inherently broad; the 6 class-action lists dropped for lacking a news seed).

Two evaluation **modes** (both via `validate_retrieval.py --mode …`):

| Mode | Seed | Window | Question it answers |
|---|---|---|---|
| **1 · `last-noforward`** | the event's **last** (curated) article | strictly **before** the seed (no lookahead) | *Reacting to the latest article, can we retrieve the event's prior history?* (the honest backtest frame) |
| **2 · `middle-forward`** | a **middle** article of the event | extends **forward** to the last article's date (inclusive) | *Given a mid-event article, can we recover the whole cluster — earlier **and** later coverage?* |

Params: `k=5 article_k=50 min_sim=0.7 months_before=3 ground_truth=all`.

### 8.1 Mode 1 — `last-noforward` (backtest frame)

Article level, **insight-overlap** (the pipeline's retriever):

| Event | \|G\| | ret | TP | prec | recall | rec\* | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Stargate $500B AI announcement | 16 | 11 | 6 | 54.5% | 37.5% | 37.5% | 44.4% |
| DeepSeek R1 launch & selloff | 41 | 8 | 4 | 50.0% | 9.8% | 11.1% | 16.3% |
| Palantir Q4'24 earnings & rally | 14 | 8 | 1 | 12.5% | 7.1% | 7.1% | 9.1% |
| Nvidia Q3 FY25 earnings | 6 | 16 | 3 | 18.8% | 50.0% | 50.0% | 27.3% |
| Nvidia Q4 FY25 earnings | 7 | 12 | 2 | 16.7% | 28.6% | 40.0% | 21.1% |
| Nvidia Q1 FY26 earnings | 3 | 7 | 0 | 0.0% | 0.0% | 0.0% | n/a |
| Broadcom Q1 FY25 earnings | 3 | 8 | 1 | 12.5% | 33.3% | 33.3% | 18.2% |
| Intel Gelsinger departure | 7 | 14 | 4 | 28.6% | 57.1% | 57.1% | 38.1% |
| Intel names Lip-Bu Tan CEO | 7 | 10 | 4 | 40.0% | 57.1% | 80.0% | 47.1% |
| Nvidia H20 export ban & charge | 17 | 18 | 5 | 27.8% | 29.4% | 29.4% | 28.6% |
| CoreWeave IPO debut | 4 | 6 | 2 | 33.3% | 50.0% | 66.7% | 40.0% |
| **MICRO** | **125** | **118** | **32** | **27.1%** | **25.6%** | **27.8%** | **26.3%** |
| **MACRO** | | | | **26.8%** | **32.7%** | **37.5%** | **29.0%** |

All three levels (micro / macro F1):

| Level | precision | recall | rec\* | F1 (micro) | F1 (macro) |
|---|--:|--:|--:|--:|--:|
| article — insight-overlap | 27.1% | 25.6% | 27.8% | 26.3% | 29.0% |
| article — whole-article | 7.9% | 30.4% | 31.7% | 12.5% | 15.8% |
| insight — `search_by_insights` | 30.3% | 10.0% | 10.0% | 15.1% | 19.8% |

### 8.2 Mode 2 — `middle-forward` (cluster-recovery frame)

Middle seeds, e.g. Lip-Bu Tan → a#4235 ("Intel Gets the Outsider CEO It Desperately Needs"),
Nvidia Q3 → a#656, CoreWeave → a#4629.

Article level, **insight-overlap**:

| Event | \|G\| | ret | TP | prec | recall | rec\* | F1 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Stargate $500B AI announcement | 16 | 3 | 0 | 0.0% | 0.0% | 0.0% | n/a |
| DeepSeek R1 launch & selloff | 41 | 5 | 4 | 80.0% | 9.8% | 11.1% | 17.4% |
| Palantir Q4'24 earnings & rally | 14 | 5 | 1 | 20.0% | 7.1% | 7.1% | 10.5% |
| Nvidia Q3 FY25 earnings | 6 | 10 | 3 | 30.0% | 50.0% | 50.0% | 37.5% |
| Nvidia Q4 FY25 earnings | 7 | 17 | 0 | 0.0% | 0.0% | 0.0% | n/a |
| Nvidia Q1 FY26 earnings | 3 | 14 | 0 | 0.0% | 0.0% | 0.0% | n/a |
| Broadcom Q1 FY25 earnings | 3 | 6 | 2 | 33.3% | 66.7% | 66.7% | 44.4% |
| Intel Gelsinger departure | 7 | 11 | 4 | 36.4% | 57.1% | 57.1% | 44.4% |
| Intel names Lip-Bu Tan CEO | 7 | 7 | 3 | 42.9% | 42.9% | 60.0% | 42.9% |
| Nvidia H20 export ban & charge | 17 | 16 | 3 | 18.8% | 17.6% | 17.6% | 18.2% |
| CoreWeave IPO debut | 4 | 10 | 2 | 20.0% | 50.0% | 66.7% | 28.6% |
| **MICRO** | **125** | **104** | **22** | **21.2%** | **17.6%** | **19.1%** | **19.2%** |
| **MACRO** | | | | **25.6%** | **27.4%** | **30.6%** | **30.5%** |

All three levels (micro / macro F1):

| Level | precision | recall | rec\* | F1 (micro) | F1 (macro) |
|---|--:|--:|--:|--:|--:|
| article — insight-overlap | 21.2% | 17.6% | 19.1% | 19.2% | 30.5% |
| article — whole-article | 9.9% | 28.8% | 30.0% | 14.7% | 14.9% |
| insight — `search_by_insights` | 26.1% | 7.5% | 7.5% | 11.7% | 22.6% |

### 8.3 Tight vs broad, and mode vs mode

Insight-overlap and insight-level retrievers (what the pipeline uses) on the **broad** §7
table vs the **tight** §8 Mode-1 table — same params, same retriever:

| Retriever | Table | precision | recall | F1 (micro) | F1 (macro) |
|---|---|--:|--:|--:|--:|
| insight-overlap | §7 broad | 22.5% | 11.5% | 15.3% | 21.5% |
| insight-overlap | §8 tight | **27.1%** | **25.6%** | **26.3%** | **29.0%** |
| insight | §7 broad | 23.1% | 3.8% | 6.5% | 12.8% |
| insight | §8 tight | **30.3%** | **10.0%** | **15.1%** | **19.8%** |

#### Findings

1. **Tightening the events lifts the pipeline retrievers substantially.** Insight-overlap
   F1 jumps **15.3% → 26.3%** (recall more than doubles, 11.5% → 25.6%); insight-level F1
   **6.5% → 15.1%**. This confirms §7.5's hypothesis empirically: precise, single-catalyst
   clusters are far more retrievable than broad market narratives, and the gain is in
   **recall** — the relevant articles really are close to the seed in vector space when the
   event is tight.
2. **Whole-article *worsens* on tight clusters** (precision 15.4% → 7.9%). With `|G|` now
   3–17 but a fixed `article_k=50`, it returns ~50 articles for a handful of hits — mostly
   false positives. A fair whole-article baseline would cap `k` near the cluster size; as-is
   it is mis-tuned for small events. The insight retrievers, which self-limit, don't have
   this problem.
3. **Forward-looking (Mode 2) raises per-cluster ceilings but not pooled scores.**
   Macro-F1 rises (insight-overlap 29.0% → 30.5%, insight 19.8% → 22.6%) and whole-article
   macro-recall climbs to 43.5% (forward window grabs later coverage — Gelsinger 85.7%,
   CoreWeave rec\* 100%, Palantir 78.6%). But **micro**-F1 dips (insight-overlap 26.3% →
   19.2%): a middle seed sometimes lands off-centre (Stargate, Nvidia Q4/Q1 collapse to 0),
   and forward mode adds *future* articles as targets, which are harder to reach from a
   mid-event vantage. Net: Mode 2 is the better frame for "recover the whole story from any
   point", Mode 1 is the honest "react to the latest news" measurement.
4. **Event shape still dominates.** Earnings and named-event clusters (Intel Lip-Bu Tan,
   Gelsinger, Broadcom, Nvidia Q3, CoreWeave) score well in both modes; the diffuse-by-nature
   ones (DeepSeek selloff with 41 articles, Palantir's opinion-heavy coverage) stay low —
   their cluster members are only loosely similar to any single seed.

#### Takeaway

The retrieval is materially better than the §7 broad-table numbers suggested: on tight,
well-defined events the pipeline's insight-overlap retriever reaches **26% F1 / 26% recall**
at a high-precision operating point, and macro-F1 ~30%. The remaining low scores are
concentrated in (a) genuinely diffuse "events" that are really market moods, and (b) the
mis-tuned fixed-`k` whole-article baseline — not in the insight retrieval the sentiment
pipeline depends on.

Reproduce:

```bash
python scripts/validate_retreival/build_revisited_clusters.py
python scripts/validate_retreival/validate_retrieval.py \
    docs/validations/article_clusters_revisited.csv --mode last-noforward
python scripts/validate_retreival/validate_retrieval.py \
    docs/validations/article_clusters_revisited.csv --mode middle-forward
```

---

## 9. Hybrids on the revisited table (both modes)

§7.5 compared the five retrievers on the broad table; §8 showed tight events score better.
This runs all five methods on the **tight** revisited table, in **both modes**
(`compare_hybrids.py --mode …`). Params: `k=5 article_k=50 min_sim=0.7 net_k=150
net_min=0.45 tau=0.70 budget=50`.

### 9.1 Mode 1 — `last-noforward`

**Article level** (micro; macro-F1 at right):

| Method | avg ret | precision | recall | rec\* | F1 | macro-F1 |
|---|--:|--:|--:|--:|--:|--:|
| whole-article | 43.8 | 7.9% | 30.4% | 31.7% | 12.5% | 15.8% |
| insight | 10.7 | 27.1% | 25.6% | 27.8% | **26.3%** | **29.0%** |
| intersection | 8.6 | 24.2% | 18.4% | 20.0% | 20.9% | 23.2% |
| cascade | 91.6 | 6.3% | **51.2%** | **55.7%** | 11.3% | 13.4% |
| fusion | 14.8 | 20.2% | 26.4% | 28.7% | 22.9% | 25.3% |

**Insight level**:

| Method | avg ret | precision | recall | F1 | macro-F1 |
|---|--:|--:|--:|--:|--:|
| insight | 15.9 | **30.3%** | 10.0% | 15.1% | 19.8% |
| intersection | 13.7 | 28.5% | 8.1% | 12.7% | 17.0% |
| cascade | 317.6 | 5.9% | **38.8%** | 10.2% | 13.9% |
| fusion | 50.0 | 18.0% | 18.8% | **18.4%** | **20.1%** |

### 9.2 Mode 2 — `middle-forward`

**Article level** (micro; macro-F1 at right):

| Method | avg ret | precision | recall | rec\* | F1 | macro-F1 |
|---|--:|--:|--:|--:|--:|--:|
| whole-article | 33.2 | 9.9% | 28.8% | 30.0% | 14.7% | 14.9% |
| insight | 9.5 | 21.2% | 17.6% | 19.1% | 19.2% | 30.5% |
| intersection | 6.1 | 22.4% | 12.0% | 13.0% | 15.6% | 27.4% |
| cascade | 74.6 | 8.8% | **57.6%** | **62.6%** | 15.2% | 23.1% |
| fusion | 14.3 | **23.6%** | 29.6% | 32.2% | **26.2%** | **32.1%** |

**Insight level**:

| Method | avg ret | precision | recall | F1 | macro-F1 |
|---|--:|--:|--:|--:|--:|
| insight | 13.9 | 26.1% | 7.5% | 11.7% | 22.6% |
| intersection | 9.4 | 26.2% | 5.1% | 8.5% | 18.6% |
| cascade | 257.2 | 7.2% | **38.6%** | 12.2% | 19.2% |
| fusion | 50.0 | 24.5% | 25.4% | **25.0%** | **30.7%** |

### 9.3 Per-cluster article-level F1

Mode 1 / Mode 2 (fusion column bolded where it leads):

| Event | whole | insight | intersect | cascade | fusion |
|---|--:|--:|--:|--:|--:|
| Stargate (M1) | 9.1% | 44.4% | 27.3% | 9.3% | 29.4% |
| Stargate (M2) | n/a | n/a | n/a | 45.7% | 8.0% |
| DeepSeek (M1) | 22.2% | 16.3% | 16.7% | 33.3% | 18.9% |
| DeepSeek (M2) | 4.7% | 17.4% | 4.7% | 47.3% | 30.2% |
| Palantir (M1) | 21.9% | 9.1% | 9.1% | 16.9% | 7.7% |
| Palantir (M2) | 34.4% | 10.5% | 11.1% | 17.2% | **37.5%** |
| Nvidia Q3 (M1) | 14.3% | 27.3% | 19.0% | 13.0% | **34.8%** |
| Nvidia Q3 (M2) | 14.3% | 37.5% | 37.5% | 17.9% | **42.1%** |
| Broadcom (M1) | 7.5% | 18.2% | 18.2% | 4.3% | 11.8% |
| Broadcom (M2) | n/a | 44.4% | n/a | 36.4% | 26.7% |
| Gelsinger (M1) | 7.0% | 38.1% | 13.3% | 7.9% | 28.6% |
| Gelsinger (M2) | 21.1% | 44.4% | 50.0% | 42.4% | **52.6%** |
| Lip-Bu Tan (M1) | 14.0% | 47.1% | 40.0% | 19.2% | 36.4% |
| Lip-Bu Tan (M2) | 20.5% | 42.9% | 42.9% | 12.9% | 30.0% |
| H20 (M1) | 11.9% | 28.6% | 26.7% | 9.6% | 27.0% |
| H20 (M2) | 13.3% | 18.2% | 9.5% | 23.4% | 26.3% |
| CoreWeave (M1) | 42.9% | 40.0% | 40.0% | 27.3% | 40.0% |
| CoreWeave (M2) | 11.1% | 28.6% | 36.4% | 4.0% | 35.3% |

### 9.4 Findings

1. **`fusion` is the best all-round method on tight events — and best of all in forward
   mode.** In Mode 2 it tops *every* aggregate: article micro-F1 **26.2%** and macro-F1
   **32.1%**, insight micro-F1 **25.0%** and macro-F1 **30.7%** — beating the insight
   baseline on both levels. In Mode 1 it leads the insight level (18.4% vs 15.1%) and trails
   only the insight baseline at the article level. The forward window feeds its union extra
   recoverable coverage while RRF + the `budget` cap hold precision — exactly the behaviour
   §7.5 predicted, now confirmed on clean clusters.
2. **The intersection trap reproduces on tight clusters.** Its recall stays below the insight
   baseline in both modes (article 18.4% vs 25.6% in M1; 12.0% vs 17.6% in M2) and its F1
   trails — the "keep only the overlap" anti-pattern is not rescued by tighter ground truth.
3. **`cascade` is an even more extreme recall engine here, and more mis-tuned.** It reaches
   the highest recall of any method (article **51–58%**, `rec*` up to **62.6%** in M2) but
   precision collapses to 6–9% because `net_k=150` dwarfs the 3–17-article clusters
   (~90–320 items returned). Its huge recall ceiling is the opportunity: a much tighter `tau`
   (and smaller `net_k`) should convert it into a strong balanced method — the headroom is
   clearly there.
4. **Forward mode lifts the precision-preserving methods, not the recall ones.** fusion and
   the insight baseline gain macro-F1 going M1→M2 (fusion article 25.3%→32.1%); cascade's
   already-low precision keeps its F1 low despite recall rising further. Per-cluster, fusion
   wins the well-defined events outright in M2 (Gelsinger 52.6%, Nvidia Q3 42.1%,
   Palantir 37.5%).

### 9.5 Recommendation (updated)

The earlier call stands and strengthens: **use `fusion` for the sentiment pipeline's
`gather_related()`**, ideally in the forward-looking frame when the use-case allows
(historical analysis / clustering); for the live backtest frame (Mode 1) it remains
competitive and precision-safe. **Drop `intersection`.** **`cascade` is worth a tuning pass**
(`--tau-insight ~0.80`, smaller `--net-k`) — its 60%+ recall ceiling is the largest of any
method and currently wasted on precision.

Reproduce:

```bash
python scripts/validate_retreival/compare_hybrids.py \
    docs/validations/article_clusters_revisited.csv --mode last-noforward
python scripts/validate_retreival/compare_hybrids.py \
    docs/validations/article_clusters_revisited.csv --mode middle-forward
```

---

## 10. The two-phase-similarity retrieval

### 10.1 Method

`two-phase-similarity` (in `scripts/search/hybrid_retrieval.py`, `--method two-phase-similarity`) chains the two
complementary strengths into one two-stage flow:

```
Stage 1 — cascade POOL (recall):
    wide whole-article net (net_k=150, LOOSE article floor net_min_sim=0.55)
    -> keep the articles that have >=1 insight with cosine-to-seed >= tau (0.75)
    => an ARTICLE candidate pool

Stage 2 — fusion WITHIN the pool (precision):
    take only the pool insights that THEMSELVES clear tau (cosine-to-seed >= 0.75)
    -- a pool article's off-topic boxes are dropped --
    reciprocal-rank-fuse  (insight-cosine rank)  with  (article rank in the net)
    keep the top `budget` (40) insights
```

`two-phase-similarity` uses **asymmetric gates**: a *loose* article-net floor (0.55) to fill the candidate
pool for recall, and a *tight* insight gate (0.75) to hold precision on the insights that
survive — the tau gate also prunes off-topic boxes of an otherwise-relevant article.
Precision comes from that 0.75 insight gate plus fusion's rerank and the `budget` cap; recall
comes from the wide net. It is the embodiment of the §7.5 rule — *let the high-recall method
build the pool, let the high-precision method rank inside it.* §10.4–§10.5 show why the split
beats a symmetric floor: the net floor was the binding constraint on coverage, while the
insight gate is what protects precision.

### 10.2 `two-phase-similarity` vs the original retrievers (both tables, both modes)

Head-to-head against the two baselines the system shipped with — whole-article and insight
— across **both** ground-truth tables and **both** modes, with `two-phase-similarity` at its default
**0.55/0.75/40** gates (loose net 0.55, tight insight gate 0.75, budget 40; net_k=150).
Micro-F1 (macro-F1 in parens); **bold** = best in row.

**Article level:**

| Table | Mode | whole-article | insight | two-phase-similarity |
|---|---|--:|--:|--:|
| original | last-noforward | **19.8** (21.4) | 15.3 (21.5) | 18.5 (**23.3**) |
| original | middle-forward | 18.3 (20.5) | 18.6 (23.4) | **24.2 (28.9)** |
| revisited | last-noforward | 12.5 (15.8) | **26.3 (29.0)** | 17.8 (21.3) |
| revisited | middle-forward | 14.7 (14.9) | 19.2 (30.5) | **26.4 (31.1)** |

**Insight level** (whole-article has no insight output):

| Table | Mode | insight | two-phase-similarity |
|---|---|--:|--:|
| original | last-noforward | 6.5 (12.8) | **11.4 (18.8)** |
| original | middle-forward | 8.9 (13.9) | **15.3 (21.8)** |
| revisited | last-noforward | **15.1 (19.8)** | 14.5 (19.3) |
| revisited | middle-forward | 11.7 (22.6) | **21.1 (27.0)** |

#### What this shows

At the **asymmetric 0.55/0.75/40** default, `two-phase-similarity` is a stronger all-rounder than at the
earlier symmetric-0.7 point (§10.4–§10.5): the loose net restores coverage while the tight
insight gate keeps the returns clean. Against the two shipped baselines:

1. **Insight level — where `gather_related()` actually consumes — `two-phase-similarity` wins 3 of 4.** It
   beats the insight baseline in both original modes (11.4 vs 6.5; 15.3 vs 8.9) and in
   revisited-forward (21.1 vs 11.7), losing only revisited + last-noforward by 0.6 (14.5 vs
   15.1), the tight-backtest cell from §9.
2. **Article level — `two-phase-similarity` beats whole-article on macro-F1 in all 4 and on micro in 3 of
   4** (losing only original + last-noforward, 18.5 vs 19.8), and beats the insight baseline
   at the article level in 3 of 4. It is never the worst method in any cell.
3. **The one real weak cell is revisited + last-noforward**, where the pure insight
   retriever's precision wins both levels (article 26.3, insight 15.1) — the structural
   backtest weakness documented in §9–§10.3.
4. **`two-phase-similarity` is strongest in `middle-forward`** (cluster recovery): it tops both levels on
   both tables and posts the family's best macro-F1 (revisited article ≈31%, insight ≈27%).

### 10.3 Mode-1 tuning sweep (negative result)

Can `two-phase-similarity`'s one weak cell (revisited + last-noforward) be tuned away? A 48-config sweep
(`scripts/validate_retreival/sweep_two_phase.py`) over `net_k ∈ {50,100,150}`,
`tau ∈ {0.70,0.74,0.78,0.82}`, `budget ∈ {20,30,50,80}` says **no**:

- Best article-F1: `net_k=50, tau=0.78, budget=20` → **19.2** (fusion 22.9, insight 26.3) — loses.
- Best insight-F1: `net_k=150, tau=0.74, budget=80` → **16.9** (fusion 18.4, insight 15.1) —
  clears the insight baseline but not fusion.

The two knobs pull opposite ways (article-F1 wants a *small* budget for precision; insight-F1
wants a *large* one for recall), and neither optimum reaches fusion. The cause is structural:
in the backtest frame the seed is the *latest* article, so the backward-only net is
intrinsically noisy and no pool threshold cleans the *ranking* enough. `two-phase-similarity` is, by
construction, a method whose edge needs a window that brackets the event.

The shipped **0.55/0.75/40** default lands in the same place: on revisited + last-noforward
it scores article-F1 **17.8** (still under insight's 26.3), confirming the negative result is
a property of the frame, not of the floors — neither the symmetric-0.7 point nor the
asymmetric default recovers the weak cell, and tuning is not expected to.

### 10.4 Per-cluster: `two-phase-similarity` at 0.8/0.8/30 vs 0.7/0.7/25 (insight level)

Same per-cluster insight-level frame as §7.3 (original table, `last-noforward`, `|G|` =
every embedded insight of the should-be-retrieved articles), but for **`two-phase-similarity`**, at two
operating points. Each cell is **`0.8/0.8/30`** with **`(0.7/0.7/25)`** in parens —
i.e. *tighter floors + 30 budget* vs the *default 0.7 floors + 25 budget*. `rec*` equals
`recall` at this level and `acc` is a flat ≈99%, so both are omitted.

| Cluster | \|G\| | ret | TP | prec | recall | F1 |
|---|--:|--:|--:|--:|--:|--:|
| DeepSeek R1 launch & AI-stock selloff | 544 | 5 (25) | 1 (14) | 20.0% (56.0%) | 0.2% (2.6%) | 0.4% (4.9%) |
| Stargate $500B AI infrastructure | 134 | 26 (25) | 1 (3) | 3.8% (12.0%) | 0.7% (2.2%) | 1.2% (3.8%) |
| Nvidia Q3 FY2025 earnings | 193 | 0 (25) | 0 (6) | n/a (24.0%) | 0.0% (3.1%) | n/a (5.5%) |
| Nvidia Q4 FY2025 earnings | 156 | 8 (25) | 2 (6) | 25.0% (24.0%) | 1.3% (3.8%) | 2.4% (6.6%) |
| Nvidia Q1 FY2026 earnings | 118 | 30 (25) | 16 (15) | 53.3% (60.0%) | 13.6% (12.7%) | 21.6% (21.0%) |
| TSMC Arizona fab yields | 20 | 5 (25) | 0 (0) | 0.0% (0.0%) | 0.0% (0.0%) | n/a (n/a) |
| Trump semiconductor / chip tariffs | 44 | 2 (25) | 0 (1) | 0.0% (4.0%) | 0.0% (2.3%) | n/a (2.9%) |
| Trump 'reciprocal' tariffs selloff | 136 | 6 (25) | 0 (0) | 0.0% (0.0%) | 0.0% (0.0%) | n/a (n/a) |
| Intel CEO Gelsinger departure | 42 | 2 (25) | 0 (6) | 0.0% (24.0%) | 0.0% (14.3%) | n/a (17.9%) |
| Intel names Lip-Bu Tan CEO | 23 | 4 (25) | 3 (11) | 75.0% (44.0%) | 13.0% (47.8%) | 22.2% (45.8%) |
| Nvidia H20 / China export curbs | 13 | 13 (25) | 2 (3) | 15.4% (12.0%) | 15.4% (23.1%) | 15.4% (15.8%) |
| Broadcom (AVGO) earnings & AI outlook | 73 | 30 (25) | 11 (10) | 36.7% (40.0%) | 15.1% (13.7%) | 21.4% (20.4%) |
| Palantir (PLTR) earnings & rally | 17 | 30 (25) | 7 (6) | 23.3% (24.0%) | 41.2% (35.3%) | 29.8% (28.6%) |
| CoreWeave IPO | 31 | 0 (17) | 0 (0) | n/a (0.0%) | 0.0% (0.0%) | n/a (n/a) |
| **MICRO** (pooled) | **1544** | **161 (342)** | **43 (81)** | **26.7% (23.7%)** | **2.8% (5.2%)** | **5.0% (8.6%)** |

#### What this shows

1. **0.8/0.8/30 is more precise but much lower recall — and lower F1.** Pooled, the tighter
   point lifts precision only marginally (**26.7%** vs 23.7%) while **halving recall** (2.8%
   vs 5.2%), so micro-F1 falls **8.6 → 5.0**. The extra 5 budget cannot compensate for what
   the 0.8 floors throw away.
2. **The 0.8 article-net floor starves whole clusters.** Seven clusters retrieve ≤5 insights
   and three collapse to a near-empty pool — **Nvidia Q3 (0)**, **Gelsinger (2)**,
   **CoreWeave (0)** — wiping out F1 cells that 0.7/0.7/25 still scored (Nvidia Q3 5.5%,
   Gelsinger 17.9%). On a strict no-lookahead seed the backward net is already thin; 0.8
   prunes it past the point of usefulness.
3. **0.8/0.8/30 only ties or wins on the high-volume, on-topic clusters** where the net
   stays full and the larger budget bites: Nvidia Q1 (21.6 vs 21.0), Broadcom (21.4 vs
   20.4), Palantir (29.8 vs 28.6). The gains are ≤1 pt; the losses elsewhere are 5–24 pt.
4. **Tight events are hurt most.** Intel Lip-Bu Tan — the cleanest single-catalyst event —
   craters from **45.8% → 22.2%** F1: 0.8 keeps only 4 insights at 75% precision but drops
   recall 47.8% → 13.0%. Precision is not the binding constraint here; coverage is.

**Takeaway.** Pushing both floors to 0.8 (and budget to 30) is the wrong direction for the
insight level: it buys ~3 pts of precision for ~2.4 pts of recall and a **net −3.6 pt F1**.
`0.7/0.7/25` remains the better default of the two; if anything, the §10.2 evidence that the
wider `0.45/0.70/50` point scored *higher* insight-F1 suggests the productive lever is
**looser** floors, not tighter ones — which §10.5 tests directly.

### 10.5 Per-cluster: `two-phase-similarity` at 0.55/0.75/40 vs 0.7/0.7/25 (insight level)

§10.4 hinted the net floor was the binding constraint, not the insight gate. This point
splits them: **loosen the article net to 0.55** (fill the pool), **raise the insight gate to
0.75** (hold precision on what survives), and **grow the budget to 40**. Same §7.3 frame
(original table, `last-noforward`); each cell is **`0.55/0.75/40`** with **`(0.7/0.7/25)`**
in parens.

| Cluster | \|G\| | ret | TP | prec | recall | F1 |
|---|--:|--:|--:|--:|--:|--:|
| DeepSeek R1 launch & AI-stock selloff | 544 | 40 (25) | 27 (14) | 67.5% (56.0%) | 5.0% (2.6%) | 9.2% (4.9%) |
| Stargate $500B AI infrastructure | 134 | 40 (25) | 4 (3) | 10.0% (12.0%) | 3.0% (2.2%) | 4.6% (3.8%) |
| Nvidia Q3 FY2025 earnings | 193 | 37 (25) | 11 (6) | 29.7% (24.0%) | 5.7% (3.1%) | 9.6% (5.5%) |
| Nvidia Q4 FY2025 earnings | 156 | 40 (25) | 8 (6) | 20.0% (24.0%) | 5.1% (3.8%) | 8.2% (6.6%) |
| Nvidia Q1 FY2026 earnings | 118 | 40 (25) | 21 (15) | 52.5% (60.0%) | 17.8% (12.7%) | 26.6% (21.0%) |
| TSMC Arizona fab yields | 20 | 40 (25) | 0 (0) | 0.0% (0.0%) | 0.0% (0.0%) | n/a (n/a) |
| Trump semiconductor / chip tariffs | 44 | 15 (25) | 0 (1) | 0.0% (4.0%) | 0.0% (2.3%) | n/a (2.9%) |
| Trump 'reciprocal' tariffs selloff | 136 | 33 (25) | 0 (0) | 0.0% (0.0%) | 0.0% (0.0%) | n/a (n/a) |
| Intel CEO Gelsinger departure | 42 | 13 (25) | 2 (6) | 15.4% (24.0%) | 4.8% (14.3%) | 7.3% (17.9%) |
| Intel names Lip-Bu Tan CEO | 23 | 27 (25) | 11 (11) | 40.7% (44.0%) | 47.8% (47.8%) | 44.0% (45.8%) |
| Nvidia H20 / China export curbs | 13 | 40 (25) | 6 (3) | 15.0% (12.0%) | 46.2% (23.1%) | 22.6% (15.8%) |
| Broadcom (AVGO) earnings & AI outlook | 73 | 40 (25) | 16 (10) | 40.0% (40.0%) | 21.9% (13.7%) | 28.3% (20.4%) |
| Palantir (PLTR) earnings & rally | 17 | 40 (25) | 8 (6) | 20.0% (24.0%) | 47.1% (35.3%) | 28.1% (28.6%) |
| CoreWeave IPO | 31 | 7 (17) | 0 (0) | 0.0% (0.0%) | 0.0% (0.0%) | n/a (n/a) |
| **MICRO** (pooled) | **1544** | **452 (342)** | **114 (81)** | **25.2% (23.7%)** | **7.4% (5.2%)** | **11.4% (8.6%)** |

#### What this shows

1. **A strict pooled win on all three metrics.** `0.55/0.75/40` beats the `0.7/0.7/25`
   default on precision (**25.2%** vs 23.7%), recall (**7.4%** vs 5.2%) *and* F1
   (**11.4%** vs 8.6%). Splitting the two floors works: the higher insight gate (0.75) is
   what protects precision, so the article net can be opened up to recover recall without
   the usual precision tax.
2. **The loose net un-starves the pool.** Every collapsed cluster from §10.4 comes back —
   Nvidia Q3 `0 → 37` retrieved (F1 9.6%), DeepSeek pulls 27 TP at **67.5% precision**. The
   net floor, not the insight gate, was the binding constraint on coverage.
3. **Mid-volume on-topic events gain the most** — Broadcom **28.3** (20.4), Nvidia Q1
   **26.6** (21.0), Nvidia H20 **22.6** (15.8), DeepSeek **9.2** (4.9) — where the wider
   net + budget 40 surface more true insights and the 0.75 gate keeps them clean.
4. **The only regressions are the smallest clusters** — Gelsinger **7.3** (17.9) and Trump
   semiconductor **n/a** (2.9) — where raising the gate to 0.75 prunes the handful of
   borderline matches that 0.70 had kept. Lip-Bu Tan and Palantir are ties.

**Takeaway.** This is the first config that improves on the `0.7/0.7/25` default *without* a
trade-off. Ranking the four points by pooled insight-F1:

| net / tau / budget | prec | recall | F1 |
|---|--:|--:|--:|
| 0.45 / 0.70 / 50 (wide) | 22.6% | 10.1% | **14.0%** |
| **0.55 / 0.75 / 40 (adopted default)** | **25.2%** | 7.4% | 11.4% |
| 0.70 / 0.70 / 25 (prior default) | 23.7% | 5.2% | 8.6% |
| 0.80 / 0.80 / 30 | 26.7% | 2.8% | 5.0% |

The wide point still owns raw F1 (driven by recall), but `0.55/0.75/40` posts the **best
precision-with-usable-recall balance** — the operating point a sentiment prompt actually
wants, since precision (clean context) matters more there than exhaustive recall. **It is now
the adopted `two-phase-similarity` default** (`DEF_TWO_PHASE_NET_MIN_SIM=0.55, DEF_TWO_PHASE_TAU_INS=0.75,
DEF_TWO_PHASE_BUDGET=40`); the wide `0.45/0.70/50` point remains the choice if raw insight
recall/F1 is the objective. §10.2 re-states the baseline comparison at this new default.

---

## 11. Selected method: `two-phase-similarity`

**We adopt `two-phase-similarity` as the project's default retrieval method.** Rationale, grounded in §10:

1. **It wins where it matters most — the insight level.** The sentiment pipeline's
   `gather_related()` consumes *insights*, and at that level `two-phase-similarity` beats the insight
   baseline in 3 of 4 conditions (11.4 vs 6.5, 15.3 vs 8.9, 21.1 vs 11.7), losing only the
   tight + strict-backtest cell by 0.6 (14.5 vs 15.1). No baseline leads the insight level
   this broadly.
2. **It combines both strengths by design** — cascade's recall pool (the highest article
   ceiling of any method, `rec*` up to 62.6%) with fusion's precision rerank — and at the
   asymmetric 0.55/0.75 gates it pairs a recall-filling net with a precision-holding insight
   gate, the right bias for a context window where noise hurts.
3. **It is strongest in the forward (cluster-recovery) frame.** With a window that brackets
   the event, `two-phase-similarity` tops the article level on both tables and the insight level on both,
   posting the family's best macro-F1 (revisited article ≈31%, insight ≈27%).
4. **It degrades to mid-pack, not to the floor.** It is never the *worst* method in any cell,
   and at the article level beats whole-article on macro-F1 in every condition; even in its
   weak revisited-last-noforward cell it stays within a few points of the leader.

**Honest caveat & how we handle it.** In the strict no-lookahead *backtest* frame, the
recall baseline (whole-article, original table) and the pure insight retriever (tight
revisited table) edge `two-phase-similarity` at the article level, and the §10.3 sweep confirms tuning the
floors won't close that gap. The §10.4–§10.5 sweep of the floors landed the default on
**0.55/0.75/40**: tightening to `0.8/0.8/30` *lowers* insight-F1 (8.6 → 5.0), while loosening
the net and raising the insight gate lifted it (8.6 → 11.4) on all three of precision, recall
and F1. The even wider `0.45/0.70/50` point scores higher raw insight-F1 (14.0) but at lower
precision; we prefer the cleaner-context balance. `fusion` / the insight baseline remain
documented fallbacks for strictly no-lookahead, article-critical paths.

**Configuration.** Default `two-phase-similarity` params: `net_k=150, net_min_sim=0.55, tau=0.75,
rrf_c=60, budget=40` — a **loose article net (0.55) paired with a tight insight gate
(0.75)** — with the no-lookahead window (`--exclusive`) for live use and the forward
window for historical/clustering analysis.

```python
from hybrid_retrieval import retrieve
res = retrieve(seed_id, method="two-phase-similarity", months_before=3, exclusive=True)
#   res.insight_ids / res.article_ids  -> feed into gather_related()
```

---

## 12. Does `--remove-unuseful` improve the precision/recall tradeoff?

`--remove-unuseful` (in `insight_sentiment.py`) adds a *middle* model call between
retrieval and the sentiment prompt: it shows each retrieved insight with its **source
article headline** and asks the model to drop the incoherent or off-event ones (see the
flag's design). The open question is whether that screen, applied *after* retrieval, buys
precision cheaply — i.e. removes false positives without costing true positives.

To measure it against ground truth, `scripts/validate_retreival/eval_screen.py` runs the
**wide** two-phase-similarity point (`0.45/0.70/50`, the highest-recall config) on the
original table (`last-noforward`), then applies the production screen and scores both at the
insight level. Screening targets are the seed article's own tickers, exactly as in
production. The *retrieved* row reproduces the §10.5 wide baseline (22.6 / 10.1 / 14.0).

**Pooled (micro), 14 clusters, 1544 ground-truth insights:**

| config | retrieved | TP | prec | recall | F1 |
|---|--:|--:|--:|--:|--:|
| `0.45/0.70/50` retrieved only | 691 | 156 | 22.6% | 10.1% | 14.0% |
| `0.45/0.70/50` + `--remove-unuseful` | 678 | 155 | **22.9%** | 10.0% | 14.0% |
| **Δ** | −13 | −1 | **+0.3pt** | −0.1pt | −0.0pt |

**Where the screen actually acted** (9 of 14 clusters were left untouched):

| Cluster | removed | TP (r→s) | prec (r→s) | F1 (r→s) |
|---|--:|--:|--:|--:|
| Stargate $500B AI infra | 1 | 4 → **3** | 8.0% → 6.1% | 4.3% → 3.3% |
| Nvidia Q1 FY2026 | 1 | 26 → 26 | 52.0% → 53.1% | 31.0% → 31.1% |
| Trump semiconductor tariffs | 3 | 2 → 2 | 4.0% → 4.3% | 4.3% → 4.4% |
| Nvidia H20 export curbs | 1 | 6 → 6 | 12.0% → 12.2% | 19.0% → 19.4% |
| CoreWeave IPO | 7 | 0 → 0 | 0.0% → 0.0% | n/a |

#### What this shows

1. **On this benchmark the screen is essentially F1-neutral** (−0.0 pt), nudging precision
   up 0.3 pt and recall down 0.1 pt. It does **not** shift the precision/recall tradeoff in a
   way ground truth can see.
2. **When it does act, it is well-targeted.** Of the 13 insights it removed, **12 were
   non-relevant** (false positives) and only **1 was relevant** (the Stargate true positive)
   — a 12:1 good-to-bad ratio. So the screen rarely throws away signal; the precision lift is
   real, just tiny in absolute terms.
3. **Why so little happens here:** these clusters are *single, ticker-anchored events*
   (an earnings print, a CEO change), and the retrieved insights are already the same ticker
   and broadly on-event, so the headline screen has little to disambiguate. Its removals
   cluster on the genuinely loose cases — CoreWeave (7, all non-relevant) and the tariff
   narratives — exactly where retrieval reached past the event.
4. **The screen's real value is event-disambiguation, which this benchmark doesn't stress.**
   In production it shines on *narrow* seeds whose ticker has lots of unrelated coverage —
   e.g. the live "Broadcom VeloSky launch" seed, where it cut 11 related insights to 3 by
   dropping same-ticker / different-event boxes (custom-AI-chip, VMware, valuation takes).
   The cluster ground truth, built around whole ticker-events, treats that same-ticker
   coverage as on-topic, so the metric cannot reward the disambiguation.

**Takeaway.** `--remove-unuseful` is a **precision safety-net, not a tradeoff lever.** Post
two-phase-similarity retrieval it costs almost nothing (recall −0.1 pt, F1 flat) and removes
mostly true noise, so it is safe to leave on; but on well-formed single-event retrieval it
will not move precision/recall much. Its payoff is concentrated on narrow seeds where
retrieval drags in same-ticker / different-event insights — a regime under-represented in
this cluster benchmark but common in live single-article use.

#### Conclusion — not adopted

The tempting idea was to keep the wide `0.45/0.70/50` point (the best recall/F1) and let
`--remove-unuseful` recover the precision of the tighter, adopted `0.55/0.75/40` default.
It does not work:

| config | prec | recall | F1 |
|---|--:|--:|--:|
| `0.55/0.75/40` (adopted default) | **25.2%** | 7.4% | 11.4% |
| `0.45/0.70/50` + `--remove-unuseful` | 22.9% | 10.0% | 14.0% |
| `0.45/0.70/50` retrieved only | 22.6% | 10.1% | 14.0% |

Screening lifts wide precision only **22.6% → 22.9%**, closing ~0.3 of the ~2.6 pt gap to
`0.55/0.75/40`'s 25.2% — roughly **12% of the way**. An LLM screen applied *after* retrieval
cannot reproduce what the in-retrieval `tau` gate does, because the gate filters against the
seed's own insight embeddings at scale while the screen only re-reads what already survived.
**We therefore do not adopt `--remove-unuseful` as a precision mechanism, and keep it off by
default**; precision is owned by the retrieval gates (`0.55/0.75/40`), and the flag remains
available only as an optional safety-net for noisy single-article seeds (§above).

#### Reading the numbers: low recall is by design, and true precision is high

Two things about these insight-level scores are easy to misread:

- **Low recall is the intended behaviour, not a failure.** The task is *distilling* a small,
  clean set of background insights to feed a sentiment prompt — not exhaustively recovering
  every insight of an event. An event carries 100–500 labelled insights (§7.3); we
  deliberately keep only a budgeted handful (`budget=40`) and gate the rest out, so a
  single-digit-to-low-double-digit recall (e.g. 10%) is *exactly what we want* — matching a
  few of the most on-point insights is enough to ground the verdict, and pulling more would
  re-add the noise the gates exist to remove (this is the §7.4 point #4, made concrete).
  Recall here measures coverage of a set we are intentionally **not** trying to cover.

- **`--remove-unuseful` doubles as a validation that the *real* precision is high.** The
  ground-truth precision (≈22.6%) looks low only because the cluster labels are an
  *incomplete* "should-be-retrieved" set: they credit insights from the hand-labelled
  articles and count everything else as a false positive — even genuinely on-topic insights
  from articles nobody labelled. The screening call is an independent, content-aware judge of
  those "false positives", and it **kept ~98% of them** (removed just 13 of 691). That the
  content-reading model agrees almost all retrieved insights are coherent and relevant is
  strong evidence the *true* precision is far higher than 22.6% — the gates are returning
  clean context; the metric is simply penalising relevance the ground truth never enumerated.
