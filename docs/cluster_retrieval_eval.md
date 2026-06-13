# Cluster retrieval evaluation

Seed = each cluster's `later act` real-news article. lookback = cluster column **+1d**. threshold 0.7, limit 60, all categories.
Retrieval core = `stages._ranked_insight_hits`; article-level hits (a target is hit if ≥1 of its boxes is retrieved).
`insight_orig` scored vs the article-insights column; distilled variants vs the distilled column.


# Per-method (vs own member column)


## insight_orig

| cluster | seed | lookback | expected | retrieved | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek R1 launch & AI-stock selloff (Jan 20-27 2025) | 2723 | 3d | 17 | 8 | 8 | 0 | 9 | 1.00 | 0.47 | 0.64 |
| Amazon Q4 2024 earnings beat (Feb 6 2025) | 3036 | 20d | 9 | 23 | 4 | 19 | 5 | 0.17 | 0.44 | 0.25 |
| Broadcom Q1 FY2025 earnings & AI outlook (Mar 6 2025) | 3984 | 90d | 37 | 34 | 18 | 16 | 19 | 0.53 | 0.49 | 0.51 |
| Broadcom Q2 FY2025 earnings (Jun 5 2025) | 6412 | 88d | 32 | 35 | 15 | 20 | 17 | 0.43 | 0.47 | 0.45 |
| Oracle Q4 FY2025 cloud earnings (Jun 11 2025) | 6563 | 85d | 4 | 9 | 4 | 5 | 0 | 0.44 | 1.00 | 0.62 |
| Broadcom Q3 FY2025 earnings (Sep 4 2025) | 9128 | 85d | 24 | 31 | 15 | 16 | 9 | 0.48 | 0.62 | 0.55 |
| Palantir Q3 2025 earnings beat (Nov 3 2025) | 11433 | 22d | 17 | 28 | 13 | 15 | 4 | 0.46 | 0.76 | 0.58 |
| Dell Q4 FY2026 earnings beat (Feb 26 2026) | 16346 | 88d | 4 | 11 | 4 | 7 | 0 | 0.36 | 1.00 | 0.53 |

**Aggregate.** micro P/R/F1 = 0.45/0.56/0.50; macro P/R/F1 = 0.49/0.66/0.51 (ΣTP=81,ΣFP=98,ΣFN=63).

## insights_distilled_two_pass

| cluster | seed | lookback | expected | retrieved | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek R1 launch & AI-stock selloff (Jan 20-27 2025) | 2723 | 3d | 16 | 10 | 10 | 0 | 6 | 1.00 | 0.62 | 0.77 |
| Amazon Q4 2024 earnings beat (Feb 6 2025) | 3036 | 20d | 8 | 21 | 3 | 18 | 5 | 0.14 | 0.38 | 0.21 |
| Broadcom Q1 FY2025 earnings & AI outlook (Mar 6 2025) | 3984 | 90d | 34 | 37 | 14 | 23 | 20 | 0.38 | 0.41 | 0.39 |
| Broadcom Q2 FY2025 earnings (Jun 5 2025) | 6412 | 88d | 28 | 37 | 12 | 25 | 16 | 0.32 | 0.43 | 0.37 |
| Oracle Q4 FY2025 cloud earnings (Jun 11 2025) | 6563 | 85d | 3 | 5 | 2 | 3 | 1 | 0.40 | 0.67 | 0.50 |
| Broadcom Q3 FY2025 earnings (Sep 4 2025) | 9128 | 85d | 23 | 34 | 12 | 22 | 11 | 0.35 | 0.52 | 0.42 |
| Palantir Q3 2025 earnings beat (Nov 3 2025) | 11433 | 22d | 12 | 15 | 6 | 9 | 6 | 0.40 | 0.50 | 0.44 |
| Dell Q4 FY2026 earnings beat (Feb 26 2026) | 16346 | 88d | 4 | 9 | 3 | 6 | 1 | 0.33 | 0.75 | 0.46 |

**Aggregate.** micro P/R/F1 = 0.37/0.48/0.42; macro P/R/F1 = 0.42/0.53/0.45 (ΣTP=62,ΣFP=106,ΣFN=66).

## insights_distilled_two_pass_drop_filtered

| cluster | seed | lookback | expected | retrieved | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek R1 launch & AI-stock selloff (Jan 20-27 2025) | 2723 | 3d | 16 | 10 | 10 | 0 | 6 | 1.00 | 0.62 | 0.77 |
| Amazon Q4 2024 earnings beat (Feb 6 2025) | 3036 | 20d | 8 | 19 | 2 | 17 | 6 | 0.11 | 0.25 | 0.15 |
| Broadcom Q1 FY2025 earnings & AI outlook (Mar 6 2025) | 3984 | 90d | 34 | 40 | 14 | 26 | 20 | 0.35 | 0.41 | 0.38 |
| Broadcom Q2 FY2025 earnings (Jun 5 2025) | 6412 | 88d | 28 | 37 | 13 | 24 | 15 | 0.35 | 0.46 | 0.40 |
| Oracle Q4 FY2025 cloud earnings (Jun 11 2025) | 6563 | 85d | 3 | 5 | 2 | 3 | 1 | 0.40 | 0.67 | 0.50 |
| Broadcom Q3 FY2025 earnings (Sep 4 2025) | 9128 | 85d | 23 | 34 | 12 | 22 | 11 | 0.35 | 0.52 | 0.42 |
| Palantir Q3 2025 earnings beat (Nov 3 2025) | 11433 | 22d | 12 | 13 | 5 | 8 | 7 | 0.38 | 0.42 | 0.40 |
| Dell Q4 FY2026 earnings beat (Feb 26 2026) | 16346 | 88d | 4 | 8 | 3 | 5 | 1 | 0.38 | 0.75 | 0.50 |

**Aggregate.** micro P/R/F1 = 0.37/0.48/0.41; macro P/R/F1 = 0.41/0.51/0.44 (ΣTP=61,ΣFP=105,ΣFN=67).

### Per-method (vs own member column) — summary

| method | micro P | micro R | micro F1 | macro P | macro R | macro F1 |
|---|---|---|---|---|---|---|
| insight_orig | 0.45 | 0.56 | 0.50 | 0.49 | 0.66 | 0.51 |
| insights_distilled_two_pass | 0.37 | 0.48 | 0.42 | 0.42 | 0.53 | 0.45 |
| insights_distilled_two_pass_drop_filtered | 0.37 | 0.48 | 0.41 | 0.41 | 0.51 | 0.44 |

# Apples-to-apples (vs overlap = articles in both corpora)


## insight_orig

| cluster | seed | lookback | expected | retrieved | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek R1 launch & AI-stock selloff (Jan 20-27 2025) | 2723 | 3d | 16 | 8 | 8 | 0 | 8 | 1.00 | 0.50 | 0.67 |
| Amazon Q4 2024 earnings beat (Feb 6 2025) | 3036 | 20d | 8 | 23 | 4 | 19 | 4 | 0.17 | 0.50 | 0.26 |
| Broadcom Q1 FY2025 earnings & AI outlook (Mar 6 2025) | 3984 | 90d | 34 | 34 | 16 | 18 | 18 | 0.47 | 0.47 | 0.47 |
| Broadcom Q2 FY2025 earnings (Jun 5 2025) | 6412 | 88d | 28 | 35 | 13 | 22 | 15 | 0.37 | 0.46 | 0.41 |
| Oracle Q4 FY2025 cloud earnings (Jun 11 2025) | 6563 | 85d | 3 | 9 | 3 | 6 | 0 | 0.33 | 1.00 | 0.50 |
| Broadcom Q3 FY2025 earnings (Sep 4 2025) | 9128 | 85d | 23 | 31 | 14 | 17 | 9 | 0.45 | 0.61 | 0.52 |
| Palantir Q3 2025 earnings beat (Nov 3 2025) | 11433 | 22d | 12 | 28 | 8 | 20 | 4 | 0.29 | 0.67 | 0.40 |
| Dell Q4 FY2026 earnings beat (Feb 26 2026) | 16346 | 88d | 4 | 11 | 4 | 7 | 0 | 0.36 | 1.00 | 0.53 |

**Aggregate.** micro P/R/F1 = 0.39/0.55/0.46; macro P/R/F1 = 0.43/0.65/0.47 (ΣTP=70,ΣFP=109,ΣFN=58).

## insights_distilled_two_pass

| cluster | seed | lookback | expected | retrieved | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek R1 launch & AI-stock selloff (Jan 20-27 2025) | 2723 | 3d | 16 | 10 | 10 | 0 | 6 | 1.00 | 0.62 | 0.77 |
| Amazon Q4 2024 earnings beat (Feb 6 2025) | 3036 | 20d | 8 | 21 | 3 | 18 | 5 | 0.14 | 0.38 | 0.21 |
| Broadcom Q1 FY2025 earnings & AI outlook (Mar 6 2025) | 3984 | 90d | 34 | 37 | 14 | 23 | 20 | 0.38 | 0.41 | 0.39 |
| Broadcom Q2 FY2025 earnings (Jun 5 2025) | 6412 | 88d | 28 | 37 | 12 | 25 | 16 | 0.32 | 0.43 | 0.37 |
| Oracle Q4 FY2025 cloud earnings (Jun 11 2025) | 6563 | 85d | 3 | 5 | 2 | 3 | 1 | 0.40 | 0.67 | 0.50 |
| Broadcom Q3 FY2025 earnings (Sep 4 2025) | 9128 | 85d | 23 | 34 | 12 | 22 | 11 | 0.35 | 0.52 | 0.42 |
| Palantir Q3 2025 earnings beat (Nov 3 2025) | 11433 | 22d | 12 | 15 | 6 | 9 | 6 | 0.40 | 0.50 | 0.44 |
| Dell Q4 FY2026 earnings beat (Feb 26 2026) | 16346 | 88d | 4 | 9 | 3 | 6 | 1 | 0.33 | 0.75 | 0.46 |

**Aggregate.** micro P/R/F1 = 0.37/0.48/0.42; macro P/R/F1 = 0.42/0.53/0.45 (ΣTP=62,ΣFP=106,ΣFN=66).

## insights_distilled_two_pass_drop_filtered

| cluster | seed | lookback | expected | retrieved | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek R1 launch & AI-stock selloff (Jan 20-27 2025) | 2723 | 3d | 16 | 10 | 10 | 0 | 6 | 1.00 | 0.62 | 0.77 |
| Amazon Q4 2024 earnings beat (Feb 6 2025) | 3036 | 20d | 8 | 19 | 2 | 17 | 6 | 0.11 | 0.25 | 0.15 |
| Broadcom Q1 FY2025 earnings & AI outlook (Mar 6 2025) | 3984 | 90d | 34 | 40 | 14 | 26 | 20 | 0.35 | 0.41 | 0.38 |
| Broadcom Q2 FY2025 earnings (Jun 5 2025) | 6412 | 88d | 28 | 37 | 13 | 24 | 15 | 0.35 | 0.46 | 0.40 |
| Oracle Q4 FY2025 cloud earnings (Jun 11 2025) | 6563 | 85d | 3 | 5 | 2 | 3 | 1 | 0.40 | 0.67 | 0.50 |
| Broadcom Q3 FY2025 earnings (Sep 4 2025) | 9128 | 85d | 23 | 34 | 12 | 22 | 11 | 0.35 | 0.52 | 0.42 |
| Palantir Q3 2025 earnings beat (Nov 3 2025) | 11433 | 22d | 12 | 13 | 5 | 8 | 7 | 0.38 | 0.42 | 0.40 |
| Dell Q4 FY2026 earnings beat (Feb 26 2026) | 16346 | 88d | 4 | 8 | 3 | 5 | 1 | 0.38 | 0.75 | 0.50 |

**Aggregate.** micro P/R/F1 = 0.37/0.48/0.41; macro P/R/F1 = 0.41/0.51/0.44 (ΣTP=61,ΣFP=105,ΣFN=67).

### Apples-to-apples (vs overlap = articles in both corpora) — summary

| method | micro P | micro R | micro F1 | macro P | macro R | macro F1 |
|---|---|---|---|---|---|---|
| insight_orig | 0.39 | 0.55 | 0.46 | 0.43 | 0.65 | 0.47 |
| insights_distilled_two_pass | 0.37 | 0.48 | 0.42 | 0.42 | 0.53 | 0.45 |
| insights_distilled_two_pass_drop_filtered | 0.37 | 0.48 | 0.41 | 0.41 | 0.51 | 0.44 |

## Caveats

- Precision is vs. the curated set → lower bound on true relevance.
- Apples-to-apples uses the overlap set (retrievable by all methods); recall there is the cleanest comparison.
- Many remaining misses sit at 0.60–0.70 similarity (just under the 0.7 floor) → recall is threshold-limited.
