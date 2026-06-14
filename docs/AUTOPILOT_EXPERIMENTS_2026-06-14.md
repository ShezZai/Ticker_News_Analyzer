# Autopilot verdict-optimization session — 2026-06-14 (04:07–07:00 JST)

Autonomous experiment sweep to find how to lift the sentiment **verdict_score**
(buy/sell/hold vs realized entry→close price move) on the 400-article
`400-articles-sentiment-verdict` dataset. All runs: `--skip-stages all`,
`--verdict-model flash`. Code changes live on branch
`exp/verdict-optimization-autopilot` (additive; defaults preserve prior behaviour).

**Baseline:** `ticker-history` + corrected header = **verdict_score 0.396**
(act/no-act fixed across all runs: acc=0.57 prec=0.82 rec=0.61 f1=0.70, 245 actionable).

## Results

| # | Experiment | What it changes | verdict_score | vs 0.396 |
|---|---|---|---|---|
| 0 | `ticker-history` (re-baseline, `by0paeqt1`) | champion, recency, cap 15 | 0.396 | — |
| 0b | `ticker-history` (rebaseline-same-session) | identical config, variance check | **0.404** | +0.008 (noise) |
| 1 | `ticker-history` + `--summary-model lite` | cheap pre-verdict body distillation | **0.351** | −0.045 ❌ |
| 2 | `ticker-history-fallback` | supplement thin-history names w/ cross-corpus | **0.371** | −0.025 ❌ |
| 3 | `ticker-blend` | relevance-seeded recency within ticker | **0.384** | −0.012 ❌ |
| 4 | `ticker-history` cap 25 | more of the ticker's own history | _pending_ | |
| 5 | `ticker-history` cap 10 | tighter | **0.393** | −0.003 ≈ |
| 6 | `ticker-history` lookback 180d | reach further back | **0.392** | −0.004 ≈ |
| 7 | `ticker-history` lookback 30d | tighter recency window | **0.396** | 0.000 ≈ |
| 8 | `ticker-history` + `--summary-model flash` | stronger summarizer (vs lite's −0.05) | **0.347** | −0.049 ❌ |

_(cap 25 was queued but not run — out of time before the 07:00 stop. cap 10 and the
lookback sweep already showed the champion is flat in these hyperparameters.)_

**Champion stands: `ticker-history`, recency, cap 15, lookback 90d ≈ 0.40.** No
tested change beat it outside the ±0.01 noise floor.

## Conclusions

1. **`ticker-history` (recency, cap 15, lookback 90d) is a robust local optimum
   at 0.40 ± 0.01.** Nothing tested beats it outside noise; the cap and lookback
   sweeps in both directions land within ±0.01 (neutral). The champion is also the
   cheapest mode (one SQL join, no ANN, no extra LLM call) — so it wins on cost too.

2. **Every attempt to add MORE context to the precedents hurt, monotonically with
   "distance from pure ticker recency":**
   - pre-verdict summarization: lite **−0.05**, flash **−0.05** — a summarizer strips
     signal the verdict needs along with the boilerplate. A STRONGER summarizer did
     not recover it (flash 0.347 ≈ lite 0.351), so the loss is intrinsic to
     distillation, not a model-quality problem. The raw body beats a lossy brief.
   - cross-corpus fallback for thin names: **−0.03** — padding a thin ticker history
     with off-name similar articles is worse than leaving it thin. *Ticker-scoping
     purity matters more than precedent quantity.*
   - relevance-seeded blend: **−0.02** — any relevance injection dilutes recency
     (consistent with the earlier `ticker-relevant` 0.367 negative result).

3. **verdict_score is at its retrieval-tuning ceiling (~0.40).** It measures
   *directional* accuracy on 245 near-coin-flip price moves; precedent retrieval
   gives topical/narrative context but little genuine directional edge. Tuning
   *which* prior articles the verdict reads has run out of room.

### Recommended next lever (not retrieval tuning)

**Outcome-enriched precedents** — annotate each precedent line with the realized
price move that FOLLOWED that prior article ("2026-05-30 [NVDA] beat & raise →
+4.1% next session"). This gives the verdict actual directional evidence for "what
usually happens to this name after news like this," rather than just what the prior
article said. It's the one untested change that targets the directional ceiling
head-on (flagged in the TradingAgents borrow list). Cost: a price lookup per
precedent — too slow for this session, but precomputable into a column. This is the
recommended direction over any further precedent-retrieval tuning.

## New code (branch `exp/verdict-optimization-autopilot`)

- **`ticker-history-fallback`** precedent mode — `stages.ticker_history_fallback`:
  ticker's own recency boxes; when fewer than `SENTIMENT_PRECEDENT_TICKER_FALLBACK_MIN`
  (default 3), append cross-corpus `article_similarity` lines. Targets actionable
  articles where pure ticker-history is sparse.
- **`ticker-blend`** precedent mode — `stages.ticker_blend_insights`: relevance-first
  (the few most-similar prior boxes) then recency fills the remainder, deduped by
  source article.
- **`SENTIMENT_PRECEDENT_TICKER_HISTORY_LIMIT`** (default 15) — cap for all
  ticker-scoped modes, swept via env at launch.
- Mode-aware verdict headers for both new kinds; `_render_box_groups` extracted and
  shared by the ticker-scoped modes.

## Methodology caveats

- **Run-to-run noise ≈ ±0.01 (measured).** Two runs of the *identical* champion
  config scored **0.396** (`by0paeqt1`) and **0.404** (same session) — a 0.008
  spread from model non-determinism alone. So the champion is **0.40 ± 0.01**, and
  any delta ≲ 0.01–0.02 (≈ ≤2–5 of 245 items flipping) is indistinguishable from
  noise. Deltas are quoted vs 0.396 in the table; read them against this floor.
- **What 0.40 means.** verdict_score is *directional accuracy* (buy/sell/hold vs
  the realized ±0.3% deadband move) on the 245 is_act items (TP=200 real movers +
  FP=45 flat-truth). Predicting next-move direction from news is near-random, so
  ~0.40 sits close to a "direction is chance" band. Precedent retrieval can only
  help insofar as it confers genuine directional edge (what's already priced in).
- The served verdict prompt is Langfuse **v4** (no `provider_sentiment`, has the
  mode-aware `precedents_header`), identical for every run including `by0paeqt1`.

## Notes / decisions

- Runs are **sequential** — two concurrent 400-runs each fan ~7 Gemini calls/item
  and hit the rate limiter, which drops/penalizes items and corrupts comparisons.
- act/no-act is unchanged by any of these (precedent source never touches
  classification), so every row shares the same confusion matrix; only
  verdict_score moves.
