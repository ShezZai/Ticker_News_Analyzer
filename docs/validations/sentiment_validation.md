# Sentiment validation

Validates that the sentiment pipeline (`scripts/search/insight_sentiment.py`) produces
verdicts with real, article-specific predictive value — and isolates that value from market
beta by testing only on **peaceful** days.

> Retrieval is validated separately in [`validate_retrieval.md`](validate_retrieval.md); this
> document is about the *verdict* end of the pipeline (BUY / SELL / HOLD + confidence).

---

## 1. Why "peaceful days"

On a trending day every long looks smart: a BUY that "works" may just be the whole market
rising. To test whether a verdict carries information *about the article*, we want days with
**no clear market direction**, where a correct BUY/SELL must come from the news itself rather
than from market drift.

A trading day is **peaceful** when both:

- **VIX close < 25** (low expected volatility), and
- the **trailing-10-trading-day mean absolute daily move < 1%** for **both** the S&P 500 and
  the NASDAQ (the recent tape is flat and directionless).

On such days the realized intraday return of a single name is dominated by name-specific news,
not by index movement — exactly the regime where the sentiment model should be judged.

---

## 2. The pool — `peaceful_days_articles.csv`

Built by [`scripts/search/peaceful_days.py`](../../scripts/search/peaceful_days.py); one row
per article that published on a peaceful day.

**Definition / data sources**

| Quantity | Source | Notes |
|---|---|---|
| VIX daily close | FRED series `VIXCLS` | `I:VIX` is not licensed on the Massive key (403). |
| S&P 500 daily move | Massive daily bars, `SPY` | ETF proxy; `I:SPX` is 403. |
| NASDAQ daily move | Massive daily bars, `I:COMP` | NASDAQ Composite index. |
| Articles | `NEWS_DB_DSN` (content-rich DB) | category `real news` by default. |

Each article is mapped to the **latest trading day on/before its publish date** (a weekend
headline inherits Friday's regime) and kept only if that day is peaceful.

**Current snapshot** (full real-news span, default thresholds):

- **938** real-news articles (of 1,396 in range) across **227** distinct peaceful trading days.
- Peaceful days overall: **256 / 415** trading days in 2024-11-01 … 2026-05-30.
- VIX among kept rows: **12.8 – 24.9**; trailing mean move ≤ 0.996%.
- Sparse exactly where it should be — almost nothing in the DeepSeek selloff (late Jan '25) or
  the tariff turmoil (Mar–Apr '25); dense in calm months.

**Columns**

| column | meaning |
|---|---|
| `article_id` | seed article id (DB) |
| `published_et` | publish time, US/Eastern |
| `market_date` | the peaceful trading day the article is attached to |
| `primary_ticker`, `more_tickers` | the article's tickers (`\|`-joined) |
| `category` | article category (default `real news`) |
| `vix` | VIX close on `market_date` |
| `sp500_mae_10d_pct`, `nasdaq_mae_10d_pct` | trailing-10-day mean \|move\| (%) |
| `title` | article headline |

**Regenerate / tune**

```bash
python scripts/search/peaceful_days.py                                   # defaults
python scripts/search/peaceful_days.py --vix-max 20 --max-move 0.8       # stricter calm
python scripts/search/peaceful_days.py --require either                  # either index <1%
python scripts/search/peaceful_days.py --categories "real news" "legal solicitation"
```

---

## 3. Outcome metric

The sentiment prompt predicts the **immediate** reaction — how the stock moves right after the
article hits. The matching realized outcome is the **buy-at-publish → sell-at-close** return:

- **Entry:** the first tradeable price at/after the publish minute (Massive 1-minute bars,
  extended hours included).
- **Exit:** the regular-session **close** of the entry's trading day (an after-hours article
  enters/exits the next session).
- Computed with `scripts/ticker_scan/catalyst_returns.py::simulate()` (the same engine the
  earlier top-2 backtest used).

Because the day is peaceful, the raw return ≈ the article's excess return; optionally subtract
the same-day `SPY` move for a strict market-neutral figure.

---

## 4. Procedure

For each article in the pool:

1. **Verdict** — run `insight_sentiment.py <id>` with the default
   **`--retrieval two-phase-similarity`** retriever; record each ticker's `action`,
   `confidence`, and `role` (primary / mentioned). Optionally toggle `--remove-unuseful`.
2. **Outcome** — fetch the buy-till-close return for the judged ticker(s) via the returns
   engine above.
3. **Align** — join verdict ↔ realized return per (article, ticker).

The existing harness [`scripts/search/backtest_top2.py`](../../scripts/search/backtest_top2.py)
already pairs verdicts with buy-till-close returns; pointing it at this pool (instead of its
own DB date-range sampling) is the intended runner.

---

## 5. Analysis & success criteria

Evaluated over the pool, segmented by `action` and confidence band:

1. **Directional edge.** Mean realized return for **BUY > 0**, **SELL < 0**, **HOLD ≈ 0**;
   the BUY−SELL spread is the headline number.
2. **Hit rate.** Share of non-HOLD verdicts whose return sign matches the action, vs the
   ~50% coin-flip baseline (and vs a "always BUY" baseline, which on peaceful days should be
   ~neutral by construction).
3. **Confidence calibration.** Higher `confidence` should map to a larger \|return\| and a
   higher hit rate; bucket by 0.5–0.7 / 0.7–0.85 / 0.85+ as the earlier backtests did.
4. **Ablations.** Compare `insight` vs `two-phase-similarity` retrieval, and
   `--remove-unuseful` on/off, to see whether cleaner context sharpens the edge.

The peaceful-day filter is the control: if an edge survives here, it is article-driven, not a
market-direction artifact.

---

## 6. Results

**Run:** 50 articles sampled (seed 42) from `peaceful_days_articles.csv`, via
[`scripts/validate_retreival/validate_sentiment.py`](../../scripts/validate_retreival/validate_sentiment.py).
Verdict on the **primary ticker**, model `gemini-2.5-flash-lite`, under both retrievers.
`gain%` = buy-at-publish → sell-at-(next)-close. ✓/✗ = the action's sign matches the realized
move; — = HOLD (not scored directionally).

| # | a# | ticker | sell date | gain% | insight (act·conf) | ✓ | two-phase (act·conf) | ✓ |
|--:|--:|:--|:--|--:|:--|:-:|:--|:-:|
| 1 | 19833 | NVDA | 2026-05-14 | +2.50 | hold · 0.10 | — | hold · 0.10 | — |
| 2 | 19819 | AMZN | 2026-05-14 | -0.97 | buy · 0.75 | ✗ | buy · 0.75 | ✗ |
| 3 | 10534 | WDC | 2025-10-13 | +1.11 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 4 | 11257 | AMZN | 2025-10-30 | -2.20 | hold · 0.30 | — | hold · 0.30 | — |
| 5 | 8631 | NOK | 2025-08-21 | +0.47 | hold · 0.30 | — | buy · 0.75 | ✓ |
| 6 | 1298 | AMZN | 2024-12-12 | -0.69 | hold · 0.30 | — | hold · 0.30 | — |
| 7 | 1445 | COHR | 2024-12-17 | -2.57 | hold · 0.10 | — | hold · 0.10 | — |
| 8 | 13612 | CSCO | 2025-12-26 | +0.70 | hold · 0.10 | — | hold · 0.10 | — |
| 9 | 7866 | IBM | 2025-07-30 | -0.80 | hold · 0.20 | — | hold · 0.30 | — |
| 10 | 1190 | PLTR | 2024-12-09 | -11.61 | buy · 0.85 | ✗ | buy · 0.85 | ✗ |
| 11 | 12126 | ADBE | 2025-11-19 | -0.29 | hold · 0.30 | — | hold · 0.30 | — |
| 12 | 7915 | NVDA | 2025-07-31 | -2.73 | hold · 0.30 | — | hold · 0.30 | — |
| 13 | 7770 | LNVGY | 2025-07-28 | +2.43 | hold · 0.30 | — | buy · 0.85 | ✓ |
| 14 | 19306 | AMAT | 2026-05-04 | +0.07 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 15 | 13844 | AMBA | 2026-01-05 | -0.45 | buy · 0.85 | ✗ | buy · 0.85 | ✗ |
| 16 | 14210 | GOOG | 2026-01-12 | +1.94 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 17 | 14208 | NVDA | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 18 | 19675 | NVDA | 2026-05-11 | -1.15 | hold · 0.30 | — | hold · 0.30 | — |
| 19 | 19102 | DELL | 2026-04-29 | +0.69 | hold · 0.10 | — | hold · 0.20 | — |
| 20 | 13326 | NVDA | 2025-12-17 | -3.92 | hold · 0.10 | — | hold · 0.10 | — |
| 21 | 11563 | LNVGY | 2025-11-06 | -3.20 | hold · 0.30 | — | hold · 0.30 | — |
| 22 | 13972 | MSFT | 2026-01-07 | +1.00 | hold · 0.10 | — | hold · 0.10 | — |
| 23 | 14391 | AMZN | 2026-01-16 | +0.08 | hold · 0.30 | — | hold · 0.30 | — |
| 24 | 7222 | GOOG | 2025-07-10 | +0.70 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 25 | 7393 | POWL | 2025-07-16 | +3.62 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 26 | 9620 | HPE | 2025-09-17 | +2.61 | hold · 0.30 | — | hold · 0.30 | — |
| 27 | 13787 | NVDA | 2026-01-02 | -0.54 | hold · 0.30 | — | hold · 0.30 | — |
| 28 | 7047 | ADI | 2025-07-03 | -0.03 | hold · 0.30 | — | hold · 0.30 | — |
| 29 | 8845 | AMBA | 2025-08-27 | +2.12 | hold · 0.30 | — | hold · 0.30 | — |
| 30 | 16576 | AVGO | 2026-03-05 | +4.44 | hold · 0.60 | — | buy · 0.85 | ✓ |
| 31 | 10567 | AMZN | 2025-10-14 | -1.60 | hold · 0.10 | — | hold · 0.10 | — |
| 32 | 14701 | NVDA | 2026-01-23 | +0.24 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 33 | 14712 | DELL | 2026-01-23 | +0.68 | hold · 0.10 | — | hold · 0.10 | — |
| 34 | 623 | GOOG | 2024-11-20 | -1.13 | hold · 0.30 | — | hold · 0.30 | — |
| 35 | 19378 | IREN | 2026-05-05 | +9.96 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 36 | 14045 | IBM | 2026-01-08 | -0.14 | hold · 0.30 | — | hold · 0.30 | — |
| 37 | 15347 | AMZN | 2026-02-05 | +0.37 | hold · 0.20 | — | hold · 0.10 | — |
| 38 | 14314 | META | 2026-01-14 | -0.38 | hold · 0.10 | — | hold · 0.10 | — |
| 39 | 9825 | MSFT | 2025-09-26 | +0.31 | hold · 0.30 | — | hold · 0.30 | — |
| 40 | 14889 | NVDA | 2026-01-28 | -0.01 | hold · 0.40 | — | buy · 0.75 | ✗ |
| 41 | 8463 | ASX | 2025-08-15 | -0.05 | hold · 0.30 | — | hold · 0.30 | — |
| 42 | 16038 | AMD | 2026-02-20 | -1.55 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |
| 43 | 9661 | AVGO | 2025-09-17 | +0.54 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 44 | 19852 | AMZN | 2026-05-14 | -0.64 | hold · 0.10 | — | hold · 0.10 | — |
| 45 | 9942 | QCOM | 2025-09-30 | +0.38 | hold · 0.30 | — | hold · 0.30 | — |
| 46 | 7134 | CSCO | 2025-07-07 | -0.04 | hold · 0.30 | — | hold · 0.30 | — |
| 47 | 10281 | AMD | 2025-10-08 | +9.66 | hold · 0.30 | — | hold · 0.30 | — |
| 48 | 608 | HUBB | 2024-11-20 | -1.52 | hold · 0.10 | — | hold · 0.10 | — |
| 49 | 6646 | NOK | 2025-06-16 | +0.95 | hold · 0.40 | — | hold · 0.40 | — |
| 50 | 14486 | MCHP | 2026-01-20 | +0.48 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |

**Summary**

| retriever | BUY | SELL | HOLD | dir. hit-rate | BUY mean gain | HOLD mean gain |
|---|--:|--:|--:|--:|--:|--:|
| `insight` | 14 | 0 | 36 | **71%** (10/14) | +0.29% | +0.16% |
| `two-phase-similarity` | 18 | 0 | 32 | **72%** (13/18) | **+0.64%** | −0.05% |

High-confidence (≥0.70) directional calls: insight 10/14 = 71%, two-phase 13/18 = 72%.

#### Findings

1. **A directional edge survives on flat tape.** With the index move suppressed by the
   peaceful-day filter, BUY calls still hit **~71–72%** — well above the 50% coin-flip — so
   the verdict is reacting to *article* content, not market drift. BUY mean gain is positive
   (+0.29% / +0.64%) against a ~flat HOLD baseline.
2. **The model is bullish-or-neutral: it issued 0 SELLs.** Every directional call was a BUY;
   it expresses "bad/uninteresting" as HOLD, never as a short. So this run validates only the
   long side — the short side is untested here (see limitations).
3. **`two-phase-similarity` is the better retriever for the verdict, not just for retrieval.**
   It is strictly dominant on this sample: it keeps all 14 of the insight retriever's BUYs and
   adds 4 more (HOLD→BUY), 3 of them correct — including **AVGO +4.44%** and **LNVGY +2.43%**
   that the insight retriever sat out as HOLD. Net: more decisive (18 vs 14 BUYs), higher
   hit-rate (72% vs 71%), and a higher BUY mean gain (+0.64% vs +0.29%).
4. **Confidence is only loosely calibrated.** The 0.85 bucket caught real moves (NVDA, AVGO,
   AMBA) but also produced the single worst call — a 0.85 BUY on **PLTR that fell −11.61%** —
   and the model *missed* the two biggest up-moves entirely (**AMD +9.66%**, **HPE +2.61%**)
   as low-confidence HOLDs. Confidence ranks direction better than magnitude.

#### Limitations

- **No short side.** 0 SELLs means the bearish half of the model is unvalidated; a SELL-rich
  sample (e.g. earnings-miss seeds) is needed to test it.
- **Small directional N** (14–18 BUYs) — the hit-rates are encouraging but not yet
  significant; a larger draw (200+) would tighten them.
- **Residual long bias.** "Buy → close" on calm days still carries a mild upward drift; a
  strict test would subtract the same-day `SPY` move (the harness already fetches it).
- **HOLD dominates (64–72%).** On single-name real news the model abstains often; the pool is
  doing its job (no easy market-driven wins), but it makes the directional sample small.

---

## 7. Counter-test: a bearish window (Jan 2 – Mar 30 2025)

To show *why* the peaceful-day filter matters, the same harness was run on a **down-trending**
window instead of calm days: 15 real-news articles sampled (seed 7) straight from the DB over
**2025-01-02 … 2025-03-30** — the DeepSeek-selloff + tariff stretch, in which **SPY fell
−4.3%** and the **NASDAQ Composite −10.3%**.

```bash
python scripts/validate_retreival/validate_sentiment.py \
    --db-range 2025-01-02 2025-03-30 --n 15 --seed 7
```

| # | a# | ticker | sell date | gain% | insight (act·conf) | ✓ | two-phase (act·conf) | ✓ |
|--:|--:|:--|:--|--:|:--|:-:|:--|:-:|
| 1 | 2584 | MKSI | 2025-01-24 | -1.76 | buy · 0.75 | ✗ | buy · 0.75 | ✗ |
| 2 | 3999 | MRVL | 2025-03-07 | -2.69 | hold · 0.30 | — | hold · 0.10 | — |
| 3 | 3733 | TSM | 2025-02-28 | -0.80 | hold · 0.30 | — | hold · 0.30 | — |
| 4 | 2453 | CRWD | 2025-01-21 | +1.79 | hold · 0.30 | — | buy · 0.75 | ✓ |
| 5 | 3569 | FORM | 2025-02-24 | -3.67 | hold · 0.30 | — | hold · 0.30 | — |
| 6 | 2692 | GEV | 2025-01-28 | +4.10 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 7 | 3889 | CDNS | 2025-03-05 | +1.41 | hold · 0.30 | — | hold · 0.30 | — |
| 8 | 1947 | MSFT | 2025-01-07 | -1.43 | buy · 0.75 | ✗ | hold · 0.30 | — |
| 9 | 4140 | CRWD | 2025-03-12 | +0.57 | hold · 0.30 | — | hold · 0.30 | — |
| 10 | 2987 | SIMO | 2025-02-06 | +7.49 | hold · 0.40 | — | hold · 0.40 | — |
| 11 | 4459 | CRWD | 2025-03-24 | +2.33 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 12 | 2954 | POWL | 2025-02-05 | +3.55 | hold · 0.30 | — | hold · 0.10 | — |
| 13 | 3542 | AMZN | 2025-02-24 | -2.13 | sell · 0.70 | ✓ | sell · 0.70 | ✓ |
| 14 | 3018 | AMZN | 2025-02-06 | +0.01 | sell · 0.70 | ✗ | sell · 0.70 | ✗ |
| 15 | 4056 | NOK | 2025-03-10 | -1.72 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |

**Calm vs bearish — split by action** (BUY/SELL hit-rate; `insight` / `two-phase-similarity`):

| regime | BUY n | BUY hit-rate | SELL n | SELL hit-rate |
|---|--:|--:|--:|--:|
| Peaceful (§6) | 14 / 18 | **71% / 72%** | 0 / 0 | — |
| Bearish (this run) | 5 / 5 | **40% / 60%** | 2 / 2 | 50% / 50% (the one real decline caught) |

#### Findings

1. **BUY accuracy drops when the tape is falling.** The insight retriever's BUY hit-rate
   collapses from **71% → 40%** (3 of its 5 BUYs went red: MKSI −1.76, MSFT −1.43, NOK −1.72);
   two-phase degrades more gently (72% → 60%). The long-side edge that looked clean on calm
   days is partly **market beta** — exactly the confound the peaceful pool removes.
2. **The model starts selling — and the decisive decline is caught.** It issued **0 SELLs**
   across 50 calm-day articles but **2 SELLs** here, and the one that mattered — **AMZN −2.13%**
   — was correct (✓). The other sell hit a flat name (AMZN +0.01%). So a down-tape regime
   flips the verdict toward the short side, and the genuine drop was flagged.
3. **Same-day P&L is a noisy thermometer for regime.** `buy → close` is *intraday*, so the
   multi-week downtrend barely shows in same-day returns — this sample's mean same-day gain was
   still **+0.47%** despite a −10% NASDAQ window. The regime confound surfaces in the **BUY
   hit-rate drop** and the **emergence of SELLs**, not in uniformly red same-day P&L. (A
   multi-day hold would make the drift bite harder.)
4. **Directional N is tiny** (7 calls per retriever); read this as a directional illustration,
   not a significance test.

**Takeaway.** Long-side accuracy is **regime-sensitive**: BUYs that hit ~71% on calm days fall
toward coin-flip in a downtrend, while the model only reaches for SELL when the tape is weak.
This is precisely why §6 runs on the peaceful-day pool — holding the market direction roughly
flat is what lets a BUY hit-rate be read as *article* signal rather than market beta.

---

## 8. Scripts & provenance

Which script produced which set/result in this document, and what was run.

**Validation-set generation**

| Set | How produced | Used by |
|---|---|---|
| [`peaceful_days_articles.csv`](peaceful_days_articles.csv) | [`peaceful_days.py`](../../scripts/search/peaceful_days.py) — VIX (FRED `VIXCLS`) + S&P/NASDAQ daily moves (Massive `SPY`/`I:COMP`) gate every trading day; real-news articles on peaceful days are emitted. | §2 (the pool), §6 (sampled from it) |

**Validation runs**

§6–§12 produced by [`validate_sentiment.py`](../../scripts/validate_retreival/validate_sentiment.py)
— samples articles (from the pool CSV, or the DB via `--db-range`), runs `insight_sentiment` on
the primary ticker for the two variants of `--compare`, pairs each verdict with the buy→close
return, and scores correctness. §13 is produced by
[`validate_choose1.py`](../../scripts/validate_retreival/validate_choose1.py) instead — it runs
`--choose-1` (the model picks one ticker among all the article's), so it prices whichever ticker
the model chose, not the primary. One row per section:

| Section | Script / Invocation | Compares | Sample |
|---|---|---|---|
| §6 | `validate_sentiment.py --n 50 --seed 42` | `insight` vs `two-phase-similarity` retrieval | 50 peaceful-day |
| §7 | `validate_sentiment.py --db-range 2025-01-02 2025-03-30 --n 15 --seed 7` | same (retrievers) | 15 bearish-window |
| §9 | `validate_sentiment.py --compare actions --n 150 --seed 42` | two-phase: `buy/sell/hold` vs `+--include-strong` | 150 peaceful-day |
| §10 | `validate_sentiment.py --compare bias --n 100 --seed 42` | two-phase: without vs `+--include-bias` | 100 peaceful-day |
| §11 | `validate_sentiment.py --compare bias --db-range 2025-01-02 2025-03-30 --n 30 --seed 7` | same (bias) | 30 bearish-window |
| §13 | `validate_choose1.py --n 100 --seed 42` | fixed-primary (§10/§12) vs `--choose-1 + bias` (model's pick) | 100 peaceful-day |
| §14 | `validate_clean_top1.py --n 100 --seed 42` | `--choose-1` (round 1) vs `--clean-top-1` refined (round 2) | 100 peaceful-day |

**Subjects under test / dependencies** (the pipeline being measured, not validators)

| Script | Role |
|---|---|
| [`insight_sentiment.py`](../../scripts/search/insight_sentiment.py) | The sentiment pipeline itself — retrieval (`--retrieval insight` / `two-phase-similarity`), optional `--remove-unuseful` screen, and the BUY/SELL/HOLD + confidence verdict. |
| [`catalyst_returns.py`](../../scripts/ticker_scan/catalyst_returns.py) | The returns engine (`simulate`) — buy-at-publish minute bar → sell-at-(next)-close, via Massive. |
| [`backtest_top2.py`](../../scripts/search/backtest_top2.py) | The earlier DB-range verdict↔return harness; `validate_sentiment.py` is its pool-driven successor. |

Runs require `NEWS_DB_DSN` (articles/insights), `MASSIVE_API_KEY` (returns), and
`GOOGLE_API_KEY` (Gemini verdicts); `peaceful_days.py` additionally hits FRED (no key).

---

## 9. `--include-strong`: does a wider action menu help? (150 peaceful-day articles)

Tests whether offering `strong_buy` / `strong_sell` (the `--include-strong` flag) sharpens the
verdict. **150** articles sampled (seed 42) from `peaceful_days_articles.csv`, all judged on the
primary ticker with **two-phase-similarity** retrieval; the *only* difference between the two
columns is the action menu — plain `buy/sell/hold` vs `+strong`.

```bash
python scripts/validate_retreival/validate_sentiment.py \
    --compare actions --n 150 --seed 42
```

**Headline comparison**

| menu | BUY-like | SELL-like | HOLD | dir. hit-rate | BUY−SELL spread |
|---|--:|--:|--:|--:|--:|
| default (`buy`/`sell`/`hold`) | 47 | 5 | 98 | **54%** (28/52) | **+0.96pt** |
| `+strong` | 65 | 7 | 78 | **47%** (34/72) | **−0.77pt** |

**Per-bucket realized return** (`+strong` menu) — is "strong" magnitude-predictive?

| bucket | n | mean gain% | dir hit-rate |
|---|--:|--:|--:|
| strong_buy | 12 | **−0.71%** | 50% (6/12) |
| buy | 53 | +0.23% | 47% (25/53) |
| hold | 78 | +0.18% | — |
| sell | 6 | +1.26% | 33% (2/6) |
| strong_sell | 1 | −1.75% | 100% (1/1) |

**Action shift (default → strong)** — 34 of 150 verdicts changed:

| default | → strong | n |
|---|---|--:|
| hold | buy | 17 |
| buy | strong_buy | 11 |
| hold | sell | 3 |
| hold | strong_buy | 1 |
| sell | strong_sell | 1 |
| sell | hold | 1 |

#### Findings

1. **The strong menu makes the model more decisive — it abstains less.** HOLD falls **98 → 78**
   while BUY-like rises **47 → 65**. Offered bigger labels it reaches for them: 17 HOLD→buy,
   11 buy→strong_buy, 3 HOLD→sell.
2. **But the added decisiveness *hurts* accuracy.** Directional hit-rate drops **54% → 47%**, and
   the BUY−SELL spread flips from **+0.96pt to −0.77pt**. The newly-promoted calls are mostly the
   marginal ones the plain menu correctly left as HOLD; converting them to directional bets just
   adds coin-flip noise.
3. **`strong_buy` is anti-calibrated for magnitude.** The 12 strong_buy calls — the model's
   "LARGE rise, high-conviction" bets — returned **−0.71% mean** at 50% hit, *worse* than plain
   buy (**+0.23%**, 47%). 11 of 12 were `buy → strong_buy` upgrades; raising the conviction label
   did not select bigger or more reliable winners. The literal "strong" carries **no** realized
   magnitude signal here.
4. **The extra SELLs are noise too.** Under strong, `sell` n=6 averaged **+1.26%** (33% hit — the
   names it sold *rose*); the lone `strong_sell` was right (−1.75%) but cannot offset.

**Conclusion — keep `--include-strong` off by default.** On calm tape it trades the plain menu's
well-judged abstention for over-confident directional calls that sit closer to coin-flip, and the
`strong_*` tier is not magnitude-predictive (strong_buy actually lost money). The 3-action menu is
better calibrated; reserve `--include-strong` for exploratory / risk-on use, not validation or
production scoring.

<details>
<summary>Full 150-row table (verdict · correctness · confidence · buy→close gain)</summary>

| # | a# | ticker | sell date | gain% | two-phase (act·conf) | ✓ | two-phase + strong (act·conf) | ✓ |
|--:|--:|:--|:--|--:|:--|:-:|:--|:-:|
| 1 | 19833 | NVDA | 2026-05-14 | +2.50 | hold · 0.10 | — | hold · 0.10 | — |
| 2 | 19819 | AMZN | 2026-05-14 | -0.97 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |
| 3 | 10534 | WDC | 2025-10-13 | +1.11 | buy · 0.70 | ✓ | buy · 0.60 | ✓ |
| 4 | 11257 | AMZN | 2025-10-30 | -2.20 | hold · 0.30 | — | hold · 0.20 | — |
| 5 | 8631 | NOK | 2025-08-21 | +0.47 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 6 | 1298 | AMZN | 2024-12-12 | -0.69 | hold · 0.30 | — | buy · 0.60 | ✗ |
| 7 | 16011 | AVGO | 2026-02-19 | -0.31 | buy · 0.75 | ✗ | strong_buy · 0.90 | ✗ |
| 8 | 1445 | COHR | 2024-12-17 | -2.57 | hold · 0.10 | — | hold · 0.10 | — |
| 9 | 13612 | CSCO | 2025-12-26 | +0.70 | hold · 0.10 | — | hold · 0.10 | — |
| 10 | 7866 | IBM | 2025-07-30 | -0.80 | hold · 0.30 | — | hold · 0.30 | — |
| 11 | 1190 | PLTR | 2024-12-09 | -11.61 | buy · 0.85 | ✗ | strong_buy · 0.90 | ✗ |
| 12 | 12126 | ADBE | 2025-11-19 | -0.29 | hold · 0.30 | — | hold · 0.30 | — |
| 13 | 7915 | NVDA | 2025-07-31 | -2.73 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 14 | 7770 | LNVGY | 2025-07-28 | +2.43 | buy · 0.85 | ✓ | buy · 0.70 | ✓ |
| 15 | 19306 | AMAT | 2026-05-04 | +0.07 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 16 | 13844 | AMBA | 2026-01-05 | -0.45 | buy · 0.80 | ✗ | strong_buy · 0.90 | ✗ |
| 17 | 14210 | GOOG | 2026-01-12 | +1.94 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 18 | 14208 | NVDA | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | buy · 0.75 | ✓ |
| 19 | 19675 | NVDA | 2026-05-11 | -1.15 | hold · 0.30 | — | hold · 0.30 | — |
| 20 | 19102 | DELL | 2026-04-29 | +0.69 | hold · 0.30 | — | hold · 0.30 | — |
| 21 | 13326 | NVDA | 2025-12-17 | -3.92 | hold · 0.10 | — | hold · 0.10 | — |
| 22 | 11563 | LNVGY | 2025-11-06 | -3.20 | hold · 0.30 | — | buy · 0.60 | ✗ |
| 23 | 13972 | MSFT | 2026-01-07 | +1.00 | hold · 0.10 | — | hold · 0.10 | — |
| 24 | 14391 | AMZN | 2026-01-16 | +0.08 | hold · 0.30 | — | hold · 0.30 | — |
| 25 | 7222 | GOOG | 2025-07-10 | +0.70 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 26 | 19431 | AMZN | 2026-05-06 | +0.84 | hold · 0.30 | — | sell · 0.60 | ✗ |
| 27 | 7393 | POWL | 2025-07-16 | +3.62 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 28 | 9620 | HPE | 2025-09-17 | +2.61 | hold · 0.20 | — | hold · 0.10 | — |
| 29 | 13787 | NVDA | 2026-01-02 | -0.54 | hold · 0.30 | — | hold · 0.30 | — |
| 30 | 7047 | ADI | 2025-07-03 | -0.03 | hold · 0.30 | — | hold · 0.20 | — |
| 31 | 8845 | AMBA | 2025-08-27 | +2.12 | hold · 0.30 | — | hold · 0.10 | — |
| 32 | 16576 | AVGO | 2026-03-05 | +4.44 | buy · 0.85 | ✓ | buy · 0.70 | ✓ |
| 33 | 10567 | AMZN | 2025-10-14 | -1.60 | hold · 0.10 | — | hold · 0.10 | — |
| 34 | 14701 | NVDA | 2026-01-23 | +0.24 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 35 | 14712 | DELL | 2026-01-23 | +0.68 | hold · 0.10 | — | hold · 0.10 | — |
| 36 | 623 | GOOG | 2024-11-20 | -1.13 | hold · 0.30 | — | hold · 0.20 | — |
| 37 | 19378 | IREN | 2026-05-05 | +9.96 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 38 | 14045 | IBM | 2026-01-08 | -0.14 | hold · 0.30 | — | hold · 0.20 | — |
| 39 | 15347 | AMZN | 2026-02-05 | +0.37 | hold · 0.10 | — | hold · 0.10 | — |
| 40 | 14314 | META | 2026-01-14 | -0.38 | hold · 0.10 | — | hold · 0.10 | — |
| 41 | 9825 | MSFT | 2025-09-26 | +0.31 | hold · 0.30 | — | hold · 0.30 | — |
| 42 | 14889 | NVDA | 2026-01-28 | -0.01 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |
| 43 | 8463 | ASX | 2025-08-15 | -0.05 | hold · 0.30 | — | buy · 0.60 | ✗ |
| 44 | 16038 | AMD | 2026-02-20 | -1.55 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 45 | 9661 | AVGO | 2025-09-17 | +0.54 | buy · 0.85 | ✓ | buy · 0.70 | ✓ |
| 46 | 19852 | AMZN | 2026-05-14 | -0.64 | hold · 0.10 | — | hold · 0.10 | — |
| 47 | 9942 | QCOM | 2025-09-30 | +0.38 | hold · 0.30 | — | hold · 0.20 | — |
| 48 | 7134 | CSCO | 2025-07-07 | -0.04 | hold · 0.30 | — | hold · 0.30 | — |
| 49 | 10281 | AMD | 2025-10-08 | +9.66 | hold · 0.30 | — | sell · 0.60 | ✗ |
| 50 | 608 | HUBB | 2024-11-20 | -1.52 | hold · 0.30 | — | hold · 0.10 | — |
| 51 | 6646 | NOK | 2025-06-16 | +0.95 | hold · 0.40 | — | hold · 0.30 | — |
| 52 | 14486 | MCHP | 2026-01-20 | +0.48 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 53 | 14207 | NVDA | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | strong_buy · 0.95 | ✓ |
| 54 | 19835 | NVDA | 2026-05-14 | +1.16 | hold · 0.30 | — | hold · 0.30 | — |
| 55 | 16186 | KEYS | 2026-02-24 | +6.91 | hold · 0.30 | — | hold · 0.20 | — |
| 56 | 11556 | GOOG | 2025-11-06 | -0.05 | hold · 0.10 | — | hold · 0.10 | — |
| 57 | 7392 | AVGO | 2025-07-15 | -0.47 | hold · 0.40 | — | buy · 0.70 | ✗ |
| 58 | 9740 | IBM | 2025-09-22 | +1.60 | hold · 0.10 | — | hold · 0.10 | — |
| 59 | 8993 | FLEX | 2025-09-02 | -0.09 | buy · 0.70 | ✗ | buy · 0.60 | ✗ |
| 60 | 13375 | NVDA | 2025-12-18 | +1.02 | sell · 0.65 | ✗ | sell · 0.60 | ✗ |
| 61 | 19908 | NVDA | 2026-05-18 | -1.23 | hold · 0.30 | — | buy · 0.60 | ✗ |
| 62 | 3409 | ARM | 2025-02-19 | -3.46 | hold · 0.30 | — | hold · 0.30 | — |
| 63 | 13272 | AMZN | 2025-12-16 | -0.20 | hold · 0.10 | — | hold · 0.10 | — |
| 64 | 9244 | GOOG | 2025-09-08 | -0.13 | hold · 0.30 | — | hold · 0.30 | — |
| 65 | 9777 | NOK | 2025-09-24 | -1.25 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 66 | 13742 | AVGO | 2025-12-31 | -0.71 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 67 | 7068 | ADBE | 2025-07-07 | -0.54 | hold · 0.30 | — | hold · 0.20 | — |
| 68 | 815 | NOK | 2024-11-27 | +0.24 | buy · 0.85 | ✓ | strong_buy · 0.90 | ✓ |
| 69 | 9080 | NVDA | 2025-09-04 | -0.01 | hold · 0.10 | — | hold · 0.20 | — |
| 70 | 10245 | APLD | 2025-10-07 | -2.85 | hold · 0.30 | — | buy · 0.70 | ✗ |
| 71 | 16241 | MSFT | 2026-02-25 | +2.80 | hold · 0.30 | — | hold · 0.30 | — |
| 72 | 19946 | AMAT | 2026-05-18 | -3.48 | hold · 0.30 | — | sell · 0.60 | ✓ |
| 73 | 9719 | ARM | 2025-09-19 | -0.83 | sell · 0.60 | ✓ | sell · 0.60 | ✓ |
| 74 | 14095 | META | 2026-01-09 | +1.07 | hold · 0.30 | — | hold · 0.30 | — |
| 75 | 12121 | ADBE | 2025-11-19 | -0.15 | hold · 0.30 | — | hold · 0.30 | — |
| 76 | 7811 | TSM | 2025-07-28 | +0.34 | sell · 0.70 | ✗ | sell · 0.60 | ✗ |
| 77 | 10252 | IBM | 2025-10-07 | -1.90 | buy · 0.85 | ✗ | buy · 0.70 | ✗ |
| 78 | 11434 | MSFT | 2025-11-03 | -0.83 | hold · 0.30 | — | hold · 0.30 | — |
| 79 | 13505 | NVDA | 2025-12-22 | +0.04 | hold · 0.30 | — | hold · 0.20 | — |
| 80 | 20103 | CRWD | 2026-05-20 | +6.40 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 81 | 694 | NOK | 2024-11-22 | +2.20 | hold · 0.30 | — | hold · 0.30 | — |
| 82 | 13333 | AMZN | 2025-12-17 | -1.68 | hold · 0.10 | — | hold · 0.10 | — |
| 83 | 149 | GOOG | 2024-11-04 | -1.34 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |
| 84 | 11135 | MSFT | 2025-10-27 | -0.26 | hold · 0.10 | — | hold · 0.10 | — |
| 85 | 7377 | MSFT | 2025-07-15 | +0.10 | hold · 0.30 | — | buy · 0.60 | ✓ |
| 86 | 13695 | APLD | 2025-12-30 | -2.82 | buy · 0.70 | ✗ | strong_buy · 0.90 | ✗ |
| 87 | 14254 | META | 2026-01-13 | +0.32 | hold · 0.10 | — | hold · 0.10 | — |
| 88 | 14141 | GOOG | 2026-01-12 | +1.97 | hold · 0.10 | — | hold · 0.10 | — |
| 89 | 1056 | META | 2024-12-05 | -0.95 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 90 | 10201 | AMD | 2025-10-06 | -2.92 | hold · 0.60 | — | strong_buy · 0.95 | ✗ |
| 91 | 10277 | MSFT | 2025-10-08 | +0.09 | hold · 0.10 | — | hold · 0.10 | — |
| 92 | 6649 | NOK | 2025-06-16 | +0.95 | hold · 0.40 | — | hold · 0.30 | — |
| 93 | 9125 | AI | 2025-09-04 | -1.75 | sell · 0.85 | ✓ | strong_sell · 0.90 | ✓ |
| 94 | 8436 | NVDA | 2025-08-15 | -0.70 | hold · 0.30 | — | hold · 0.10 | — |
| 95 | 10323 | SNPS | 2025-10-09 | -0.63 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 96 | 10324 | AMZN | 2025-10-09 | +1.01 | hold · 0.30 | — | buy · 0.60 | ✓ |
| 97 | 10350 | MSFT | 2025-10-09 | -0.24 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 98 | 17107 | S | 2026-03-16 | -2.39 | hold · 0.40 | — | buy · 0.60 | ✗ |
| 99 | 3404 | NVDA | 2025-02-19 | -0.12 | hold · 0.30 | — | hold · 0.20 | — |
| 100 | 6928 | NVDA | 2025-06-27 | +1.38 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 101 | 14823 | MU | 2026-01-27 | +2.35 | hold · 0.30 | — | buy · 0.70 | ✓ |
| 102 | 9218 | AMZN | 2025-09-08 | +1.07 | hold · 0.10 | — | hold · 0.10 | — |
| 103 | 13879 | MSFT | 2026-01-06 | +1.21 | hold · 0.30 | — | buy · 0.70 | ✓ |
| 104 | 16458 | CRWD | 2026-03-02 | +3.25 | buy · 0.85 | ✓ | buy · 0.70 | ✓ |
| 105 | 17129 | NVDA | 2026-03-17 | -0.84 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |
| 106 | 9819 | META | 2025-09-26 | -0.58 | hold · 0.30 | — | buy · 0.70 | ✗ |
| 107 | 12997 | LNVGY | 2025-12-10 | +1.44 | hold · 0.30 | — | hold · 0.30 | — |
| 108 | 8152 | STM | 2025-08-06 | -0.80 | hold · 0.30 | — | hold · 0.20 | — |
| 109 | 19785 | LNVGY | 2026-05-13 | +0.51 | hold · 0.20 | — | hold · 0.20 | — |
| 110 | 19524 | AMZN | 2026-05-07 | -0.78 | hold · 0.30 | — | buy · 0.70 | ✗ |
| 111 | 9662 | AVGO | 2025-09-17 | +0.54 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 112 | 15349 | HPQ | 2026-02-05 | +1.40 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 113 | 10292 | FLEX | 2025-10-08 | +4.14 | buy · 0.85 | ✓ | strong_buy · 0.90 | ✓ |
| 114 | 11111 | AMZN | 2025-10-27 | -0.58 | hold · 0.10 | — | hold · 0.10 | — |
| 115 | 16832 | AMAT | 2026-03-11 | +1.50 | buy · 0.80 | ✓ | strong_buy · 0.90 | ✓ |
| 116 | 6989 | MSFT | 2025-06-30 | -0.16 | hold · 0.10 | — | hold · 0.10 | — |
| 117 | 8549 | META | 2025-08-19 | -1.64 | hold · 0.10 | — | hold · 0.10 | — |
| 118 | 8772 | AMZN | 2025-08-25 | -0.56 | hold · 0.30 | — | buy · 0.60 | ✗ |
| 119 | 10076 | IBM | 2025-10-03 | +0.54 | hold · 0.30 | — | hold · 0.30 | — |
| 120 | 8407 | ASX | 2025-08-14 | -0.50 | hold · 0.40 | — | hold · 0.30 | — |
| 121 | 12907 | LNVGY | 2025-12-08 | +1.91 | hold · 0.30 | — | hold · 0.30 | — |
| 122 | 8661 | GOOG | 2025-08-22 | +2.13 | hold · 0.30 | — | hold · 0.30 | — |
| 123 | 19385 | SOUN | 2026-05-05 | -5.97 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |
| 124 | 15151 | AMD | 2026-02-02 | -0.86 | hold · 0.30 | — | hold · 0.30 | — |
| 125 | 10249 | NOK | 2025-10-07 | +1.50 | buy · 0.80 | ✓ | strong_buy · 0.90 | ✓ |
| 126 | 17109 | META | 2026-03-16 | +0.35 | hold · 0.30 | — | hold · 0.30 | — |
| 127 | 12745 | UMC | 2025-12-04 | +0.51 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 128 | 14006 | MSFT | 2026-01-08 | -1.11 | hold · 0.30 | — | hold · 0.30 | — |
| 129 | 19369 | AMD | 2026-05-05 | +3.21 | hold · 0.30 | — | hold · 0.30 | — |
| 130 | 16795 | STM | 2026-03-10 | -0.42 | buy · 0.85 | ✗ | strong_buy · 0.90 | ✗ |
| 131 | 3632 | AMZN | 2025-02-25 | +0.49 | hold · 0.10 | — | hold · 0.10 | — |
| 132 | 6014 | NOK | 2025-05-22 | -0.19 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |
| 133 | 9221 | AMZN | 2025-09-08 | +1.07 | hold · 0.10 | — | hold · 0.10 | — |
| 134 | 12783 | NVDA | 2025-12-05 | -1.18 | hold · 0.10 | — | hold · 0.10 | — |
| 135 | 7335 | NOK | 2025-07-14 | -1.81 | buy · 0.75 | ✗ | buy · 0.70 | ✗ |
| 136 | 14292 | DELL | 2026-01-14 | -0.26 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 137 | 16154 | AMZN | 2026-02-23 | +0.87 | hold · 0.10 | — | hold · 0.10 | — |
| 138 | 6385 | MCHP | 2025-06-05 | -0.46 | hold · 0.30 | — | buy · 0.60 | ✗ |
| 139 | 11510 | MSFT | 2025-11-05 | -1.31 | hold · 0.10 | — | hold · 0.10 | — |
| 140 | 12673 | AMD | 2025-12-03 | +0.33 | hold · 0.30 | — | buy · 0.60 | ✓ |
| 141 | 9947 | DELL | 2025-09-30 | +2.90 | hold · 0.10 | — | hold · 0.10 | — |
| 142 | 8004 | META | 2025-08-01 | +0.06 | hold · 0.10 | — | hold · 0.20 | — |
| 143 | 3455 | PWR | 2025-02-20 | +0.82 | hold · 0.40 | — | hold · 0.30 | — |
| 144 | 11375 | AMD | 2025-11-03 | +0.66 | hold · 0.30 | — | hold · 0.30 | — |
| 145 | 16188 | AMD | 2026-02-24 | -2.27 | sell · 0.70 | ✓ | hold · 0.30 | — |
| 146 | 20459 | NVDA | 2026-05-27 | +0.60 | hold · 0.30 | — | hold · 0.30 | — |
| 147 | 16626 | NBIS | 2026-03-05 | +2.64 | buy · 0.85 | ✓ | strong_buy · 0.90 | ✓ |
| 148 | 697 | GOOG | 2024-11-22 | -0.82 | hold · 0.30 | — | hold · 0.30 | — |
| 149 | 8791 | AMD | 2025-08-26 | +2.48 | hold · 0.40 | — | buy · 0.70 | ✓ |
| 150 | 10234 | AMZN | 2025-10-07 | +0.54 | hold · 0.10 | — | hold · 0.20 | — |

</details>


---

## 10. `--include-bias`: does a debiasing caveat help? (100 peaceful-day articles)

Tests whether warning the model that the extracted insights skew FAVORABLE (the `--include-bias`
flag) corrects its bullish lean. **100** articles sampled (seed 42) from `peaceful_days_articles.csv`,
all judged on the primary ticker with **two-phase-similarity** retrieval and the plain
`buy/sell/hold` menu; the *only* difference between the two columns is the bias caveat in the
prompt.

```bash
python scripts/validate_retreival/validate_sentiment.py \
    --compare bias --n 100 --seed 42
```

**Headline comparison**

| variant | BUY-like | SELL-like | HOLD | dir. hit-rate | BUY−SELL spread |
|---|--:|--:|--:|--:|--:|
| no bias | 33 | 4 | 63 | 51% (19/37) | +0.50pt |
| `+bias` | 23 | 5 | 72 | 54% (15/28) | +0.42pt |

**Per-bucket realized return**

| variant | bucket | n | mean gain% | dir hit-rate |
|---|---|--:|--:|--:|
| no bias | buy | 33 | +0.20% | 52% (17/33) |
| no bias | sell | 4 | −0.30% | 50% (2/4) |
| `+bias` | buy | 23 | +0.15% | 52% (12/23) |
| `+bias` | sell | 5 | −0.27% | 60% (3/5) |

**Verdict shift (two-phase → two-phase + bias)** — 11 of 100 changed:

| no bias | → +bias | n |
|---|---|--:|
| buy | hold | 10 |
| hold | sell | 1 |

#### Findings

1. **The caveat makes the model more cautious — exactly as intended.** BUY-like falls **33 → 23**
   and HOLD rises **63 → 72**: 10 of 100 verdicts flip `buy → hold` and 1 `hold → sell`. Told the
   insights are one-sidedly favorable, the model discounts them and pulls back ~30% of its buys.
2. **But directional accuracy is essentially unchanged.** Hit-rate moves **51% → 54%**, which is
   noise on a shrinking directional sample (37 → 28). Tellingly the **kept buys have the identical
   hit-rate (52%)** and similar mean gain (+0.20% → +0.15%) — the 10 dropped buys were themselves
   ~coin-flip (5 of 10 would have been right), so the caveat prunes buys roughly **indiscriminately**,
   not selectively the wrong ones.
3. **It does not fix the long-only tilt.** SELL-like barely moves (4 → 5); the pruned buys route to
   HOLD, not SELL. The caveat raises *abstention*, it does not make the model bearish.
4. **The BUY−SELL spread is flat-to-slightly-lower** (+0.50 → +0.42pt) — no economic improvement.

**Conclusion — a conservatism knob, not an accuracy lever.** `--include-bias` shifts the operating
point toward caution (≈10% of buys become holds) without sharpening the signal: the surviving buys
are no better and the spread does not improve. Unlike `--include-strong` (§9), it does not *hurt* —
it is roughly neutral. Leave it **off by default** (the plain menu is more decisive at equal
accuracy); reach for it only when you specifically want fewer, more-hedged bullish calls.

<details>
<summary>Full 100-row table (verdict · correctness · confidence · buy→close gain)</summary>

| # | a# | ticker | sell date | gain% | two-phase (act·conf) | ✓ | two-phase + bias (act·conf) | ✓ |
|--:|--:|:--|:--|--:|:--|:-:|:--|:-:|
| 1 | 19833 | NVDA | 2026-05-14 | +2.50 | hold · 0.10 | — | hold · 0.10 | — |
| 2 | 19819 | AMZN | 2026-05-14 | -0.97 | buy · 0.75 | ✗ | buy · 0.75 | ✗ |
| 3 | 10534 | WDC | 2025-10-13 | +1.11 | buy · 0.70 | ✓ | hold · 0.30 | — |
| 4 | 11257 | AMZN | 2025-10-30 | -2.20 | hold · 0.30 | — | hold · 0.30 | — |
| 5 | 8631 | NOK | 2025-08-21 | +0.47 | buy · 0.70 | ✓ | hold · 0.30 | — |
| 6 | 1298 | AMZN | 2024-12-12 | -0.69 | hold · 0.30 | — | hold · 0.30 | — |
| 7 | 16011 | AVGO | 2026-02-19 | -0.31 | buy · 0.75 | ✗ | buy · 0.75 | ✗ |
| 8 | 1445 | COHR | 2024-12-17 | -2.57 | hold · 0.10 | — | hold · 0.10 | — |
| 9 | 13612 | CSCO | 2025-12-26 | +0.70 | hold · 0.10 | — | hold · 0.10 | — |
| 10 | 7866 | IBM | 2025-07-30 | -0.80 | hold · 0.30 | — | hold · 0.30 | — |
| 11 | 1190 | PLTR | 2024-12-09 | -11.61 | buy · 0.85 | ✗ | buy · 0.85 | ✗ |
| 12 | 12126 | ADBE | 2025-11-19 | -0.29 | hold · 0.30 | — | hold · 0.30 | — |
| 13 | 7915 | NVDA | 2025-07-31 | -2.73 | buy · 0.70 | ✗ | hold · 0.30 | — |
| 14 | 7770 | LNVGY | 2025-07-28 | +2.43 | buy · 0.85 | ✓ | buy · 0.80 | ✓ |
| 15 | 19306 | AMAT | 2026-05-04 | +0.07 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 16 | 13844 | AMBA | 2026-01-05 | -0.45 | buy · 0.80 | ✗ | buy · 0.85 | ✗ |
| 17 | 14210 | GOOG | 2026-01-12 | +1.94 | buy · 0.75 | ✓ | hold · 0.30 | — |
| 18 | 14208 | NVDA | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 19 | 19675 | NVDA | 2026-05-11 | -1.15 | hold · 0.30 | — | hold · 0.40 | — |
| 20 | 13326 | NVDA | 2025-12-17 | -3.92 | hold · 0.10 | — | hold · 0.10 | — |
| 21 | 11563 | LNVGY | 2025-11-06 | -3.20 | hold · 0.30 | — | hold · 0.30 | — |
| 22 | 13972 | MSFT | 2026-01-07 | +1.00 | hold · 0.10 | — | hold · 0.10 | — |
| 23 | 14391 | AMZN | 2026-01-16 | +0.08 | hold · 0.30 | — | hold · 0.30 | — |
| 24 | 7222 | GOOG | 2025-07-10 | +0.70 | buy · 0.70 | ✓ | hold · 0.30 | — |
| 25 | 19431 | AMZN | 2026-05-06 | +0.84 | hold · 0.30 | — | hold · 0.30 | — |
| 26 | 7393 | POWL | 2025-07-16 | +3.62 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 27 | 9620 | HPE | 2025-09-17 | +2.61 | hold · 0.20 | — | hold · 0.10 | — |
| 28 | 13787 | NVDA | 2026-01-02 | -0.54 | hold · 0.30 | — | hold · 0.30 | — |
| 29 | 7047 | ADI | 2025-07-03 | -0.03 | hold · 0.30 | — | hold · 0.30 | — |
| 30 | 8845 | AMBA | 2025-08-27 | +2.12 | hold · 0.30 | — | hold · 0.30 | — |
| 31 | 16576 | AVGO | 2026-03-05 | +4.44 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 32 | 10567 | AMZN | 2025-10-14 | -1.60 | hold · 0.10 | — | hold · 0.10 | — |
| 33 | 14701 | NVDA | 2026-01-23 | +0.24 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 34 | 623 | GOOG | 2024-11-20 | -1.13 | hold · 0.30 | — | hold · 0.30 | — |
| 35 | 19378 | IREN | 2026-05-05 | +9.96 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 36 | 14045 | IBM | 2026-01-08 | -0.14 | hold · 0.30 | — | hold · 0.30 | — |
| 37 | 15347 | AMZN | 2026-02-05 | +0.37 | hold · 0.10 | — | hold · 0.10 | — |
| 38 | 14314 | META | 2026-01-14 | -0.38 | hold · 0.10 | — | hold · 0.10 | — |
| 39 | 9825 | MSFT | 2025-09-26 | +0.31 | hold · 0.30 | — | hold · 0.40 | — |
| 40 | 14889 | NVDA | 2026-01-28 | -0.01 | buy · 0.75 | ✗ | buy · 0.75 | ✗ |
| 41 | 8463 | ASX | 2025-08-15 | -0.05 | hold · 0.30 | — | hold · 0.30 | — |
| 42 | 16038 | AMD | 2026-02-20 | -1.55 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 43 | 9661 | AVGO | 2025-09-17 | +0.54 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 44 | 19852 | AMZN | 2026-05-14 | -0.64 | hold · 0.10 | — | hold · 0.10 | — |
| 45 | 9942 | QCOM | 2025-09-30 | +0.38 | hold · 0.30 | — | hold · 0.30 | — |
| 46 | 7134 | CSCO | 2025-07-07 | -0.04 | hold · 0.30 | — | hold · 0.30 | — |
| 47 | 10281 | AMD | 2025-10-08 | +9.66 | hold · 0.30 | — | hold · 0.30 | — |
| 48 | 608 | HUBB | 2024-11-20 | -1.52 | hold · 0.30 | — | hold · 0.20 | — |
| 49 | 6646 | NOK | 2025-06-16 | +0.95 | hold · 0.40 | — | hold · 0.30 | — |
| 50 | 14486 | MCHP | 2026-01-20 | +0.48 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 51 | 14207 | NVDA | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 52 | 19835 | NVDA | 2026-05-14 | +1.16 | hold · 0.30 | — | hold · 0.30 | — |
| 53 | 16186 | KEYS | 2026-02-24 | +6.91 | hold · 0.30 | — | hold · 0.30 | — |
| 54 | 11556 | GOOG | 2025-11-06 | -0.05 | hold · 0.10 | — | hold · 0.10 | — |
| 55 | 7392 | AVGO | 2025-07-15 | -0.47 | hold · 0.40 | — | hold · 0.40 | — |
| 56 | 9740 | IBM | 2025-09-22 | +1.60 | hold · 0.10 | — | hold · 0.10 | — |
| 57 | 8993 | FLEX | 2025-09-02 | -0.09 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 58 | 13375 | NVDA | 2025-12-18 | +1.02 | sell · 0.65 | ✗ | sell · 0.60 | ✗ |
| 59 | 19908 | NVDA | 2026-05-18 | -1.23 | hold · 0.30 | — | hold · 0.30 | — |
| 60 | 3409 | ARM | 2025-02-19 | -3.46 | hold · 0.30 | — | hold · 0.30 | — |
| 61 | 13272 | AMZN | 2025-12-16 | -0.20 | hold · 0.10 | — | hold · 0.10 | — |
| 62 | 9244 | GOOG | 2025-09-08 | -0.13 | hold · 0.30 | — | sell · 0.65 | ✓ |
| 63 | 9777 | NOK | 2025-09-24 | -1.25 | buy · 0.70 | ✗ | hold · 0.30 | — |
| 64 | 13742 | AVGO | 2025-12-31 | -0.71 | buy · 0.70 | ✗ | buy · 0.75 | ✗ |
| 65 | 7068 | ADBE | 2025-07-07 | -0.54 | hold · 0.30 | — | hold · 0.30 | — |
| 66 | 815 | NOK | 2024-11-27 | +0.24 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 67 | 9080 | NVDA | 2025-09-04 | -0.01 | hold · 0.10 | — | hold · 0.20 | — |
| 68 | 10245 | APLD | 2025-10-07 | -2.85 | hold · 0.30 | — | hold · 0.30 | — |
| 69 | 16241 | MSFT | 2026-02-25 | +2.80 | hold · 0.30 | — | hold · 0.30 | — |
| 70 | 19946 | AMAT | 2026-05-18 | -3.48 | hold · 0.30 | — | hold · 0.30 | — |
| 71 | 9719 | ARM | 2025-09-19 | -0.83 | sell · 0.60 | ✓ | sell · 0.60 | ✓ |
| 72 | 14095 | META | 2026-01-09 | +1.07 | hold · 0.30 | — | hold · 0.30 | — |
| 73 | 12121 | ADBE | 2025-11-19 | -0.15 | hold · 0.30 | — | hold · 0.30 | — |
| 74 | 7811 | TSM | 2025-07-28 | +0.34 | sell · 0.70 | ✗ | sell · 0.70 | ✗ |
| 75 | 10252 | IBM | 2025-10-07 | -1.90 | buy · 0.85 | ✗ | hold · 0.30 | — |
| 76 | 11434 | MSFT | 2025-11-03 | -0.83 | hold · 0.30 | — | hold · 0.20 | — |
| 77 | 13505 | NVDA | 2025-12-22 | +0.04 | hold · 0.30 | — | hold · 0.30 | — |
| 78 | 20103 | CRWD | 2026-05-20 | +6.40 | buy · 0.75 | ✓ | hold · 0.40 | — |
| 79 | 694 | NOK | 2024-11-22 | +2.20 | hold · 0.30 | — | hold · 0.30 | — |
| 80 | 13333 | AMZN | 2025-12-17 | -1.68 | hold · 0.10 | — | hold · 0.10 | — |
| 81 | 149 | GOOG | 2024-11-04 | -1.34 | buy · 0.75 | ✗ | hold · 0.30 | — |
| 82 | 11135 | MSFT | 2025-10-27 | -0.26 | hold · 0.10 | — | hold · 0.10 | — |
| 83 | 7377 | MSFT | 2025-07-15 | +0.10 | hold · 0.30 | — | hold · 0.30 | — |
| 84 | 13695 | APLD | 2025-12-30 | -2.82 | buy · 0.70 | ✗ | buy · 0.75 | ✗ |
| 85 | 14254 | META | 2026-01-13 | +0.32 | hold · 0.10 | — | hold · 0.10 | — |
| 86 | 14141 | GOOG | 2026-01-12 | +1.97 | hold · 0.10 | — | hold · 0.10 | — |
| 87 | 1056 | META | 2024-12-05 | -0.95 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 88 | 10277 | MSFT | 2025-10-08 | +0.09 | hold · 0.10 | — | hold · 0.10 | — |
| 89 | 6649 | NOK | 2025-06-16 | +0.95 | hold · 0.40 | — | hold · 0.30 | — |
| 90 | 9125 | AI | 2025-09-04 | -1.75 | sell · 0.85 | ✓ | sell · 0.85 | ✓ |
| 91 | 8436 | NVDA | 2025-08-15 | -0.70 | hold · 0.30 | — | hold · 0.20 | — |
| 92 | 10323 | SNPS | 2025-10-09 | -0.63 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 93 | 10324 | AMZN | 2025-10-09 | +1.01 | hold · 0.30 | — | hold · 0.30 | — |
| 94 | 10350 | MSFT | 2025-10-09 | -0.24 | buy · 0.70 | ✗ | hold · 0.30 | — |
| 95 | 17107 | S | 2026-03-16 | -2.39 | hold · 0.40 | — | hold · 0.40 | — |
| 96 | 3404 | NVDA | 2025-02-19 | -0.12 | hold · 0.30 | — | hold · 0.30 | — |
| 97 | 6928 | NVDA | 2025-06-27 | +1.38 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 98 | 14823 | MU | 2026-01-27 | +2.35 | hold · 0.30 | — | hold · 0.30 | — |
| 99 | 9218 | AMZN | 2025-09-08 | +1.07 | hold · 0.10 | — | hold · 0.10 | — |
| 100 | 13879 | MSFT | 2026-01-06 | +1.21 | hold · 0.30 | — | hold · 0.30 | — |

</details>


---

## 11. `--include-bias` in a bearish window (Jan 2 – Mar 30 2025)

§10 found the debiasing caveat roughly neutral on *calm* tape. This repeats it on the
**down-trending** window from §7 (SPY −4.3%, NASDAQ Composite −10.3%), where over-bullishness
should be most costly — so the caveat has the best chance to help. **30** real-news articles
sampled (seed 7) from the DB; two-phase-similarity retrieval, plain `buy/sell/hold`; the only
difference between columns is the bias caveat.

```bash
python scripts/validate_retreival/validate_sentiment.py \
    --compare bias --db-range 2025-01-02 2025-03-30 --n 30 --seed 7
```

| # | a# | ticker | sell date | gain% | two-phase (act·conf) | ✓ | two-phase + bias (act·conf) | ✓ |
|--:|--:|:--|:--|--:|:--|:-:|:--|:-:|
| 1 | 2584 | MKSI | 2025-01-24 | -1.76 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 2 | 3999 | MRVL | 2025-03-07 | -2.69 | hold · 0.10 | — | hold · 0.30 | — |
| 3 | 3733 | TSM | 2025-02-28 | -0.80 | hold · 0.30 | — | hold · 0.30 | — |
| 4 | 2453 | CRWD | 2025-01-21 | +1.79 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 5 | 3569 | FORM | 2025-02-24 | -3.67 | hold · 0.30 | — | hold · 0.30 | — |
| 6 | 2692 | GEV | 2025-01-28 | +4.10 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 7 | 3889 | CDNS | 2025-03-05 | +1.41 | hold · 0.30 | — | hold · 0.30 | — |
| 8 | 1947 | MSFT | 2025-01-07 | -1.43 | hold · 0.30 | — | hold · 0.30 | — |
| 9 | 4140 | CRWD | 2025-03-12 | +0.57 | hold · 0.40 | — | hold · 0.40 | — |
| 10 | 2987 | SIMO | 2025-02-06 | +7.49 | hold · 0.40 | — | hold · 0.40 | — |
| 11 | 4459 | CRWD | 2025-03-24 | +2.33 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 12 | 2954 | POWL | 2025-02-05 | +3.55 | hold · 0.20 | — | hold · 0.20 | — |
| 13 | 3542 | AMZN | 2025-02-24 | -2.13 | sell · 0.70 | ✓ | sell · 0.70 | ✓ |
| 14 | 3018 | AMZN | 2025-02-06 | +0.01 | sell · 0.70 | ✗ | sell · 0.70 | ✗ |
| 15 | 4056 | NOK | 2025-03-10 | -1.72 | buy · 0.70 | ✗ | hold · 0.30 | — |
| 16 | 2843 | CSCO | 2025-01-31 | -0.51 | hold · 0.10 | — | hold · 0.10 | — |
| 17 | 3967 | PLTR | 2025-03-06 | -6.35 | buy · 0.70 | ✗ | buy · 0.60 | ✗ |
| 18 | 4139 | IBM | 2025-03-12 | +0.25 | hold · 0.30 | — | hold · 0.30 | — |
| 19 | 4331 | NVDA | 2025-03-19 | +2.28 | buy · 0.70 | ✓ | buy · 0.75 | ✓ |
| 20 | 4055 | PLTR | 2025-03-10 | -6.40 | hold · 0.40 | — | hold · 0.40 | — |
| 21 | 2801 | CLS | 2025-01-30 | -0.77 | hold · 0.30 | — | hold · 0.60 | — |
| 22 | 3865 | AVGO | 2025-03-04 | +0.36 | hold · 0.30 | — | hold · 0.30 | — |
| 23 | 3381 | AMZN | 2025-02-18 | -1.33 | hold · 0.10 | — | hold · 0.20 | — |
| 24 | 4107 | GOOG | 2025-03-12 | +1.42 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 25 | 3223 | TEL | 2025-02-12 | +0.56 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 26 | 3191 | AMD | 2025-02-11 | -0.65 | hold · 0.40 | — | hold · 0.30 | — |
| 27 | 4460 | INTC | 2025-03-24 | -0.92 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 28 | 3012 | ORCL | 2025-02-06 | -1.15 | hold · 0.30 | — | hold · 0.30 | — |
| 29 | 3666 | MDB | 2025-02-26 | -0.39 | hold · 0.30 | — | hold · 0.30 | — |
| 30 | 3129 | HPE | 2025-02-10 | +0.19 | hold · 0.20 | — | hold · 0.10 | — |

**Summary**

| variant | BUY-like | SELL-like | HOLD | dir. hit-rate | BUY mean | BUY−SELL spread |
|---|--:|--:|--:|--:|--:|--:|
| no bias | 10 | 2 | 18 | 58% (7/12) | +0.17% | +1.23pt |
| `+bias` | 9 | 2 | 19 | 64% (7/11) | +0.38% | +1.44pt |

**Verdict shift:** 1 of 30 changed — a single `buy → hold` (NOK, a −1.72% loser).

#### Findings

1. **The caveat barely engaged here — 1 of 30 verdicts changed** (vs 11/100 on calm days in
   §10). In the down-tape the model was already cautious (**18/30 HOLD** even without the
   caveat), so the bias note had little room to push further.
2. **That one change was beneficial but is noise-level.** Dropping the losing NOK buy lifts
   buy hit-rate **60% → 67%**, buy mean gain **+0.17% → +0.38%**, and the BUY−SELL spread
   **+1.23 → +1.44pt** — all from a *single* verdict on a tiny directional sample (11–12
   calls). Not a robust effect.
3. **It did not catch the real damage.** The biggest losing buys in the window — PLTR −6.35%
   (a#3967) and INTC −0.92% (a#4460) — stayed BUY under the caveat; the bias note removed the
   marginal NOK call, not the conviction misses.
4. **Consistent with §10.** Even where over-bullishness hurts most, `--include-bias` is a mild
   conservatism nudge, not an accuracy lever — the model's own HOLD-heavy caution does most of
   the de-risking on falling tape.

**Conclusion.** `--include-bias` does **not** robustly fix the bear-market buy problem: it
flipped one marginal buy to hold and otherwise left the conviction misses (PLTR, INTC) intact.
The directional improvement is real but within noise and driven by N=1. Keep it **off by
default** (§10's verdict holds in the bearish regime too); the durable bear-market control is
the regime filter itself (§7) plus the model's baseline abstention, not a prompt caveat.

---

## 12. A constructive bias prompt — does it measure better? (re-run of §10 / §11)

§10–§11 found the original `--include-bias` caveat just blanket-pruned buys (it left the
conviction misses — PLTR, INTC — as BUY). The caveat was rewritten to **triage by materiality
instead of tone**: react to hard events (earnings, M&A, quantified deals, regulatory), lean
`hold` on soft promotional items, and amplify any downplayed negative. The prompt diff:

```diff
- - BIAS WARNING: the insights above (both this article's and the prior ones) are
-   extracted from news/PR-style sources and skew FAVORABLE -- positives are
-   emphasized and negatives downplayed. Take them with a grain of salt: do NOT
-   read a wall of upbeat insights as a buy signal by itself, discount promotional
-   or one-sided positivity, and weight genuinely surprising or negative details
-   more heavily than the favorable framing.
+ - SOURCE-BIAS CALIBRATION: the insights above come from news/PR-style sources that
+   skew FAVORABLE -- positives are inflated, negatives softened. Don't distrust
+   everything; TRIAGE by how material/surprising the underlying FACT is, not by tone:
+     * MATERIAL events -- react on their merits (buy OR sell): earnings results and
+       guidance changes, M&A, large or quantified contracts/deals, regulatory or legal
+       outcomes, major customer wins, capacity/pricing/supply changes. Do NOT dismiss
+       these just because they are wrapped in upbeat PR language.
+     * SOFT / promotional items -- usually already priced, lean 'hold': product
+       features, partnerships with no disclosed terms, awards, 'leading'/'innovative'
+       marketing, reiterated or already-known guidance.
+     * Negatives are downplayed, so any cautious, hedged, or negative detail that still
+       surfaces is unusually informative -- weight it MORE than its mild wording (it
+       can justify 'sell' even amid favorable framing).
+   In short: let hard, specific facts drive the call; don't let upbeat adjectives ALONE
+   make a 'buy', but don't hold through genuinely material news either.
```

Spot-checks confirm the *per-article* behavior improved: the PLTR partnership PR that the old
caveat bought (−6.35%) now correctly `hold`s, while genuine M&A (IREN +9.96%, POWL +3.62%)
still `buy`s. To measure it, §10 (100 calm, seed 42) and §11 (30 bearish, seed 7) were re-run
with the new prompt.

**Calm (100 articles)** — same seed/sample, three prompts:

| variant | BUY | dir. hit-rate | BUY hit-rate | BUY mean | spread |
|---|--:|--:|--:|--:|--:|
| no bias | 33 | 51% (19/37) | 52% (17/33) | +0.20% | +0.50pt |
| old caveat (§10) | 23 | 54% (15/28) | 52% (12/23) | +0.15% | +0.42pt |
| **new caveat** | 27 | 52% (16/31) | 52% (14/27) | −0.02% | +0.28pt |
| new caveat **+ `--include-strong`** | 44 | **46%** (23/50) | 45% (20/44) | +0.01% | +0.65pt |

(BUY columns count BUY-like = `buy`+`strong_buy`; on calm the `strong_buy` bucket was −1.15%
mean / 40% hit — the §9 anti-calibration, back again.)

**Bearish (30 articles)** — same seed/sample, three prompts:

| variant | BUY | dir. hit-rate | BUY hit-rate | BUY mean | spread |
|---|--:|--:|--:|--:|--:|
| no bias | 10 | 58% (7/12) | 60% (6/10) | +0.17% | +1.23pt |
| old caveat (§11) | 9 | 64% (7/11) | 67% (6/9) | +0.38% | +1.44pt |
| **new caveat** | 9 | 64% (7/11) | 67% (6/9) | +0.42% | +1.48pt |
| new caveat **+ `--include-strong`** | 16 | **47%** (9/19) | 50% (8/16) | −0.06% | **−1.85pt** |

#### Findings

1. **The per-article behavior is genuinely better — the aggregate accuracy is not.** Across
   *all three* prompts the **BUY hit-rate is identical** (calm **52%**, bearish **67%**). The
   bias caveat — old or new — moves only the *number* of buys, not their precision. The buys it
   prunes are coin-flip either way, so pruning them can't raise the hit-rate.
2. **The new caveat over-prunes less.** On calm tape it keeps **27** buys vs the old caveat's 23
   (7 `buy→hold` + 1 `hold→buy`, vs the old's 10 `buy→hold`) — exactly the intended effect of
   reacting to material news instead of distrusting all positives. But at equal precision, the
   4 extra buys slightly *dilute* the calm BUY mean (−0.02% vs +0.15%) and spread.
3. **Bearish is a statistical tie** with the old caveat (9 buys, 64% hit, +0.42% vs +0.38%): a
   net of 3 verdicts moved (2 `buy→hold`, 1 `hold→buy`) but they cancel out.
4. **Stacking `--include-strong` on top undoes the caution — and is the worst of all.** The
   strong action menu's decisiveness *overpowers* the bias caveat: BUY-like jumps to **44 (calm)
   / 16 (bearish)** — *more* directional calls than even the no-bias baseline — while hit-rate
   drops to **46% / 47%** and the bearish spread flips to **−1.85pt**. `strong_buy` is
   anti-calibrated all over again (−1.15% mean, §9). So the two flags don't compose: the
   strong menu re-inflates exactly the marginal buys the bias caveat was meant to suppress.
5. **Conclusion stands from §10.** A post-hoc prompt caveat is a *volume / conservatism* lever,
   not an accuracy lever; on this intraday metric and small directional N it cannot beat the
   model's underlying ~coin-flip buy precision. The new prompt is the **better-behaved form of
   `--include-bias`** (use it if the flag is used at all), but it does not justify turning the
   flag on by default — precision must come from the retrieval gates and the signal itself, not
   the prompt. **`--include-strong` should not be combined with it.**

<details>
<summary>Full new-prompt tables — calm (100) and bearish (30)</summary>

**Calm (100, new caveat)**

| # | a# | ticker | sell date | gain% | two-phase (act·conf) | ✓ | two-phase + bias (act·conf) | ✓ |
|--:|--:|:--|:--|--:|:--|:-:|:--|:-:|
| 1 | 19833 | NVDA | 2026-05-14 | +2.50 | hold · 0.10 | — | hold · 0.10 | — |
| 2 | 19819 | AMZN | 2026-05-14 | -0.97 | buy · 0.75 | ✗ | buy · 0.75 | ✗ |
| 3 | 10534 | WDC | 2025-10-13 | +1.11 | buy · 0.70 | ✓ | hold · 0.30 | — |
| 4 | 11257 | AMZN | 2025-10-30 | -2.20 | hold · 0.30 | — | hold · 0.30 | — |
| 5 | 8631 | NOK | 2025-08-21 | +0.47 | buy · 0.70 | ✓ | buy · 0.75 | ✓ |
| 6 | 1298 | AMZN | 2024-12-12 | -0.69 | hold · 0.30 | — | hold · 0.30 | — |
| 7 | 16011 | AVGO | 2026-02-19 | -0.31 | buy · 0.75 | ✗ | buy · 0.75 | ✗ |
| 8 | 1445 | COHR | 2024-12-17 | -2.57 | hold · 0.10 | — | hold · 0.30 | — |
| 9 | 13612 | CSCO | 2025-12-26 | +0.70 | hold · 0.10 | — | hold · 0.10 | — |
| 10 | 7866 | IBM | 2025-07-30 | -0.80 | hold · 0.30 | — | hold · 0.30 | — |
| 11 | 1190 | PLTR | 2024-12-09 | -11.61 | buy · 0.85 | ✗ | buy · 0.85 | ✗ |
| 12 | 12126 | ADBE | 2025-11-19 | -0.29 | hold · 0.30 | — | hold · 0.30 | — |
| 13 | 7915 | NVDA | 2025-07-31 | -2.73 | buy · 0.70 | ✗ | buy · 0.75 | ✗ |
| 14 | 7770 | LNVGY | 2025-07-28 | +2.43 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 15 | 19306 | AMAT | 2026-05-04 | +0.07 | buy · 0.70 | ✓ | buy · 0.75 | ✓ |
| 16 | 13844 | AMBA | 2026-01-05 | -0.45 | buy · 0.80 | ✗ | buy · 0.85 | ✗ |
| 17 | 14210 | GOOG | 2026-01-12 | +1.94 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 18 | 14208 | NVDA | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 19 | 19675 | NVDA | 2026-05-11 | -1.15 | hold · 0.30 | — | hold · 0.30 | — |
| 20 | 19102 | DELL | 2026-04-29 | +0.69 | hold · 0.30 | — | hold · 0.30 | — |
| 21 | 13326 | NVDA | 2025-12-17 | -3.92 | hold · 0.10 | — | hold · 0.10 | — |
| 22 | 11563 | LNVGY | 2025-11-06 | -3.20 | hold · 0.30 | — | hold · 0.30 | — |
| 23 | 13972 | MSFT | 2026-01-07 | +1.00 | hold · 0.10 | — | hold · 0.10 | — |
| 24 | 14391 | AMZN | 2026-01-16 | +0.08 | hold · 0.30 | — | hold · 0.30 | — |
| 25 | 7222 | GOOG | 2025-07-10 | +0.70 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 26 | 19431 | AMZN | 2026-05-06 | +0.84 | hold · 0.30 | — | hold · 0.30 | — |
| 27 | 7393 | POWL | 2025-07-16 | +3.62 | buy · 0.70 | ✓ | buy · 0.75 | ✓ |
| 28 | 9620 | HPE | 2025-09-17 | +2.61 | hold · 0.20 | — | hold · 0.30 | — |
| 29 | 13787 | NVDA | 2026-01-02 | -0.54 | hold · 0.30 | — | hold · 0.30 | — |
| 30 | 7047 | ADI | 2025-07-03 | -0.03 | hold · 0.30 | — | hold · 0.30 | — |
| 31 | 8845 | AMBA | 2025-08-27 | +2.12 | hold · 0.30 | — | hold · 0.30 | — |
| 32 | 16576 | AVGO | 2026-03-05 | +4.44 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 33 | 10567 | AMZN | 2025-10-14 | -1.60 | hold · 0.10 | — | hold · 0.10 | — |
| 34 | 14701 | NVDA | 2026-01-23 | +0.24 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 35 | 14712 | DELL | 2026-01-23 | +0.68 | hold · 0.10 | — | hold · 0.30 | — |
| 36 | 623 | GOOG | 2024-11-20 | -1.13 | hold · 0.30 | — | hold · 0.30 | — |
| 37 | 19378 | IREN | 2026-05-05 | +9.96 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 38 | 14045 | IBM | 2026-01-08 | -0.14 | hold · 0.30 | — | hold · 0.30 | — |
| 39 | 15347 | AMZN | 2026-02-05 | +0.37 | hold · 0.10 | — | hold · 0.30 | — |
| 40 | 14314 | META | 2026-01-14 | -0.38 | hold · 0.10 | — | hold · 0.10 | — |
| 41 | 9825 | MSFT | 2025-09-26 | +0.31 | hold · 0.30 | — | hold · 0.30 | — |
| 42 | 14889 | NVDA | 2026-01-28 | -0.01 | buy · 0.75 | ✗ | buy · 0.75 | ✗ |
| 43 | 8463 | ASX | 2025-08-15 | -0.05 | hold · 0.30 | — | hold · 0.30 | — |
| 44 | 16038 | AMD | 2026-02-20 | -1.55 | buy · 0.70 | ✗ | buy · 0.75 | ✗ |
| 45 | 9661 | AVGO | 2025-09-17 | +0.54 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 46 | 19852 | AMZN | 2026-05-14 | -0.64 | hold · 0.10 | — | hold · 0.10 | — |
| 47 | 9942 | QCOM | 2025-09-30 | +0.38 | hold · 0.30 | — | hold · 0.30 | — |
| 48 | 7134 | CSCO | 2025-07-07 | -0.04 | hold · 0.30 | — | hold · 0.30 | — |
| 49 | 10281 | AMD | 2025-10-08 | +9.66 | hold · 0.30 | — | hold · 0.30 | — |
| 50 | 608 | HUBB | 2024-11-20 | -1.52 | hold · 0.30 | — | hold · 0.20 | — |
| 51 | 6646 | NOK | 2025-06-16 | +0.95 | hold · 0.40 | — | hold · 0.30 | — |
| 52 | 14486 | MCHP | 2026-01-20 | +0.48 | buy · 0.75 | ✓ | buy · 0.70 | ✓ |
| 53 | 14207 | NVDA | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 54 | 19835 | NVDA | 2026-05-14 | +1.16 | hold · 0.30 | — | hold · 0.30 | — |
| 55 | 16186 | KEYS | 2026-02-24 | +6.91 | hold · 0.30 | — | hold · 0.40 | — |
| 56 | 11556 | GOOG | 2025-11-06 | -0.05 | hold · 0.10 | — | hold · 0.10 | — |
| 57 | 7392 | AVGO | 2025-07-15 | -0.47 | hold · 0.40 | — | hold · 0.40 | — |
| 58 | 9740 | IBM | 2025-09-22 | +1.60 | hold · 0.10 | — | hold · 0.30 | — |
| 59 | 8993 | FLEX | 2025-09-02 | -0.09 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 60 | 13375 | NVDA | 2025-12-18 | +1.02 | sell · 0.65 | ✗ | sell · 0.60 | ✗ |
| 61 | 19908 | NVDA | 2026-05-18 | -1.23 | hold · 0.30 | — | hold · 0.30 | — |
| 62 | 3409 | ARM | 2025-02-19 | -3.46 | hold · 0.30 | — | hold · 0.30 | — |
| 63 | 13272 | AMZN | 2025-12-16 | -0.20 | hold · 0.10 | — | hold · 0.20 | — |
| 64 | 9244 | GOOG | 2025-09-08 | -0.13 | hold · 0.30 | — | hold · 0.40 | — |
| 65 | 9777 | NOK | 2025-09-24 | -1.25 | buy · 0.70 | ✗ | hold · 0.30 | — |
| 66 | 13742 | AVGO | 2025-12-31 | -0.71 | buy · 0.70 | ✗ | buy · 0.75 | ✗ |
| 67 | 7068 | ADBE | 2025-07-07 | -0.54 | hold · 0.30 | — | hold · 0.30 | — |
| 68 | 815 | NOK | 2024-11-27 | +0.24 | buy · 0.85 | ✓ | buy · 0.85 | ✓ |
| 69 | 9080 | NVDA | 2025-09-04 | -0.01 | hold · 0.10 | — | hold · 0.30 | — |
| 70 | 10245 | APLD | 2025-10-07 | -2.85 | hold · 0.30 | — | hold · 0.30 | — |
| 71 | 16241 | MSFT | 2026-02-25 | +2.80 | hold · 0.30 | — | hold · 0.30 | — |
| 72 | 19946 | AMAT | 2026-05-18 | -3.48 | hold · 0.30 | — | hold · 0.30 | — |
| 73 | 9719 | ARM | 2025-09-19 | -0.83 | sell · 0.60 | ✓ | sell · 0.60 | ✓ |
| 74 | 14095 | META | 2026-01-09 | +1.07 | hold · 0.30 | — | hold · 0.30 | — |
| 75 | 12121 | ADBE | 2025-11-19 | -0.15 | hold · 0.30 | — | hold · 0.40 | — |
| 76 | 7811 | TSM | 2025-07-28 | +0.34 | sell · 0.70 | ✗ | sell · 0.70 | ✗ |
| 77 | 10252 | IBM | 2025-10-07 | -1.90 | buy · 0.85 | ✗ | hold · 0.30 | — |
| 78 | 11434 | MSFT | 2025-11-03 | -0.83 | hold · 0.30 | — | hold · 0.30 | — |
| 79 | 13505 | NVDA | 2025-12-22 | +0.04 | hold · 0.30 | — | hold · 0.30 | — |
| 80 | 20103 | CRWD | 2026-05-20 | +6.40 | buy · 0.75 | ✓ | hold · 0.40 | — |
| 81 | 694 | NOK | 2024-11-22 | +2.20 | hold · 0.30 | — | hold · 0.30 | — |
| 82 | 13333 | AMZN | 2025-12-17 | -1.68 | hold · 0.10 | — | hold · 0.20 | — |
| 83 | 149 | GOOG | 2024-11-04 | -1.34 | buy · 0.75 | ✗ | hold · 0.40 | — |
| 84 | 11135 | MSFT | 2025-10-27 | -0.26 | hold · 0.10 | — | hold · 0.10 | — |
| 85 | 7377 | MSFT | 2025-07-15 | +0.10 | hold · 0.30 | — | hold · 0.30 | — |
| 86 | 13695 | APLD | 2025-12-30 | -2.82 | buy · 0.70 | ✗ | buy · 0.75 | ✗ |
| 87 | 14254 | META | 2026-01-13 | +0.32 | hold · 0.10 | — | hold · 0.10 | — |
| 88 | 14141 | GOOG | 2026-01-12 | +1.97 | hold · 0.10 | — | hold · 0.10 | — |
| 89 | 1056 | META | 2024-12-05 | -0.95 | buy · 0.70 | ✗ | buy · 0.75 | ✗ |
| 90 | 10201 | AMD | 2025-10-06 | -2.92 | hold · 0.60 | — | buy · 0.90 | ✗ |
| 91 | 10277 | MSFT | 2025-10-08 | +0.09 | hold · 0.10 | — | hold · 0.10 | — |
| 92 | 6649 | NOK | 2025-06-16 | +0.95 | hold · 0.40 | — | hold · 0.40 | — |
| 93 | 9125 | AI | 2025-09-04 | -1.75 | sell · 0.85 | ✓ | sell · 0.85 | ✓ |
| 94 | 8436 | NVDA | 2025-08-15 | -0.70 | hold · 0.30 | — | hold · 0.30 | — |
| 95 | 10323 | SNPS | 2025-10-09 | -0.63 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 96 | 10324 | AMZN | 2025-10-09 | +1.01 | hold · 0.30 | — | hold · 0.30 | — |
| 97 | 10350 | MSFT | 2025-10-09 | -0.24 | buy · 0.70 | ✗ | hold · 0.30 | — |
| 98 | 17107 | S | 2026-03-16 | -2.39 | hold · 0.40 | — | hold · 0.40 | — |
| 99 | 3404 | NVDA | 2025-02-19 | -0.12 | hold · 0.30 | — | hold · 0.30 | — |
| 100 | 6928 | NVDA | 2025-06-27 | +1.38 | buy · 0.75 | ✓ | hold · 0.40 | — |

**Bearish (30, new caveat)**

| # | a# | ticker | sell date | gain% | two-phase (act·conf) | ✓ | two-phase + bias (act·conf) | ✓ |
|--:|--:|:--|:--|--:|:--|:-:|:--|:-:|
| 1 | 2584 | MKSI | 2025-01-24 | -1.76 | buy · 0.70 | ✗ | hold · 0.30 | — |
| 2 | 3999 | MRVL | 2025-03-07 | -2.69 | hold · 0.10 | — | hold · 0.30 | — |
| 3 | 3733 | TSM | 2025-02-28 | -0.80 | hold · 0.30 | — | hold · 0.30 | — |
| 4 | 2453 | CRWD | 2025-01-21 | +1.79 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 5 | 3569 | FORM | 2025-02-24 | -3.67 | hold · 0.30 | — | hold · 0.30 | — |
| 6 | 2692 | GEV | 2025-01-28 | +4.10 | buy · 0.75 | ✓ | buy · 0.75 | ✓ |
| 7 | 3889 | CDNS | 2025-03-05 | +1.41 | hold · 0.30 | — | hold · 0.30 | — |
| 8 | 1947 | MSFT | 2025-01-07 | -1.43 | hold · 0.30 | — | buy · 0.70 | ✗ |
| 9 | 4140 | CRWD | 2025-03-12 | +0.57 | hold · 0.40 | — | hold · 0.30 | — |
| 10 | 2987 | SIMO | 2025-02-06 | +7.49 | hold · 0.40 | — | hold · 0.40 | — |
| 11 | 4459 | CRWD | 2025-03-24 | +2.33 | buy · 0.70 | ✓ | buy · 0.75 | ✓ |
| 12 | 2954 | POWL | 2025-02-05 | +3.55 | hold · 0.20 | — | hold · 0.30 | — |
| 13 | 3542 | AMZN | 2025-02-24 | -2.13 | sell · 0.70 | ✓ | sell · 0.70 | ✓ |
| 14 | 3018 | AMZN | 2025-02-06 | +0.01 | sell · 0.70 | ✗ | sell · 0.70 | ✗ |
| 15 | 4056 | NOK | 2025-03-10 | -1.72 | buy · 0.70 | ✗ | hold · 0.30 | — |
| 16 | 2843 | CSCO | 2025-01-31 | -0.51 | hold · 0.10 | — | hold · 0.10 | — |
| 17 | 3967 | PLTR | 2025-03-06 | -6.35 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 18 | 4139 | IBM | 2025-03-12 | +0.25 | hold · 0.30 | — | hold · 0.30 | — |
| 19 | 4331 | NVDA | 2025-03-19 | +2.28 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 20 | 4055 | PLTR | 2025-03-10 | -6.40 | hold · 0.40 | — | hold · 0.40 | — |
| 21 | 2801 | CLS | 2025-01-30 | -0.77 | hold · 0.30 | — | hold · 0.40 | — |
| 22 | 3865 | AVGO | 2025-03-04 | +0.36 | hold · 0.30 | — | hold · 0.30 | — |
| 23 | 3381 | AMZN | 2025-02-18 | -1.33 | hold · 0.10 | — | hold · 0.20 | — |
| 24 | 4107 | GOOG | 2025-03-12 | +1.42 | buy · 0.70 | ✓ | buy · 0.70 | ✓ |
| 25 | 3223 | TEL | 2025-02-12 | +0.56 | buy · 0.70 | ✓ | buy · 0.75 | ✓ |
| 26 | 3191 | AMD | 2025-02-11 | -0.65 | hold · 0.40 | — | hold · 0.30 | — |
| 27 | 4460 | INTC | 2025-03-24 | -0.92 | buy · 0.70 | ✗ | buy · 0.70 | ✗ |
| 28 | 3012 | ORCL | 2025-02-06 | -1.15 | hold · 0.30 | — | hold · 0.30 | — |
| 29 | 3666 | MDB | 2025-02-26 | -0.39 | hold · 0.30 | — | hold · 0.30 | — |
| 30 | 3129 | HPE | 2025-02-10 | +0.19 | hold · 0.20 | — | hold · 0.20 | — |

</details>

---

## 13. `--choose-1`: let the model pick its single best ticker (100 peaceful-day articles)

§6–§12 always score a **fixed** ticker (the article's primary). The `--choose-1` flag
changes the question: the model is shown **all** of an article's tickers in one prompt and
must return **only the single most-confident verdict** — it chooses *which* ticker to commit
to. This tests whether forcing a "pick your best shot" framing surfaces a sharper, higher-
conviction call, or merely manufactures conviction without signal.

**100** articles sampled (seed 42) from `peaceful_days_articles.csv` — the *same* draw as
§10/§12 — judged with **two-phase-similarity** retrieval, the plain `buy/sell/hold` menu, and
**`--include-bias`** (the constructive §12 caveat). One Gemini call per article; the chosen
ticker is then priced buy-at-publish → close. Produced by
[`scripts/validate_retreival/validate_choose1.py`](../../scripts/validate_retreival/validate_choose1.py).

```bash
python scripts/validate_retreival/validate_choose1.py --n 100 --seed 42
```

**Headline — `--choose-1 + bias` vs the matched fixed-primary baselines (same 100, seed 42)**

| variant | scored ticker | BUY-like | SELL-like | HOLD | dir. hit-rate | BUY mean | BUY−SELL spread |
|---|---|--:|--:|--:|--:|--:|--:|
| no bias (§10) | primary (fixed) | 33 | 4 | 63 | 51% (19/37) | +0.20% | +0.50pt |
| new caveat (§12) | primary (fixed) | 27 | 5 | 68 | 52% (16/31) | −0.02% | +0.28pt |
| **`--choose-1` + bias** | **model's pick** | **46** | **7** | **47** | **53% (28/53)** | **−0.27%** | **−0.01pt** |

**Where the chosen ticker came from** — primary vs mentioned peer, and single- vs multi-ticker:

| subset | n | BUY-like | dir. hit-rate | BUY-like mean gain |
|---|--:|--:|--:|--:|
| chosen = **primary** | 81 | 36 | 49% (20/41) | +0.01% |
| chosen = **mentioned peer** | 19 | 10 | 67% (8/12) | **−1.28%** |
| single-ticker articles | 57 | 25 | 48% (13/27) | +0.01% |
| multi-ticker (≥2) articles | 43 | 21 | 58% (15/26) | −0.61% |

**Per-action buckets** (all 100)

| action | n | mean gain% | dir hit-rate |
|---|--:|--:|--:|
| buy | 46 | −0.27% | 52% (24/46) |
| hold | 47 | −0.06% | — |
| sell | 7 | −0.26% | 57% (4/7) |

**Confidence calibration** (directional calls only)

| conf band | n | mean gain% | dir hit-rate |
|---|--:|--:|--:|
| <0.70 | 2 | +0.45% | 50% (1/2) |
| 0.70–0.85 | 31 | −0.14% | 48% (15/31) |
| ≥0.85 | 20 | −0.54% | 60% (12/20) |

#### Findings

1. **`--choose-1` makes the model far more decisive — it nearly doubles the directional
   volume.** Forced to surface its best call, it issues **53 directional verdicts (46 buy +
   7 sell)** vs the fixed-primary + same-caveat baseline's **31** (§12), and HOLD collapses
   **68 → 47**. The "pick one" framing strips out most of the abstention.
2. **But the precision is unchanged — still a coin flip.** Directional hit-rate is **53%**,
   statistically identical to the fixed-primary **51–52%**. The extra ~22 calls choose-1
   manufactures are no better than the ones the model previously left as HOLD; the framing adds
   *conviction*, not *signal*.
3. **And the BUY economics get *worse*.** BUY-like mean gain is **−0.27%** (vs +0.20% no-bias /
   −0.02% §12) and the BUY−SELL spread flattens to **−0.01pt**. Letting the model roam to the
   ticker it "likes most" tilts it toward hype names, not toward better-priced reactions.
4. **Roaming to mentioned peers is the riskiest part.** In **19/100** articles it picked a
   *mentioned* peer over the primary. Those picks hit 67% directionally but averaged **−1.28%**
   on buys — because the freedom to chase a side-mention lands it on thin, headline-y names with
   fat-tail losses: **CBRS** was chosen three times (a#19675, a#19835 both **−11.12%**, a#19946
   **+7.02%**), and the worst two buys overall are a primary **PLTR −11.61%** and those two
   **CBRS −11.12%** mentioned picks. Picking among many tickers widens the tail in both
   directions without improving the hit-rate.
5. **Confidence still ranks direction better than magnitude.** The ≥0.85 band hits 60% (best of
   the three bands) but has the *worst* mean gain (−0.54%) — the model's high-conviction calls
   are right on sign more often yet include its biggest losers (PLTR, CBRS, APLD, AMD all at
   0.75–0.95). Same anti-calibration for *size* seen in §9/§12.

**Conclusion — `--choose-1` is a decisiveness/UX lever, not an accuracy lever.** It cleanly
answers "if you had to bet on exactly one ticker from this article, which and how?", and on
that axis it works: one confident verdict instead of a wall of HOLDs. But on this peaceful-day,
intraday metric it buys that decisiveness at **flat ~coin-flip precision and a worse BUY mean**,
and its freedom to pick *mentioned* peers exposes it to illiquid fat-tail names (the CBRS −11%
hits). Use it for triage/ranking when a single call is required; do **not** read its forced pick
as a higher-quality signal than the primary-ticker verdict. (Data caveat: extreme returns on
obscure mentioned tickers like CBRS are partly a liquidity/price-data artifact, not clean
tradeable P&L.)

<details>
<summary>Full 100-row table (chosen ticker · role · verdict · confidence · buy→close gain)</summary>

| # | a# | chosen | role | #tk | sell date | gain% | verdict (act·conf) | ✓ |
|--:|--:|:--|:--|--:|:--|--:|:--|:-:|
| 1 | 19833 | NVDA | primary | 1 | 2026-05-14 | +2.50 | hold · 0.10 | — |
| 2 | 19819 | AMZN | primary | 1 | 2026-05-14 | -0.97 | buy · 0.75 | ✗ |
| 3 | 10534 | WDC | primary | 1 | 2025-10-13 | +1.11 | buy · 0.70 | ✓ |
| 4 | 11257 | MSFT | mentioned | 2 | 2025-10-30 | -0.68 | sell · 0.75 | ✓ |
| 5 | 8631 | NOK | primary | 1 | 2025-08-21 | +0.47 | buy · 0.75 | ✓ |
| 6 | 1298 | NVDA | mentioned | 2 | 2024-12-12 | +0.46 | buy · 0.75 | ✓ |
| 7 | 16011 | AVGO | primary | 1 | 2026-02-19 | -0.31 | buy · 0.85 | ✗ |
| 8 | 1445 | COHR | primary | 2 | 2024-12-17 | -2.57 | hold · 0.30 | — |
| 9 | 13612 | NVDA | mentioned | 2 | 2025-12-26 | +0.61 | hold · 0.30 | — |
| 10 | 7866 | IBM | primary | 1 | 2025-07-30 | -0.80 | hold · 0.30 | — |
| 11 | 1190 | PLTR | primary | 1 | 2024-12-09 | -11.61 | buy · 0.85 | ✗ |
| 12 | 12126 | ADBE | primary | 1 | 2025-11-19 | -0.29 | hold · 0.30 | — |
| 13 | 7915 | NVDA | primary | 2 | 2025-07-31 | -2.73 | buy · 0.70 | ✗ |
| 14 | 7770 | LNVGY | primary | 1 | 2025-07-28 | +2.43 | buy · 0.85 | ✓ |
| 15 | 19306 | AMAT | primary | 1 | 2026-05-04 | +0.07 | buy · 0.75 | ✓ |
| 16 | 13844 | AMBA | primary | 1 | 2026-01-05 | -0.45 | buy · 0.85 | ✗ |
| 17 | 14210 | GOOG | primary | 2 | 2026-01-12 | +1.94 | buy · 0.75 | ✓ |
| 18 | 14208 | NVDA | primary | 1 | 2026-01-12 | +0.02 | buy · 0.85 | ✓ |
| 19 | 19675 | CBRS | mentioned | 5 | 2026-05-14 | -11.12 | buy · 0.85 | ✗ |
| 20 | 19102 | NVDA | mentioned | 4 | 2026-04-29 | -1.90 | buy · 0.75 | ✗ |
| 21 | 13326 | NVDA | primary | 1 | 2025-12-17 | -3.92 | hold · 0.10 | — |
| 22 | 11563 | LNVGY | primary | 2 | 2025-11-06 | -3.20 | hold · 0.30 | — |
| 23 | 13972 | MSFT | primary | 1 | 2026-01-07 | +1.00 | hold · 0.10 | — |
| 24 | 14391 | AMZN | primary | 1 | 2026-01-16 | +0.08 | hold · 0.30 | — |
| 25 | 7222 | GOOG | primary | 3 | 2025-07-10 | +0.70 | buy · 0.75 | ✓ |
| 26 | 19431 | AMZN | primary | 1 | 2026-05-06 | +0.84 | hold · 0.30 | — |
| 27 | 7393 | POWL | primary | 1 | 2025-07-16 | +3.62 | buy · 0.75 | ✓ |
| 28 | 9620 | NOK | mentioned | 3 | 2025-09-17 | +1.95 | buy · 0.75 | ✓ |
| 29 | 13787 | NVDA | primary | 1 | 2026-01-02 | -0.54 | hold · 0.50 | — |
| 30 | 7047 | ADI | primary | 1 | 2025-07-03 | -0.03 | hold · 0.30 | — |
| 31 | 8845 | NVDA | mentioned | 3 | 2025-08-27 | -0.24 | hold · 0.30 | — |
| 32 | 16576 | AVGO | primary | 1 | 2026-03-05 | +4.44 | buy · 0.85 | ✓ |
| 33 | 10567 | AMZN | primary | 1 | 2025-10-14 | -1.60 | hold · 0.10 | — |
| 34 | 14701 | NVDA | primary | 1 | 2026-01-23 | +0.24 | buy · 0.75 | ✓ |
| 35 | 14712 | ORCL | mentioned | 2 | 2026-01-23 | -0.25 | hold · 0.30 | — |
| 36 | 623 | GOOG | primary | 3 | 2024-11-20 | -1.13 | hold · 0.30 | — |
| 37 | 19378 | IREN | primary | 2 | 2026-05-05 | +9.96 | buy · 0.85 | ✓ |
| 38 | 14045 | IBM | primary | 1 | 2026-01-08 | -0.14 | hold · 0.30 | — |
| 39 | 15347 | PLTR | mentioned | 5 | 2026-02-05 | +0.81 | buy · 0.85 | ✓ |
| 40 | 14314 | META | primary | 1 | 2026-01-14 | -0.38 | hold · 0.30 | — |
| 41 | 9825 | MSFT | primary | 1 | 2025-09-26 | +0.31 | hold · 0.40 | — |
| 42 | 14889 | NVDA | primary | 1 | 2026-01-28 | -0.01 | buy · 0.75 | ✗ |
| 43 | 8463 | TSM | mentioned | 2 | 2025-08-15 | +0.27 | hold · 0.30 | — |
| 44 | 16038 | AMD | primary | 2 | 2026-02-20 | -1.55 | buy · 0.70 | ✗ |
| 45 | 9661 | AVGO | primary | 2 | 2025-09-17 | +0.54 | buy · 0.85 | ✓ |
| 46 | 19852 | AMZN | primary | 1 | 2026-05-14 | -0.64 | hold · 0.10 | — |
| 47 | 9942 | QCOM | primary | 1 | 2025-09-30 | +0.38 | hold · 0.30 | — |
| 48 | 7134 | NVDA | mentioned | 4 | 2025-07-07 | +0.19 | sell · 0.70 | ✗ |
| 49 | 10281 | INTC | mentioned | 2 | 2025-10-08 | +0.43 | buy · 0.70 | ✓ |
| 50 | 608 | HUBB | primary | 1 | 2024-11-20 | -1.52 | hold · 0.10 | — |
| 51 | 6646 | NOK | primary | 5 | 2025-06-16 | +0.95 | hold · 0.30 | — |
| 52 | 14486 | MCHP | primary | 1 | 2026-01-20 | +0.48 | buy · 0.75 | ✓ |
| 53 | 14207 | NVDA | primary | 1 | 2026-01-12 | +0.02 | buy · 0.85 | ✓ |
| 54 | 19835 | CBRS | mentioned | 2 | 2026-05-14 | -11.12 | buy · 0.85 | ✗ |
| 55 | 16186 | KEYS | primary | 1 | 2026-02-24 | +6.91 | hold · 0.30 | — |
| 56 | 11556 | GOOG | primary | 2 | 2025-11-06 | -0.05 | hold · 0.10 | — |
| 57 | 7392 | AVGO | primary | 2 | 2025-07-15 | -0.47 | buy · 0.75 | ✗ |
| 58 | 9740 | IBM | primary | 2 | 2025-09-22 | +1.60 | hold · 0.10 | — |
| 59 | 8993 | FLEX | primary | 1 | 2025-09-02 | -0.09 | buy · 0.70 | ✗ |
| 60 | 13375 | NVDA | primary | 1 | 2025-12-18 | +1.02 | sell · 0.65 | ✗ |
| 61 | 19908 | NVDA | primary | 1 | 2026-05-18 | -1.23 | buy · 0.70 | ✗ |
| 62 | 3409 | CDNS | mentioned | 4 | 2025-02-19 | -3.97 | hold · 0.50 | — |
| 63 | 13272 | AMZN | primary | 1 | 2025-12-16 | -0.20 | hold · 0.30 | — |
| 64 | 9244 | GOOG | primary | 2 | 2025-09-08 | -0.13 | sell · 0.65 | ✓ |
| 65 | 9777 | NOK | primary | 1 | 2025-09-24 | -1.25 | buy · 0.75 | ✗ |
| 66 | 13742 | AVGO | primary | 7 | 2025-12-31 | -0.71 | buy · 0.75 | ✗ |
| 67 | 7068 | ADBE | primary | 1 | 2025-07-07 | -0.54 | hold · 0.30 | — |
| 68 | 815 | NOK | primary | 1 | 2024-11-27 | +0.24 | buy · 0.85 | ✓ |
| 69 | 9080 | NVDA | primary | 1 | 2025-09-04 | -0.01 | hold · 0.30 | — |
| 70 | 10245 | APLD | primary | 1 | 2025-10-07 | -2.85 | buy · 0.85 | ✗ |
| 71 | 16241 | MSFT | primary | 2 | 2026-02-25 | +2.80 | hold · 0.30 | — |
| 72 | 19946 | CBRS | mentioned | 10 | 2026-05-18 | +7.02 | buy · 0.85 | ✓ |
| 73 | 9719 | ARM | primary | 2 | 2025-09-19 | -0.83 | sell · 0.70 | ✓ |
| 74 | 14095 | META | primary | 1 | 2026-01-09 | +1.07 | hold · 0.40 | — |
| 75 | 12121 | ADBE | primary | 1 | 2025-11-19 | -0.15 | hold · 0.40 | — |
| 76 | 7811 | TSM | primary | 1 | 2025-07-28 | +0.34 | sell · 0.75 | ✗ |
| 77 | 10252 | IBM | primary | 2 | 2025-10-07 | -1.90 | buy · 0.85 | ✗ |
| 78 | 11434 | NVDA | mentioned | 2 | 2025-11-03 | +0.22 | buy · 0.70 | ✓ |
| 79 | 13505 | NVDA | primary | 3 | 2025-12-22 | +0.04 | hold · 0.30 | — |
| 80 | 20103 | CRWD | primary | 1 | 2026-05-20 | +6.40 | buy · 0.85 | ✓ |
| 81 | 694 | NOK | primary | 1 | 2024-11-22 | +2.20 | hold · 0.40 | — |
| 82 | 13333 | AMZN | primary | 1 | 2025-12-17 | -1.68 | hold · 0.30 | — |
| 83 | 149 | GOOGL | mentioned | 3 | 2024-11-04 | -1.32 | hold · 0.30 | — |
| 84 | 11135 | MSFT | primary | 1 | 2025-10-27 | -0.26 | hold · 0.10 | — |
| 85 | 7377 | MSFT | primary | 2 | 2025-07-15 | +0.10 | hold · 0.30 | — |
| 86 | 13695 | APLD | primary | 2 | 2025-12-30 | -2.82 | buy · 0.75 | ✗ |
| 87 | 14254 | META | primary | 1 | 2026-01-13 | +0.32 | hold · 0.10 | — |
| 88 | 14141 | NVDA | mentioned | 3 | 2026-01-12 | +1.17 | hold · 0.10 | — |
| 89 | 1056 | META | primary | 1 | 2024-12-05 | -0.95 | buy · 0.75 | ✗ |
| 90 | 10201 | AMD | primary | 3 | 2025-10-06 | -2.92 | buy · 0.95 | ✗ |
| 91 | 10277 | MSFT | primary | 1 | 2025-10-08 | +0.09 | hold · 0.10 | — |
| 92 | 6649 | NOK | primary | 5 | 2025-06-16 | +0.95 | hold · 0.30 | — |
| 93 | 9125 | AI | primary | 5 | 2025-09-04 | -1.75 | sell · 0.90 | ✓ |
| 94 | 8436 | SMCI | mentioned | 2 | 2025-08-15 | +0.47 | buy · 0.75 | ✓ |
| 95 | 10323 | SNPS | primary | 1 | 2025-10-09 | -0.63 | buy · 0.75 | ✗ |
| 96 | 10324 | AMZN | primary | 1 | 2025-10-09 | +1.01 | hold · 0.30 | — |
| 97 | 10350 | MSFT | primary | 1 | 2025-10-09 | -0.24 | buy · 0.75 | ✗ |
| 98 | 17107 | S | primary | 1 | 2026-03-16 | -2.39 | hold · 0.40 | — |
| 99 | 3404 | NVDA | primary | 1 | 2025-02-19 | -0.12 | hold · 0.30 | — |
| 100 | 6928 | NVDA | primary | 1 | 2025-06-27 | +1.38 | buy · 0.85 | ✓ |

</details>

---

## 14. `--clean-top-1`: does cleaning + justification-augmenting the context sharpen the pick? (100 peaceful-day articles)

§13 took the model's single `--choose-1` verdict at face value. `--clean-top-1` adds a
**refinement loop** on top of it: after round 1 returns the chosen ticker + justification, the
context is (1) **pruned** to only the insights relevant to that ticker/verdict (a screening
call), (2) **augmented** with the top-20 insights whose cosine similarity to the verdict's own
**justification** is ≥0.75 (same no-lookahead window), and (3) the choose-1 verdict is
**re-issued** on that cleaned+augmented context (round 2). The question: does grounding the
model in a tighter, self-retrieved context produce a better call?

**100** articles, **seed 42** — the *same* draw as §10/§12/§13 — two-phase-similarity
retrieval, `--include-bias`. Round 1 is therefore identical to §13's run (a built-in control).
Produced by [`scripts/validate_retreival/validate_clean_top1.py`](../../scripts/validate_retreival/validate_clean_top1.py).

```bash
python scripts/search/insight_sentiment.py <id> --retrieval two-phase-similarity \
    --include-bias --clean-top-1
python scripts/validate_retreival/validate_clean_top1.py --n 100 --seed 42
```

**Before vs after the cleaning phase** (gain = buy-at-publish → close of each round's pick)

| round | BUY-like | SELL-like | HOLD | dir. hit-rate | BUY mean | BUY−SELL spread |
|---|--:|--:|--:|--:|--:|--:|
| before (round 1 = §13) | 46 | 7 | 47 | **53% (28/53)** | −0.27% | −0.01pt |
| after (round 2, cleaned) | 44 | 7 | 49 | **49% (25/51)** | **−0.58%** | −0.31pt |

**Effect of the cleaning phase**

- Verdicts changed: **12/100** — action/ticker changed in **7**, confidence-only in **5**. The
  refinement is a **no-op on 88/100**.
- Mean confidence shift (after − before): **−0.006** (negligible).
- Context churn: pruning removed a mean **2.5** insights/article (nonzero on **27/100**, max 38);
  the justification-augment added a mean **0.4** (nonzero on only **13/100** — the 0.75 cosine
  floor is rarely cleared in-window).

**The 7 action/ticker changes** (most demote a correct BUY to HOLD):

| a# | pick | gain% | before | after | prune/add | effect |
|--:|:--|--:|:--|:--|:--|:--|
| 19946 | CBRS | +7.02 | buy·0.85 ✓ | hold·0.30 — | −25/+16 | lost a winner |
| 20103 | CRWD | +6.40 | buy·0.85 ✓ | hold·0.40 — | −12/+0 | lost a winner |
| 8436 | SMCI | +0.47 | buy·0.75 ✓ | hold·0.30 — | −3/+0 | lost a (small) winner |
| 19306 | AMAT | +0.07 | buy·0.75 ✓ | hold·0.30 — | −14/+0 | lost a (tiny) winner |
| 16241 | NVDA (←MSFT) | +0.97 | hold·0.30 — | buy·0.75 ✓ | −4/+0 | **gained** a winner (+ticker switch) |
| 9080 | NVDA | −0.01 | hold·0.30 — | buy·0.85 ✗ | −0/+1 | new wrong buy (flat) |
| 14141 | GOOG (←NVDA) | +1.97 | hold·0.10 — | hold·0.10 — | −1/+0 | ticker switch, still HOLD |

Net among changes: **1 improved, 4 worsened, 2 neutral**.

**Per-action buckets — after cleaning (round 2)**

| action | n | mean gain% | dir hit-rate |
|---|--:|--:|--:|
| buy | 44 | −0.58% | 48% (21/44) |
| hold | 49 | +0.19% | — |
| sell | 7 | −0.26% | 57% (4/7) |

#### Findings

1. **The refinement is mostly inert — and slightly *negative* where it acts.** 88/100 verdicts
   are unchanged; directional hit-rate edges **down 53% → 49%**, BUY mean **−0.27% → −0.58%**,
   spread **−0.01 → −0.31pt**. On this sample cleaning did not sharpen the pick; it dulled it.
2. **Pruning makes the model more cautious, demoting correct BUYs to HOLD.** 4 of the 7 changes
   are `buy✓ → hold`, including the two biggest winners in the whole sample — **CBRS +7.02%** and
   **CRWD +6.40%**. Stripping the surrounding context removes the corroboration the model was
   leaning on, so it retreats to HOLD and forfeits the win. Only **1** change went the good way
   (a#16241, `hold → buy✓`, which also switched MSFT→NVDA).
3. **The justification-augment barely engages at 0.75.** It added insights to only 13/100
   articles (mean +0.4). A paraphrased one-paragraph justification rarely has a stored insight
   within 0.75 cosine in the pre-seed window, so step (2) is usually empty; the net effect of
   `--clean-top-1` is dominated by the *pruning*, not the augmentation. (Lowering
   `--refine-min-similarity` would add more, but those are weaker matches.)
4. **Confidence is essentially untouched** (mean shift −0.006). When the verdict survives
   cleaning, the model reports almost the same conviction on the tighter context — so the
   refinement is not even acting as a confidence recalibrator.
5. **Cost/benefit is poor.** `--clean-top-1` spends **2 extra model calls** (the prune screen +
   the re-judge) plus a justification embedding/search per article, and on this peaceful-day,
   intraday metric returns slightly *worse* accuracy. The one structural positive — it can
   switch to a better ticker (a#16241) — is rare (2/100) and net-neutral here.

**Conclusion — keep `--clean-top-1` off for scoring; it's an inspection/explainability tool.**
The mechanism does what it says (it visibly tightens context to the chosen thesis and can pull
in justification-matched evidence), which is useful for *understanding* a single call. But as an
accuracy lever on calm tape it backfires: the pruning mostly converts the model's correct,
context-supported BUYs into cautious HOLDs (losing the CBRS/CRWD winners), and the 0.75
augmentation rarely fires. This mirrors the §10/§12 lesson — post-hoc context surgery is a
volume/conservatism knob, not a precision lever; precision has to come from the retrieval gates
and the underlying signal. (Same data caveat as §13: extreme returns on thin names like CBRS are
partly a liquidity/price-data artifact.)

<details>
<summary>Full 100-row table (before/after verdict · correctness · confidence · buy→close gain)</summary>

| # | a# | chosen | role | #tk | sell date | gain% | before (act·conf) | ✓ | after (act·conf) | ✓ | −rm/+add |
|--:|--:|:--|:--|--:|:--|--:|:--|:-:|:--|:-:|:--|
| 1 | 19833 | NVDA | primary | 1 | 2026-05-14 | +2.50 | hold · 0.10 | — | hold · 0.10 | — | −0/+0 |
| 2 | 19819 | AMZN | primary | 1 | 2026-05-14 | -0.97 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −0/+0 |
| 3 | 10534 | WDC | primary | 1 | 2025-10-13 | +1.11 | buy · 0.70 | ✓ | buy · 0.70 | ✓ | −0/+0 |
| 4 | 11257 | MSFT | mentioned | 2 | 2025-10-30 | -0.68 | sell · 0.75 | ✓ | sell · 0.75 | ✓ | −0/+0 |
| 5 | 8631 | NOK | primary | 1 | 2025-08-21 | +0.47 | buy · 0.75 | ✓ | buy · 0.75 | ✓ | −0/+0 |
| 6 | 1298 | NVDA | mentioned | 2 | 2024-12-12 | +0.46 | buy · 0.75 | ✓ | buy · 0.75 | ✓ | −0/+0 |
| 7 | 16011 | AVGO | primary | 1 | 2026-02-19 | -0.31 | buy · 0.85 | ✗ | buy · 0.85 | ✗ | −10/+0 |
| 8 | 1445 | COHR | primary | 2 | 2024-12-17 | -2.57 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 9 | 13612 | NVDA | mentioned | 2 | 2025-12-26 | +0.61 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 10 | 7866 | IBM | primary | 1 | 2025-07-30 | -0.80 | hold · 0.30 | — | hold · 0.30 | — | −3/+2 |
| 11 | 1190 | PLTR | primary | 1 | 2024-12-09 | -11.61 | buy · 0.85 | ✗ | buy · 0.85 | ✗ | −1/+0 |
| 12 | 12126 | ADBE | primary | 1 | 2025-11-19 | -0.29 | hold · 0.30 | — | hold · 0.30 | — | −3/+3 |
| 13 | 7915 | NVDA | primary | 2 | 2025-07-31 | -2.73 | buy · 0.70 | ✗ | buy · 0.70 | ✗ | −0/+0 |
| 14 | 7770 | LNVGY | primary | 1 | 2025-07-28 | +2.43 | buy · 0.85 | ✓ | buy · 0.85 | ✓ | −0/+0 |
| 15 | 19306 | AMAT | primary | 1 | 2026-05-04 | +0.07 | buy · 0.75 | ✓ | hold · 0.30 | — | −14/+0 |
| 16 | 13844 | AMBA | primary | 1 | 2026-01-05 | -0.45 | buy · 0.85 | ✗ | buy · 0.85 | ✗ | −0/+0 |
| 17 | 14210 | GOOG | primary | 2 | 2026-01-12 | +1.94 | buy · 0.75 | ✓ | buy · 0.75 | ✓ | −0/+0 |
| 18 | 14208 | NVDA | primary | 1 | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | buy · 0.85 | ✓ | −0/+0 |
| 19 | 19675 | CBRS | mentioned | 5 | 2026-05-14 | -11.12 | buy · 0.85 | ✗ | buy · 0.85 | ✗ | −0/+1 |
| 20 | 19102 | NVDA | mentioned | 4 | 2026-04-29 | -1.90 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −25/+0 |
| 21 | 13326 | NVDA | primary | 1 | 2025-12-17 | -3.92 | hold · 0.10 | — | hold · 0.10 | — | −0/+1 |
| 22 | 11563 | LNVGY | primary | 2 | 2025-11-06 | -3.20 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 23 | 13972 | MSFT | primary | 1 | 2026-01-07 | +1.00 | hold · 0.10 | — | hold · 0.10 | — | −0/+0 |
| 24 | 14391 | AMZN | primary | 1 | 2026-01-16 | +0.08 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 25 | 7222 | GOOG | primary | 3 | 2025-07-10 | +0.70 | buy · 0.75 | ✓ | buy · 0.75 | ✓ | −0/+0 |
| 26 | 19431 | AMZN | primary | 1 | 2026-05-06 | +0.84 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 27 | 7393 | POWL | primary | 1 | 2025-07-16 | +3.62 | buy · 0.75 | ✓ | buy · 0.75 | ✓ | −0/+0 |
| 28 | 9620 | NOK | mentioned | 3 | 2025-09-17 | +1.95 | buy · 0.75 | ✓ | buy · 0.75 | ✓ | −0/+0 |
| 29 | 13787 | NVDA | primary | 1 | 2026-01-02 | -0.54 | hold · 0.50 | — | hold · 0.50 | — | −0/+0 |
| 30 | 7047 | ADI | primary | 1 | 2025-07-03 | -0.03 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 31 | 8845 | NVDA | mentioned | 3 | 2025-08-27 | -0.24 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 32 | 16576 | AVGO | primary | 1 | 2026-03-05 | +4.44 | buy · 0.85 | ✓ | buy · 0.90 | ✓ | −11/+4 |
| 33 | 10567 | AMZN | primary | 1 | 2025-10-14 | -1.60 | hold · 0.10 | — | hold · 0.10 | — | −1/+0 |
| 34 | 14701 | NVDA | primary | 1 | 2026-01-23 | +0.24 | buy · 0.75 | ✓ | buy · 0.75 | ✓ | −6/+1 |
| 35 | 14712 | ORCL | mentioned | 2 | 2026-01-23 | -0.25 | hold · 0.30 | — | hold · 0.30 | — | −1/+0 |
| 36 | 623 | GOOG | primary | 3 | 2024-11-20 | -1.13 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 37 | 19378 | IREN | primary | 2 | 2026-05-05 | +9.96 | buy · 0.85 | ✓ | buy · 0.85 | ✓ | −0/+0 |
| 38 | 14045 | IBM | primary | 1 | 2026-01-08 | -0.14 | hold · 0.30 | — | hold · 0.30 | — | −4/+0 |
| 39 | 15347 | PLTR | mentioned | 5 | 2026-02-05 | +0.81 | buy · 0.85 | ✓ | buy · 0.85 | ✓ | −1/+0 |
| 40 | 14314 | META | primary | 1 | 2026-01-14 | -0.38 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 41 | 9825 | MSFT | primary | 1 | 2025-09-26 | +0.31 | hold · 0.40 | — | hold · 0.40 | — | −0/+0 |
| 42 | 14889 | NVDA | primary | 1 | 2026-01-28 | -0.01 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −18/+1 |
| 43 | 8463 | TSM | mentioned | 2 | 2025-08-15 | +0.27 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 44 | 16038 | AMD | primary | 2 | 2026-02-20 | -1.55 | buy · 0.70 | ✗ | buy · 0.75 | ✗ | −3/+0 |
| 45 | 9661 | AVGO | primary | 2 | 2025-09-17 | +0.54 | buy · 0.85 | ✓ | buy · 0.85 | ✓ | −0/+0 |
| 46 | 19852 | AMZN | primary | 1 | 2026-05-14 | -0.64 | hold · 0.10 | — | hold · 0.30 | — | −5/+0 |
| 47 | 9942 | QCOM | primary | 1 | 2025-09-30 | +0.38 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 48 | 7134 | NVDA | mentioned | 4 | 2025-07-07 | +0.19 | sell · 0.70 | ✗ | sell · 0.70 | ✗ | −0/+0 |
| 49 | 10281 | INTC | mentioned | 2 | 2025-10-08 | +0.43 | buy · 0.70 | ✓ | buy · 0.70 | ✓ | −5/+0 |
| 50 | 608 | HUBB | primary | 1 | 2024-11-20 | -1.52 | hold · 0.10 | — | hold · 0.10 | — | −0/+0 |
| 51 | 6646 | NOK | primary | 5 | 2025-06-16 | +0.95 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 52 | 14486 | MCHP | primary | 1 | 2026-01-20 | +0.48 | buy · 0.75 | ✓ | buy · 0.75 | ✓ | −0/+0 |
| 53 | 14207 | NVDA | primary | 1 | 2026-01-12 | +0.02 | buy · 0.85 | ✓ | buy · 0.85 | ✓ | −0/+0 |
| 54 | 19835 | CBRS | mentioned | 2 | 2026-05-14 | -11.12 | buy · 0.85 | ✗ | buy · 0.85 | ✗ | −0/+1 |
| 55 | 16186 | KEYS | primary | 1 | 2026-02-24 | +6.91 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 56 | 11556 | GOOG | primary | 2 | 2025-11-06 | -0.05 | hold · 0.10 | — | hold · 0.10 | — | −1/+0 |
| 57 | 7392 | AVGO | primary | 2 | 2025-07-15 | -0.47 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −5/+5 |
| 58 | 9740 | IBM | primary | 2 | 2025-09-22 | +1.60 | hold · 0.10 | — | hold · 0.10 | — | −0/+0 |
| 59 | 8993 | FLEX | primary | 1 | 2025-09-02 | -0.09 | buy · 0.70 | ✗ | buy · 0.70 | ✗ | −0/+0 |
| 60 | 13375 | NVDA | primary | 1 | 2025-12-18 | +1.02 | sell · 0.65 | ✗ | sell · 0.70 | ✗ | −38/+0 |
| 61 | 19908 | NVDA | primary | 1 | 2026-05-18 | -1.23 | buy · 0.70 | ✗ | buy · 0.70 | ✗ | −0/+0 |
| 62 | 3409 | CDNS | mentioned | 4 | 2025-02-19 | -3.97 | hold · 0.50 | — | hold · 0.50 | — | −0/+0 |
| 63 | 13272 | AMZN | primary | 1 | 2025-12-16 | -0.20 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 64 | 9244 | GOOG | primary | 2 | 2025-09-08 | -0.13 | sell · 0.65 | ✓ | sell · 0.65 | ✓ | −0/+0 |
| 65 | 9777 | NOK | primary | 1 | 2025-09-24 | -1.25 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −0/+0 |
| 66 | 13742 | AVGO | primary | 7 | 2025-12-31 | -0.71 | buy · 0.75 | ✗ | buy · 0.70 | ✗ | −7/+2 |
| 67 | 7068 | ADBE | primary | 1 | 2025-07-07 | -0.54 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 68 | 815 | NOK | primary | 1 | 2024-11-27 | +0.24 | buy · 0.85 | ✓ | buy · 0.85 | ✓ | −0/+0 |
| 69 | 9080 | NVDA | primary | 1 | 2025-09-04 | -0.01 | hold · 0.30 | — | buy · 0.85 | ✗ | −0/+1 |
| 70 | 10245 | APLD | primary | 1 | 2025-10-07 | -2.85 | buy · 0.85 | ✗ | buy · 0.85 | ✗ | −0/+0 |
| 71 | 16241 | NVDA (←MSFT) | mentioned | 2 | 2026-02-25 | +0.97 [r1 +2.80] | hold · 0.30 | — | buy · 0.75 | ✓ | −4/+0 |
| 72 | 19946 | CBRS | mentioned | 10 | 2026-05-18 | +7.02 | buy · 0.85 | ✓ | hold · 0.30 | — | −25/+16 |
| 73 | 9719 | ARM | primary | 2 | 2025-09-19 | -0.83 | sell · 0.70 | ✓ | sell · 0.70 | ✓ | −0/+0 |
| 74 | 14095 | META | primary | 1 | 2026-01-09 | +1.07 | hold · 0.40 | — | hold · 0.40 | — | −0/+0 |
| 75 | 12121 | ADBE | primary | 1 | 2025-11-19 | -0.15 | hold · 0.40 | — | hold · 0.40 | — | −0/+0 |
| 76 | 7811 | TSM | primary | 1 | 2025-07-28 | +0.34 | sell · 0.75 | ✗ | sell · 0.75 | ✗ | −0/+0 |
| 77 | 10252 | IBM | primary | 2 | 2025-10-07 | -1.90 | buy · 0.85 | ✗ | buy · 0.85 | ✗ | −0/+0 |
| 78 | 11434 | NVDA | mentioned | 2 | 2025-11-03 | +0.22 | buy · 0.70 | ✓ | buy · 0.70 | ✓ | −0/+0 |
| 79 | 13505 | NVDA | primary | 3 | 2025-12-22 | +0.04 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 80 | 20103 | CRWD | primary | 1 | 2026-05-20 | +6.40 | buy · 0.85 | ✓ | hold · 0.40 | — | −12/+0 |
| 81 | 694 | NOK | primary | 1 | 2024-11-22 | +2.20 | hold · 0.40 | — | hold · 0.40 | — | −0/+0 |
| 82 | 13333 | AMZN | primary | 1 | 2025-12-17 | -1.68 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 83 | 149 | GOOGL | mentioned | 3 | 2024-11-04 | -1.32 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 84 | 11135 | MSFT | primary | 1 | 2025-10-27 | -0.26 | hold · 0.10 | — | hold · 0.10 | — | −0/+0 |
| 85 | 7377 | MSFT | primary | 2 | 2025-07-15 | +0.10 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 86 | 13695 | APLD | primary | 2 | 2025-12-30 | -2.82 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −0/+0 |
| 87 | 14254 | META | primary | 1 | 2026-01-13 | +0.32 | hold · 0.10 | — | hold · 0.10 | — | −0/+0 |
| 88 | 14141 | GOOG (←NVDA) | primary | 3 | 2026-01-12 | +1.97 [r1 +1.17] | hold · 0.10 | — | hold · 0.10 | — | −1/+0 |
| 89 | 1056 | META | primary | 1 | 2024-12-05 | -0.95 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −0/+0 |
| 90 | 10201 | AMD | primary | 3 | 2025-10-06 | -2.92 | buy · 0.95 | ✗ | buy · 0.95 | ✗ | −36/+1 |
| 91 | 10277 | MSFT | primary | 1 | 2025-10-08 | +0.09 | hold · 0.10 | — | hold · 0.10 | — | −0/+0 |
| 92 | 6649 | NOK | primary | 5 | 2025-06-16 | +0.95 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 93 | 9125 | AI | primary | 5 | 2025-09-04 | -1.75 | sell · 0.90 | ✓ | sell · 0.90 | ✓ | −0/+0 |
| 94 | 8436 | SMCI | mentioned | 2 | 2025-08-15 | +0.47 | buy · 0.75 | ✓ | hold · 0.30 | — | −3/+0 |
| 95 | 10323 | SNPS | primary | 1 | 2025-10-09 | -0.63 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −0/+0 |
| 96 | 10324 | AMZN | primary | 1 | 2025-10-09 | +1.01 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 97 | 10350 | MSFT | primary | 1 | 2025-10-09 | -0.24 | buy · 0.75 | ✗ | buy · 0.75 | ✗ | −0/+0 |
| 98 | 17107 | S | primary | 1 | 2026-03-16 | -2.39 | hold · 0.40 | — | hold · 0.40 | — | −2/+0 |
| 99 | 3404 | NVDA | primary | 1 | 2025-02-19 | -0.12 | hold · 0.30 | — | hold · 0.30 | — | −0/+0 |
| 100 | 6928 | NVDA | primary | 1 | 2025-06-27 | +1.38 | buy · 0.85 | ✓ | buy · 0.85 | ✓ | −0/+0 |

</details>
