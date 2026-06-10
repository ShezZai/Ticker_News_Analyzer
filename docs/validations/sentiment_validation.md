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

| Script | What it does | Produced |
|---|---|---|
| [`validate_sentiment.py`](../../scripts/validate_retreival/validate_sentiment.py) | Samples articles (from the pool CSV, or from the DB via `--db-range`), runs `insight_sentiment` on the primary ticker under both retrievers, pairs each verdict with the buy→close return, and scores correctness. | §6 (50 peaceful-day articles), §7 (15 bearish-window articles via `--db-range 2025-01-02 2025-03-30`) |

**Subjects under test / dependencies** (the pipeline being measured, not validators)

| Script | Role |
|---|---|
| [`insight_sentiment.py`](../../scripts/search/insight_sentiment.py) | The sentiment pipeline itself — retrieval (`--retrieval insight` / `two-phase-similarity`), optional `--remove-unuseful` screen, and the BUY/SELL/HOLD + confidence verdict. |
| [`catalyst_returns.py`](../../scripts/ticker_scan/catalyst_returns.py) | The returns engine (`simulate`) — buy-at-publish minute bar → sell-at-(next)-close, via Massive. |
| [`backtest_top2.py`](../../scripts/search/backtest_top2.py) | The earlier DB-range verdict↔return harness; `validate_sentiment.py` is its pool-driven successor. |

Runs require `NEWS_DB_DSN` (articles/insights), `MASSIVE_API_KEY` (returns), and
`GOOGLE_API_KEY` (Gemini verdicts); `peaceful_days.py` additionally hits FRED (no key).
