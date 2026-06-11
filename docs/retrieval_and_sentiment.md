# Retrieval method & sentiment experiments

A focused summary of two design questions: **how related coverage is retrieved**, and **whether the
sentiment verdict carries real signal**. Full detail lives in
[`validations/validate_retrieval.md`](validations/validate_retrieval.md) and
[`validations/sentiment_validation.md`](validations/sentiment_validation.md).

---

## 1. Choosing the retrieval method

To judge a new article, the system retrieves *related prior coverage* to gauge how surprising the news
is. Two retrievers are **complementary**, and validation made the trade-off concrete:

| Retriever | Searches | Strength |
|---|---|---|
| whole-article | `articles.embedding` | higher **recall** — nets the right documents |
| insight-level | `article_insights.embedding` | higher **precision** — matches the right *ideas* |

> **The governing rule.** Whichever method is applied *last as a hard gate* caps recall at that
> method's recall. So the high-recall method must build the wide net, and the high-precision method
> must filter inside it — never the reverse.

### Methods evaluated

| Method | Idea | Trade-off |
|---|---|---|
| `intersection` | Keep insights whose article is in **both** top-`k` lists. | Max precision; recall ≤ insight-alone (the naive overlap — an anti-pattern). |
| `cascade` | Wide article net → keep net insights clearing `tau_ins`. | Recall ceiling = whole-article (high); precision restored by the insight filter. |
| `fusion` | RRF of insight-ANN cosine **and** the insight's article-rank in the net. | Highest recall ceiling (a union); precision set by the cutoff. |
| **`two-phase-similarity`** | Two stages — a cascade *pool*, then *fusion within* the pool. | Recall ceiling = the pool (high); precision from a per-insight gate + rerank + budget. |

### Selected: `two-phase-similarity`

A two-stage hybrid that embodies the rule:

1. **Stage 1 — pool (recall):** a wide whole-article net (`net_k=150`, loose article floor
   `net_min_sim=0.55`) → keep the articles that have ≥ 1 insight on-topic to the seed.
2. **Stage 2 — rerank (precision):** of the pool articles' insights, keep only those that themselves
   clear `tau_ins=0.75`, RRF-rank them (insight-cosine rank ⊕ article net rank), and keep the top
   `budget=40`.

The **loose article floor (0.55)** fills the pool for recall; the **tight insight gate (0.75)** holds
precision on what survives — an *asymmetric* pair of floors.

**Why it was chosen** (across both ground-truth tables and both evaluation modes):

- **Best where it matters most — the insight level.** The sentiment prompt consumes *insights*; at that
  level `two-phase-similarity` beats the baselines in 3 of 4 conditions.
- **Strongest in the forward / cluster-recovery frame**, with the family's best macro-F1.
- **Degrades gracefully** — never the worst method; its one weak corner is the strict no-lookahead,
  article-level case, where it stays mid-pack rather than collapsing.

The tuning path (documented in `validate_retrieval.md` §10) showed `0.55/0.75/40` beats both the wider
`0.45/0.70/50` point (on precision balance) and the tighter symmetric `0.7` / `0.8` points (which
starve recall).

**Default config:** `net_min_sim 0.55 · tau_ins 0.75 · budget 40 · net_k 150 · rrf_c 60`.

```python
from hybrid_retrieval import retrieve
res = retrieve(seed_id, method="two-phase-similarity", months_before=3, exclusive=True)
#   res.insight_ids / res.article_ids  → feed into the sentiment prompt
```

---

## 2. Sentiment experiments

The model reads the article + retrieved prior insights under a **strict timeline** (the article is
"now"; prior insights are already-priced past, used only to judge *surprise*) and returns
**buy / sell / hold** + a confidence.

**The control.** To isolate *article* signal from market drift, verdicts are tested on a **"peaceful-days"
pool** — days with `VIX < 25` and a flat 10-day tape — and scored against the realized
**buy-at-publish → sell-at-close** return. On a trending day a winning "buy" may just be the market
rising; on calm tape a correct call must come from the news itself.

### What worked

- **A real, article-specific edge.** On 50 calm-day articles, BUY calls hit **~72%** — well above the
  50% coin-flip — with the index move filtered out. BUY mean return was positive against a ~flat HOLD
  baseline.
- **`two-phase-similarity` sharpened the verdict, not just retrieval.** It was strictly better than the
  insight retriever: it kept all of its correct buys and added a few more — e.g. **AVGO +4.44%**,
  **LNVGY +2.43%** — that the simpler retriever had sat out as "hold".

### What didn't — "more conviction" ≠ more accuracy

Two prompt levers were tested and **rejected by default**:

| Lever | Sample | Effect | Verdict |
|---|---|---|---|
| `--include-strong` (adds `strong_buy`/`strong_sell`) | 150 calm | More decisive but **less** accurate: hit-rate **54% → 47%**; `strong_buy` was anti-calibrated (it *lost* money on average, worse than a plain buy). | reject |
| `--include-bias` (caveat that insights skew bullish) | 100 calm | Roughly **neutral**: it prunes some buys, but the surviving buys have the **same** precision (~52%). A caveat is a caution/volume knob, not an accuracy lever. | off by default |
| `--include-bias` **+** `--include-strong` | 100 calm | **Worst of all** — `strong`'s decisiveness overpowers the caveat: more buys, hit-rate **46%**, bearish BUY−SELL spread flips to **−1.85pt**. | don't combine |

A more *constructive* rewrite of the bias caveat (triage by materiality: react to earnings/M&A/quantified
deals, hold on soft PR, amplify downplayed negatives) made **per-article** decisions visibly better
(e.g. it correctly held a PR-style partnership that fell −6.35%), but the **aggregate** BUY precision
stayed flat (~52%) — confirming precision is set by the retrieval gates and the news, not the prompt.

### The market-regime confound

The same model on a **bearish** window (Jan–Mar 2025, Nasdaq −10%) saw its BUY hit-rate collapse from
~72% toward coin-flip, while it began (correctly) issuing **SELLs** it never produced on calm days.
Long-side accuracy is **regime-sensitive** — which is exactly why the peaceful-days filter exists: it
holds market direction flat so a hit-rate can be read as *signal* rather than drift.

### Honest caveats

- The realized metric is **intraday** (buy→publish-price → close); multi-week regime trends barely show
  in same-day P&L.
- **Directional samples are small** (tens of buys/sells), so these are **directional findings**, not
  significance tests.
- The durable takeaway is qualitative and consistent across runs: **precision comes from the retrieval
  gates and the signal itself — not from prompt embellishments.**
