# Retrieval method & sentiment experiments

A recap of two design questions: **how related coverage is retrieved**, and **whether the sentiment
verdict carries real signal**. Full detail: [`validations/validate_retrieval.md`](validations/validate_retrieval.md)
and [`validations/sentiment_validation.md`](validations/sentiment_validation.md).

---

## Choosing the retrieval method

To judge a new article, the system retrieves *related prior coverage*. Two retrievers are
complementary, and validation made the trade-off concrete:

| Retriever | Searches | Strength |
|---|---|---|
| whole-article | `articles.embedding` | higher **recall** — nets the right documents |
| insight-level | `article_insights.embedding` | higher **precision** — matches the right *ideas* |

> 🔑 **The governing rule.** Whichever method is applied *last as a hard gate* caps recall at that
> method's recall. So the high-recall method must build the wide net, and the high-precision method
> must filter inside it — never the reverse.

### Selected: `two-phase-similarity`

A two-stage hybrid that embodies the rule:

```
Stage 1 — pool (recall)                ▶   Stage 2 — rerank (precision)
wide whole-article net (net_k 150,         keep pool insights clearing tau 0.75,
floor 0.55) → keep articles with an        RRF-rank, keep top 40
on-topic insight
```

The **loose article floor (0.55)** fills the pool for recall; the **tight insight gate (0.75)** holds
precision on what survives. Across both ground-truth tables and both evaluation modes it was the best
all-rounder — strongest at the **insight level** (what the sentiment prompt actually consumes) and in
the forward / cluster-recovery frame, degrading only to mid-pack in the strict no-lookahead corner.

**Default config:** `net_min_sim 0.55 · tau 0.75 · budget 40 · net_k 150`.

---

## Sentiment experiments — what worked, what didn't

The model reads the article + retrieved prior insights under a **strict timeline** (article = "now",
prior insights = already-priced past) and returns **buy / sell / hold** + confidence. To isolate
*article* signal from market drift, verdicts are tested on a **"peaceful-days" pool** (`VIX < 25` and a
flat 10-day tape) and scored against the realized **buy→close** return.

> ✅ **WORKED — a real, article-specific edge, and the two-phase retriever sharpens it.**
> On 50 calm-day articles, BUY calls hit **~72%** (well above the 50% coin-flip) with the index move
> filtered out — so the verdict is reacting to the news, not market beta. `two-phase-similarity` was
> strictly better than the insight retriever: it kept all of its correct buys and added a few more
> (e.g. **AVGO +4.44%**, **LNVGY +2.43%**) that the simpler retriever had sat out as "hold".

> ❌ **DIDN'T — more "conviction" did not mean more accuracy.**
> Two prompt levers were tested on 100–150 articles and **rejected by default**:
> - **`--include-strong`** (adds `strong_buy`/`strong_sell`): more decisive but *less* accurate —
>   hit-rate fell **54% → 47%**, and `strong_buy` was anti-calibrated (it *lost* money on average,
>   worse than a plain buy).
> - **`--include-bias`** (caveat that insights skew bullish): roughly neutral — it pruned some buys but
>   the surviving buys had the **same** precision (~52%). A prompt caveat is a caution/volume knob, not
>   an accuracy lever. Stacking both flags was the worst of all.

> ℹ️ **CONTEXT — why "peaceful days": the market-regime confound.**
> The same model on a **bearish** window (Jan–Mar 2025, Nasdaq −10%) saw its BUY hit-rate collapse from
> ~72% toward coin-flip, while it began (correctly) issuing SELLs. Long-side accuracy is
> **regime-sensitive** — which is exactly why the calm-day filter exists: it holds market direction flat
> so a hit-rate can be read as *signal* rather than drift.

*Honest caveat: the realized metric is intraday (buy→publish-price → close) and the directional samples
are small, so these read as **directional findings**, not significance tests. The durable takeaway is
qualitative and consistent across runs — precision comes from the retrieval gates and the news itself,
not from prompt embellishments.*

---

## Takeaways

- **Retrieval design beats prompt tricks.** `two-phase-similarity` (loose recall pool → tight precision
  gate) is the selected method; prompt levers that add "conviction" did not improve correctness.
- **Measure against the regime.** A directional edge only means something once market drift is
  controlled for — the peaceful-days pool is the control that makes the ~72% BUY hit-rate meaningful.
