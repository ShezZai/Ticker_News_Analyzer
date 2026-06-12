# Insight-Box Labeling Guide

This project labels extracted insight boxes at **two levels**:

1. **The 9-label *nature* taxonomy** — describes *what kind* of box it is
   (event-insight, Market-claim, Fluff-PR …). Descriptive; used to hand-judge
   `insights_GT.csv`. See the **Appendix** below.
2. **The 2-label *distillation* scheme** — the **production** scheme the act-on
   pipeline actually emits: keep a box as **`evidance-event`** or
   **`informative`**, else **DROP** it. This is what `extract_insights_new.py`
   produces and what the `insight_GT` column in the *distilled* files judges.

The single question behind the production scheme is:

> **Would retrieving this box genuinely help judge whether to ACT on a NEW
> breaking-news article?** (the "retrieval-relevant for act-on" test)

If yes → keep (and tier it `evidance-event` / `informative`). If no → DROP.

---

## GT artifacts (what each file holds)

| file | boxes | `insight_GT` meaning |
|---|---|---|
| `insights_GT.csv` | 471, all extracted | 9-label nature taxonomy (Appendix) |
| `insights_GT_distilled.csv` | 223, raw flash-lite extraction | retrieval-relevance: `evidance-event` / `informative` / `DROP` / blank |
| `insights_GT_distilled_gated.csv` | post-gate survivors | same, after the second-pass gate |

`spelling note:` the label is **`evidance-event`** (kept as first written); rename
globally to `evidence-event` if desired.

---

## The production scheme — keep/drop + 2 tiers

Decide in order: **`evidance-event` → `informative` → DROP**. When unsure, DROP.

| label | keep when… | examples |
|---|---|---|
| **`evidance-event`** | a concrete, first-hand, MATERIAL fact that verifiably happened or was reported in a disclosure/filing/release. A fact, bullish or bearish. | reported earnings/guidance/margins; completed M&A; lawsuit filed; recall; layoffs; regulatory ruling; macro release (rates/jobs/inflation/tariffs/GDP); disclosed insider/institutional position; price move on news |
| **`informative`** | ONE discrete ANNOUNCED but soft development. | announced partnership/MoU/LoI/proposed deal; product launch (once); funding round/expansion; planned future action; a single named major up/down-grade that is the news |
| **DROP** | everything else (see the three tests). | promotion, opinion, trends, capabilities, theses, speculation, routine filings, non-financial political news |

### Three tests that decide most DROPs

1. **Thesis / opinion test** — DROP if the box is the author's *argument*, even
   with real numbers: competitive position / market share / "moat" / growth
   story; valuation / price targets / "could-should" / ratings; capex or industry
   TRENDS; product features / capabilities / "how it works"; predictions, theses.
   *Key distinction:* a number is `evidance-event` only when **disclosed in an
   earnings report / filing** ("Q3 revenue rose 19% to $4.2B"); the same number is
   DROP when used to **argue a case** ("~40% market share", "trading at 21x").

2. **Market-mechanism test** (for political / government / legal / geopolitical /
   human-interest items) — many things *happen* without being market evidence.
   Keep ONLY if it acts on a specific company / sector / asset / rate through a
   **concrete channel**. Ask: *"through what channel would this move a specific
   stock, sector, rate, or commodity?"* No clear answer → DROP.
   - KEEP: "US acquires a 10% equity stake in Intel ($11B)"; "25% tariff on
     India"; "Fed holds rates".
   - DROP: "FBI raids a former official's home"; "Navy deploys ships";
     "4.5M voters switched parties"; "White House launches a TikTok account".

3. **One-box rule** — a product launch / deal yields **at most one** box (the
   launch/deal itself). Never add boxes for its features, specs, benefits, or the
   company description.

---

## Mapping: 9-label nature → production scheme

| 9-label (nature) | production |
|---|---|
| event-insight | **evidance-event** |
| event-claim | **informative** |
| Market-insight | DROP (a principle, not a retrievable event) |
| PR-claim, Market-claim | DROP |
| Fluff-PR, Fluff-claim, Fluff-idea, Fluff-event | DROP |

> `Market-insight` was an early "reason-scaff" keeper but was **cut**: as
> retrieval context for an act-judge a generalizable principle isn't a discrete
> event, and in practice the class became a laundering sink for market-trend
> fluff.

---

## The pipeline that produces these labels

- **Prompt:** `distill_insights_prompt.txt` — `PROMPT_TEMPLATE` (extraction,
  emits `evidance-event`/`informative` only) + `GATE_PROMPT_TEMPLATE` (the strict
  per-box second pass, carries the three tests above).
- **Script:** `scripts/enrichment/extract_insights_new.py` — flash-lite
  extraction → CSV (reads DB read-only, never writes it). `--gate` adds the
  second pass; `--gate-model` (default `gemini-2.5-flash`) judges each box.
- **Winning config:** **flash-lite extract + flash gate.** flash-lite extracts
  with high recall (it doesn't over-gate); a *per-box* flash gate gives flash-level
  precision **without** flash's whole-article over-gating (single-pass flash drops
  entire earnings articles). Two-pass also sidesteps flash-lite's run-to-run
  extraction variance, which made single-pass prompt-tuning plateau.

---

## Measured performance (gate as a retrieval-relevance classifier)

On the raw 220-box flash-lite extraction, vs the `insight_GT` relevance GT:

| metric | no gate | **+ gate (flash, market-mechanism)** |
|---|---|---|
| Precision | 67.3% | **87.8%** |
| Recall | 100% | **87.8%** |
| Accuracy | 67.3% | **83.6%** |
| F1 | 80.5% | **87.8%** |

Confusion: TP 130 · FP 18 · FN 18 · TN 54. The gate removes 54 of 72 non-relevant
boxes at a cost of 18 dropped keepers; precision = recall (balanced). Tightening
the market-mechanism rule trades ~5 pts recall for closing the political-noise
blind spot (e.g. the `8721` roundup: 16 kept → 3, keeping only the Intel stake /
trade deal / tariff revenue).

---

## Conventions

- Articles with **no surviving boxes** appear as a single placeholder row marked
  `(no insights extracted)`; leave their `insight_GT` blank.
- Leave a cell **blank** when genuinely uncertain (don't force a call).
- Quotes are always verbatim source substrings (enforced post-extraction); a
  quote that can't be matched is dropped.

---

# Appendix — the 9-label nature taxonomy (`insights_GT.csv`)

A label is a compound of **value tier** × **subject**.

**Value tier:** `insight` (material + verifiable → signal) · `claim` (asserted,
promotional/unverified → discount) · `Fluff` (no material content → ignore).
**Subject:** `event` (discrete dated occurrence) · `Market` (generalizable
principle) · `PR` (self-promotion) · `idea` (abstract concept).

### Tier 1 — `insight`
| label | definition | example |
|---|---|---|
| **event-insight** | A discrete, dated occurrence with material consequence. | lawsuit filed; late 10-K; demand-softness reported |
| **Market-insight** | A reusable, sound investing principle that holds regardless of ticker. | cash-velocity metric; "automate last" |

### Tier 2 — `claim`
| label | definition | example |
|---|---|---|
| **PR-claim** | Promotional assertion with a hard particular. | "partnership with Microsoft"; "award winners: MariaDB, Grafana" |
| **event-claim** | A real-but-soft event, terms self-reported. | "signs MoU"; "to acquire X" with no figures |
| **Market-claim** | A generalizable market assertion stated as fact but unsupported. | "AI adds $X trillion to GDP"; "market to hit 220 ZB by 2026" |

### Tier 3 — `Fluff`
| label | definition | example |
|---|---|---|
| **Fluff-PR** | Self-congratulation — own award, CEO self-praise. | "received Platinum Winner" |
| **Fluff-claim** | Promotional benefit/use-case narrative, no verifiable fact. | "accelerates time-to-market, scalability" |
| **Fluff-idea** | Abstract thesis/framework, motivational. | a book's "five-step algorithm" |
| **Fluff-event** | A real but immaterial happening. | "to present at conference on May 3"; routine NAV/buyback notice |

### Boundary tests
- **event-insight → event-claim → PR-claim**: material+verifiable → real-but-soft
  → just an assertion.
- **PR-claim vs Fluff-PR**: third-party fact vs self-congratulation.
- **PR-claim vs Fluff-claim**: a hard particular vs pure benefit language.
- **Market-insight vs Market-claim**: supported principle vs unbacked assertion.
- **Fluff-event vs event-insight**: a happening that *doesn't move anything* vs
  one with material consequence.

### Collapse rules (don't invent cells)
| tempted by… | use instead |
|---|---|
| PR-insight | event-insight / Market-insight |
| idea-insight | Market-insight |
| idea-claim | Fluff-idea |
| Market-fluff | Fluff-idea |
