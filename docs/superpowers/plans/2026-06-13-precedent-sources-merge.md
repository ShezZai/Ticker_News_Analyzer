# Precedent-Sources Merge + Langfuse Observability Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Merge the partner branch `origin/feat/pipline-v2_improved` (configurable precedent retrieval for the sentiment historical-precedent analyst: `article` / `insights` / `distilled-first` / `distilled-second`) into `refactor/pipeline-v2`, with the precedent-source choice **clearly traced in Langfuse**, via an integration branch and a grouped-commit merge-back.

**Branch strategy (user-mandated):** all work happens on integration branch `integrate/precedent-sources` (off `refactor/pipeline-v2`). Merge-back is grouped: (1) a `--no-ff` merge of the partner branch (history preserved — don't squash someone else's commits), (2) our hardening squashed into one grouped commit via `git merge --squash` of the integration branch.

**Verified facts:**
- Merge base `ac58a31`; partner has 7 commits, we have 15. Overlapping files: only `cli.py` (different commands) and `evals/pipeline_eval.py` (different functions) — trivial or no conflicts expected. They touched `service/stages.py`, `sentiment/analysts.py`, `shared/config.py`, tests, docs; we didn't.
- **Observability gaps found in audit:**
  (a) Langfuse `production` copy of `analyst-historical_precedent` predates the new `{own_insights}`/`{label_legend}` placeholders; `str.format` ignores extra kwargs, so the stale remote template **silently drops the new sections** until `ticker-news prompts push` + process restart.
  (b) `metadata["precedent_source"]` is recorded on eval runs only when the CLI flag is passed — config-default runs and the live service record nothing.
  (c) `gather_precedents` + `own_article_insights` (up to 40+ ANN queries over the tunneled DB) are invisible — no span.
- `obs.stage_span(name)` yields a span (or None when disabled) supporting `.update(input=..., output=..., metadata=...)`. A new child span under `sentiment` is additive — does not violate the stable-observation-names contract.
- Partner's `sentiment_stage(conn, url, precedent_source=None)` computes `mode = precedent_source or settings.precedent_source` and calls `gather_precedents(conn, aid, source=mode)` + `own_article_insights(...)` for insight modes.

---

### Task 0: Integration branch + merge

- [ ] `git checkout -b integrate/precedent-sources refactor/pipeline-v2`
- [ ] `git merge --no-ff origin/feat/pipline-v2_improved -m "merge: precedent-source options for the sentiment analyst panel"`
- [ ] Resolve conflicts if any (expected at worst in `cli.py` import/option regions and `pipeline_eval.py` — keep BOTH sides: our eval-classify changes + their precedent-source/skip-all changes). Record what conflicted.
- [ ] Full offline suite: `.venv/Scripts/python.exe -m pytest -m "not db and not integration" -q` → must be fully green (their branch added tests too; if their tests conflict with our Task-0..7 changes, fix forward and note it).
- [ ] Commit any conflict-resolution fixes separately on the integration branch.

### Task 1: Observability hardening (TDD)

**Files:** `src/ticker_news/service/stages.py`, `src/ticker_news/evals/pipeline_eval.py`; tests `tests/service/test_stages.py`, `tests/evals/test_pipeline_eval.py`.

- [ ] **1a — `precedents` child span.** In `sentiment_stage`, wrap the retrieval block (`gather_precedents` + `own_article_insights`) in `obs.stage_span("precedents")`; after retrieval, when the span is not None call:
```python
span.update(
    output={"n_precedents": len(precedents), "n_own_insights": len(own_insights)},
    metadata={
        "precedent_source": mode,
        "threshold": settings.precedent_insights_threshold,
        "limit": settings.precedent_insights_limit,
    },
)
```
(threshold/limit only meaningful for insight modes — include unconditionally; cheap and consistent.) Test: monkeypatch `ticker_news.service.stages.obs.stage_span` with a recording fake (contextmanager yielding a stub with `.update`) and assert it was entered with name `"precedents"` and updated with the right keys, in both an insight mode and `article` mode (existing test_stages fixtures/fakes for sentiment_stage show the pattern — extend them).
- [ ] **1b — always-record effective source on eval runs.** In `pipeline_eval.run_eval`, replace the conditional `if precedent_source: metadata[...]` with an unconditional effective value:
```python
from ticker_news.shared.config import get_settings
metadata["precedent_source"] = precedent_source or get_settings().precedent_source
```
Test: extend the existing run_eval metadata test (or add one) asserting `precedent_source` present when the arg is None (equals the settings default, `"article"`).
- [ ] Full offline suite green; commit (these are squashed at merge-back, granular commits fine).

### Task 2: Prompts + docs

- [ ] `ticker-news prompts push` (publishes the changed `analyst-historical_precedent` fallback to Langfuse `production`). Verify: `npx langfuse-cli api prompts ...` or a `get_prompt` call shows a bumped version containing `{own_insights}`.
- [ ] CLAUDE.md additions (Configuration + Commands + Observability):
  - Config bullet: `SENTIMENT_PRECEDENT_SOURCE` (`article`|`insights`|`distilled-first`|`distilled-second`, default `article`) + `..._THRESHOLD`/`..._LIMIT` knobs.
  - `eval pipeline` row: add `--precedent-source` + `--skip-stages all`.
  - Observability: `precedents` child span under `sentiment` carries the effective source + counts; eval runs always record `metadata.precedent_source`; pointer to `docs/precedent-source-options.md`.
- [ ] Commit docs.

### Task 3: Live verification (tunnel DB + Langfuse)

- [ ] `distilled_article_insights` exists + populated over the tunnel (one COUNT query).
- [ ] Smoke A/B: pick 2–3 article ids that have verdicts and distilled boxes; run
  `ticker-news eval classify` is NOT involved — use:
  `ticker-news eval pipeline --ids <ids> --skip-stages all --precedent-source article --run-name prec-smoke-article`
  then same with `--precedent-source distilled-second --run-name prec-smoke-distilled2`.
- [ ] Verify in Langfuse: both runs exist with `metadata.precedent_source` set; a trace shows the `precedents` span (under `sentiment`) with source/counts; `prompt_versions` shows the bumped `analyst-historical_precedent` version; the rendered analyst prompt contains the own-insights/legend sections in distilled mode.

### Task 4: Merge back (grouped commits)

- [ ] `git checkout refactor/pipeline-v2`
- [ ] `git merge --no-ff origin/feat/pipline-v2_improved -m "merge: precedent-source options for the sentiment analyst panel"` — re-resolve the (trivial) conflicts identically to Task 0.
- [ ] `git merge --squash integrate/precedent-sources` → stages only our delta (hardening + docs + any conflict fixes); commit as ONE grouped commit:
  `feat: trace precedent-source in Langfuse (precedents span, effective-source run metadata)`
  — if the staged delta cleanly separates docs from code, two commits (code, docs) are acceptable.
- [ ] Full offline suite green on `refactor/pipeline-v2`; `git branch -D integrate/precedent-sources`.

**Rules:** clean commit messages, no attribution trailers. Production observation names (`process-article`, `sentiment`, `analyst:<role>`, ...) must NOT be renamed — `precedents` is a new child, allowed.
