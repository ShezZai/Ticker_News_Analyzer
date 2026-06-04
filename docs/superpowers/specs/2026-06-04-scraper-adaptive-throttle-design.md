# Scraper success-rate optimization: adaptive throttle handling

**Date:** 2026-06-04
**Branch:** `optimize/scraper-adaptive-throttle`
**Goal:** Raise the scraper's success rate (usable article text per URL).

## Diagnosis (from the populated `articles` table, 20,569 rows)

- Status split: `empty` 11,195 (54%), `ok` 9,374 (46%), no `error` rows.
- The `empty` bucket is almost entirely **one domain**:
  - `fool.com`: 11,168 empty / 12,184 total (92% fail) — **all HTTP 429**.
  - Every other domain is near-perfect (investing.com 2095 ok / 9 fail,
    globenewswire 3757 / 4, benzinga 2503 / 1).
- When fool.com returns 200, extraction works fine (http rows avg ~721 words;
  playwright ~488). The extractor is not the problem.
- A single isolated live request to fool.com returns **200** (Cloudflare, no
  `Retry-After` header). The 429s were **load-induced** throttling tripped by our
  burst, not a hard block.

### Root cause
The scraper has no adaptive response to 429:
1. `_http_with_retries` retries only 3× over ~3s — far too short for sustained
   throttling.
2. Every 429 still invokes the Playwright fallback, which shares the same
   rate-limited IP, so it can't help and just burns time (11k wasted launches).
3. A never-clearing 429 is stored as `status='empty'` — misleading, looks
   unrecoverable. It should be a retryable `error`.

## Design

### 1. Adaptive `DomainLimiter` (circuit breaker)
Per-domain dynamic state on top of today's semaphore + delay:
- `penalize(domain, retry_after=None)` — on a 429/503, set
  `paused_until = now + cooldown` and grow `current_delay` exponentially
  (× factor, capped, jittered). Honor `Retry-After` when present.
- `reward(domain)` — on a clean 200, decay `current_delay` toward baseline.
- `slot()` waits until `paused_until` and sleeps `current_delay`.

One 429 slows **every** worker on that domain, then it recovers as 200s return.

### 2. Per-domain config overrides
`Settings` gains a per-domain map (base concurrency, base delay, max delay) from
env JSON `SCRAPER_DOMAIN_OVERRIDES`, with a built-in default tuning `fool.com`
to low concurrency (~1) and higher delay (~3s). `DomainLimiter` uses per-domain
values rather than the single global.

### 3. Smarter retry loop
`_http_with_retries`: more attempts, exponential backoff **with jitter**, honor
`Retry-After`, and call `limiter.penalize()` on a 429 so the breaker engages
across the pool.

### 4. Skip the browser on a throttle
`process_job` distinguishes "throttled (429/503)" from "bad for other reasons".
On a pure throttle it skips `browser_get` (same IP, can't help).

### 5. Correct, retryable labeling
A 429 that never clears -> `status='error'`, `error='rate_limited'`,
`http_status=429`. Plus a one-time migration relabeling the existing
11,168 `empty`/429 rows.

### 6. Clearing the backlog
`exists_ok` only skips `status='ok'`, so a plain re-run re-attempts every failed
fool.com row under the new pacing + breaker. Flow: migrate labels -> rerun ->
backlog drains.
**Stretch:** a `--retry-failed` mode sourcing URLs from the DB instead of
re-reading the whole CSV.

### 7. Tests & observability
Offline unit tests (`not db` marker) for: 429 -> cooldown engages, delay grows
then decays, `Retry-After` honored, browser skipped on 429, label =
`rate_limited`. One-line log on domain entering/exiting cooldown.

## Out of scope (YAGNI)
Proxies / IP rotation, UA rotation, new extractor overrides, any change to the
embedding/search stages.
