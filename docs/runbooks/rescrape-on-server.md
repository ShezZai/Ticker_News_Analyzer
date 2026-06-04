# Runbook: rebuild the `articles` table with the optimized scraper

**Audience:** an agent/operator running on the server.
**Goal:** repopulate `public.articles` by scraping the full news CSV with the new
adaptive-throttle scraper, then re-embed and re-tag.

**Context:** the previous `articles` data was lost. The scraper on branch
`optimize/scraper-adaptive-throttle` fixes the root cause of the old 92% failure
rate on `fool.com` (HTTP 429 rate-limiting): it now backs off per-domain on 429,
skips the useless browser fallback on throttles, and labels unresolved throttles
as a retryable `error/rate_limited` instead of `empty`. A 12-URL fool.com
validation run scored 12/12 ok (avg 814 words).

---

## 0. Prerequisites

- Repo checked out on branch **`optimize/scraper-adaptive-throttle`**.
- The news CSV present. Columns must be exactly:
  `tickers,article_url,published_utc,publisher_name`
  (the rebuild dataset is `ai_compute_articles_unique.csv`, 20,569 unique URLs:
  fool.com 12,184 · globenewswire 3,761 · benzinga 2,504 · investing 2,104 · misc 16).
- `.env` with `MASSIVE_API_KEY` (only needed for embedding/candles, not scraping).

```bash
git fetch && git checkout optimize/scraper-adaptive-throttle
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium                        # browser fallback for non-fool domains
```

## 1. Bring up Postgres

```bash
docker compose up -d        # pgvector/pgvector:pg16, db=news, user/pass scraper, :5432
```
The scraper applies `scraper/store/schema.sql` automatically on first run.

> ⚠️ **Do NOT run `pytest -m db` against the `news` database.** Historically the
> test fixture truncated whatever DB it pointed at — that is what destroyed the
> data. It is now isolated to a `news_test` DB, but never override
> `SCRAPER_TEST_DB_DSN` back to `news`.

## 2. Run the scrape

Pick a pacing for fool.com (12k of the 20k URLs and the long pole; all other
domains run fast regardless). The adaptive circuit breaker auto-backs-off if 429s
appear, so either option is safe — moderate is just closer to the throttle line.

**Conservative — built-in default, ~10h, lowest 429 risk (recommended):**
```bash
python run_scrape.py --csv ai_compute_articles_unique.csv
```
(fool.com defaults to 1 request / 3s via the built-in `domain_overrides`.)

**Moderate — ~3–4h:**
```bash
# bash
export SCRAPER_DOMAIN_OVERRIDES='{"fool.com":{"per_domain":2,"delay":1.5,"max_delay":60}}'
python run_scrape.py --csv ai_compute_articles_unique.csv
```
```powershell
# PowerShell
$env:SCRAPER_DOMAIN_OVERRIDES = '{"fool.com":{"per_domain":2,"delay":1.5,"max_delay":60}}'
python run_scrape.py --csv ai_compute_articles_unique.csv
```

Notes:
- **Resumable.** Safe to Ctrl+C and re-run; rows already `status='ok'` are skipped
  (autocommit, one row per transaction). Progress prints every 50 rows.
- Run it under `nohup`/`tmux`/a background job so it survives disconnects, e.g.
  `nohup python run_scrape.py --csv ai_compute_articles_unique.csv > scrape.log 2>&1 &`
  then `tail -f scrape.log`.
- Other tunables (env): `SCRAPER_CONCURRENCY` (default 8 workers),
  `SCRAPER_HTTP_MAX_RETRIES` (5), `SCRAPER_HTTP_BACKOFF_MAX` (30s).

## 3. Drain any rate-limited stragglers

If the run ends with some `error/rate_limited` rows (a 429 burst that never
cleared), relabel-then-retry just those — no need to re-read the whole CSV:

```bash
python scripts/maintenance/relabel_rate_limited.py --dry-run   # how many remain
python run_scrape.py --retry-failed                            # re-scrape only those
```
Repeat `--retry-failed` until the count is ~0. (Pace it slower if needed, e.g.
keep the conservative default or lower `delay` further via the override.)

## 4. Verify

```bash
docker exec news_pg psql -U scraper -d news -c "
  SELECT status, count(*) FROM articles GROUP BY status ORDER BY 2 DESC;"
docker exec news_pg psql -U scraper -d news -c "
  SELECT source_domain,
         count(*) FILTER (WHERE status='ok')    AS ok,
         count(*) FILTER (WHERE status='empty')  AS empty,
         count(*) FILTER (WHERE status='error')  AS err
  FROM articles GROUP BY source_domain ORDER BY 1;"
```
Target: the vast majority `ok`. fool.com should now be mostly `ok` (was 92%
failing). A residual handful of genuine 404s/dead links is expected.

## 5. Re-embed and re-tag

```bash
python scripts/embedding/embed_articles.py        # adds embedding vector(1024) + HNSW index
python scripts/enrichment/tag_segments.py          # primary_ticker / primary_segment / ...
```
(Both are incremental/idempotent. `embed_articles.py` only embeds rows where
`embedding IS NULL` unless `--reembed`.)

## 6. Sanity-check search

```bash
PYTHONPATH=scripts/embedding python scripts/search/search_articles.py \
  "nvidia data center demand" --k 5 --ticker NVDA
```

---

### Rollback / safety
- The scrape only writes to `articles`; re-running is idempotent (upsert on `url`).
- Keep `archive_mode` in mind: the dev DB has no WAL archiving. If this data
  becomes valuable, take a `pg_dump`:
  `docker exec news_pg pg_dump -U scraper news > news_backup.sql`
