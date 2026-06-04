# Runbook: run the scraper on the server against the shared Postgres

**Audience:** the agent running on the server (the one that built the `shared-pg`
Postgres and restored the `news` dump into database `sharedproject`).
**Goal:** recover the fool.com rate-limit backlog in the restored `articles`
table using the optimized scraper, and keep it runnable.

## Background (what the optimization does)
The restored `articles` table is the old scrape: ~9,374 `ok` rows and ~11,195
`empty` rows that are really HTTP 429 throttles from `fool.com` (92% of fool.com
failed). Branch `optimize/scraper-adaptive-throttle` fixes this: adaptive
per-domain 429 backoff + circuit breaker, it skips the useless browser fallback
on a throttle, and it labels unresolved 429s as retryable `error/rate_limited`
instead of `empty`. Validated locally: 12/12 fool.com URLs recovered, avg 814
words, no browser needed.

**Key point: recovery needs NO CSV.** `--retry-failed` sources jobs (url,
tickers, published_utc, publisher) directly from the `error/rate_limited` rows.

## 0. Get the code
```bash
git clone https://github.com/ShezZai/Ticker_News_Analyzer.git
cd Ticker_News_Analyzer
git checkout optimize/scraper-adaptive-throttle
```

## 1. Python + deps  ⚠️ needs Python 3.10+
The code uses PEP 604 unions (`str | None`) at runtime, so Ubuntu 20.04's default
Python 3.8 will NOT work. Use 3.10+ (pyenv / deadsnakes / `uv`).
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# Optional: only a handful of non-fool 401/403 rows would use it; the fool.com
# backlog never does (browser is skipped on a throttle).
# playwright install chromium
```

## 2. Point the scraper at the shared DB
The scraper reads `SCRAPER_DB_DSN`. Target database `sharedproject`, user
`appuser`; password is in k8s secret `shared-pg/postgres-credentials`
(`POSTGRES_PASSWORD`) and in `~/claude-workspace/.shared-pg-password.txt`.

Pick a connectivity path:
- **Simplest — port-forward on the host:**
  ```bash
  kubectl -n shared-pg port-forward svc/postgres 5432:5432 >/tmp/pf.log 2>&1 &
  PW=$(cat ~/claude-workspace/.shared-pg-password.txt)
  export SCRAPER_DB_DSN="postgresql://appuser:${PW}@localhost:5432/sharedproject"
  ```
- **In-cluster** (if you run the scraper as a Job/pod): use
  `postgresql://appuser:<pw>@postgres.shared-pg.svc.cluster.local:5432/sharedproject`.

Sanity check:
```bash
python -c "import os,psycopg;c=psycopg.connect(os.environ['SCRAPER_DB_DSN']);\
print('status counts:', c.execute(\"select status,count(*) from articles group by 1\").fetchall())"
```

## 3. Recover the fool.com backlog
```bash
# Relabel the historical empty+429 rows as retryable.
python scripts/maintenance/relabel_rate_limited.py --dry-run    # count first
python scripts/maintenance/relabel_rate_limited.py              # apply

# Drain them with adaptive throttling. fool.com defaults to 1 req / 3s
# (built-in override) — safest on a fresh server IP. ~11k rows ≈ several hours.
nohup python run_scrape.py --retry-failed > scrape.log 2>&1 &
tail -f scrape.log     # progress prints every 50 rows
```
Re-run `--retry-failed` until `relabel_rate_limited.py --dry-run` style checks /
the `rate_limited` count reach ~0. It's resumable and idempotent (upsert on
`url`, one row per transaction).

Faster pacing (optional, ~3-4h) — the circuit breaker still backs off on 429:
```bash
export SCRAPER_DOMAIN_OVERRIDES='{"fool.com":{"per_domain":2,"delay":1.5,"max_delay":60}}'
```

## 4. Verify
```bash
python -c "import os,psycopg;c=psycopg.connect(os.environ['SCRAPER_DB_DSN']);\
[print(r) for r in c.execute(\"select status,count(*) from articles group by 1 order by 2 desc\")]"
```
Target: `ok` climbs toward ~20k as fool.com recovers; `rate_limited` trends to 0.

## 5. (Optional) embed + tag the newly recovered rows
Needs `.env` with `MASSIVE_API_KEY` for downstream search/candles; embedding
itself just needs the model download. Both are incremental.
```bash
python scripts/embedding/embed_articles.py     # embeds rows where embedding IS NULL
python scripts/enrichment/tag_segments.py
```

## Safety notes
- Do NOT run `pytest -m db` against `sharedproject`. The db-test fixture is now
  isolated to a `*_test` DB with a guard, but never override
  `SCRAPER_TEST_DB_DSN` to a real database — a `TRUNCATE articles` is what
  destroyed the data originally.
- The DB password appeared in a prior transcript; rotate it when convenient
  (`ALTER ROLE appuser PASSWORD '…';` + update the secret).
- Take a dump before/after big changes: `pg_dump` of `sharedproject`.
