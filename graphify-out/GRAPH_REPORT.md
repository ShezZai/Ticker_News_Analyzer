# Graph Report - .  (2026-06-10)

## Corpus Check
- 59 files · ~145,951 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 520 nodes · 974 edges · 29 communities (26 shown, 3 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 82 edges (avg confidence: 0.6)
- Token cost: 37,349 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Scraper Core Pipeline|Scraper Core Pipeline]]
- [[_COMMUNITY_Insight Extraction|Insight Extraction]]
- [[_COMMUNITY_Backtesting & Sentiment|Backtesting & Sentiment]]
- [[_COMMUNITY_Candles & News Fetching|Candles & News Fetching]]
- [[_COMMUNITY_Insight-Based Search|Insight-Based Search]]
- [[_COMMUNITY_Semantic Article Search|Semantic Article Search]]
- [[_COMMUNITY_Article Classification|Article Classification]]
- [[_COMMUNITY_Daily Range Scanner|Daily Range Scanner]]
- [[_COMMUNITY_Article Embedding|Article Embedding]]
- [[_COMMUNITY_Text Extraction & Overrides|Text Extraction & Overrides]]
- [[_COMMUNITY_Catalyst Returns Analysis|Catalyst Returns Analysis]]
- [[_COMMUNITY_Ticker & Segment Tagging|Ticker & Segment Tagging]]
- [[_COMMUNITY_Pipeline Architecture Concepts|Pipeline Architecture Concepts]]
- [[_COMMUNITY_Ticker Overview Loader|Ticker Overview Loader]]
- [[_COMMUNITY_Ticker Data Loader|Ticker Data Loader]]
- [[_COMMUNITY_Ticker Overview Tests|Ticker Overview Tests]]
- [[_COMMUNITY_Scraper CLI|Scraper CLI]]
- [[_COMMUNITY_Article Store (DB)|Article Store (DB)]]
- [[_COMMUNITY_HTTP Quality Checks|HTTP Quality Checks]]
- [[_COMMUNITY_Article-Day Attachment|Article-Day Attachment]]
- [[_COMMUNITY_Store Tests|Store Tests]]
- [[_COMMUNITY_Render All Tickers|Render All Tickers]]
- [[_COMMUNITY_Universe News Runner|Universe News Runner]]
- [[_COMMUNITY_Venv Setup Script|Venv Setup Script]]

## God Nodes (most connected - your core abstractions)
1. `RawPage` - 26 edges
2. `Settings` - 25 edges
3. `ArticleJob` - 23 edges
4. `DomainLimiter` - 23 edges
5. `Article` - 17 edges
6. `process_job()` - 17 edges
7. `Fetcher` - 16 edges
8. `FakeFetcher` - 15 edges
9. `FakeStore` - 15 edges
10. `RobotsCache` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_extracts_real_article()` --conceptually_related_to--> `Scraper stage (scraper/ package, run_scrape.py)`  [INFERRED]
  tests/test_extractor_real.py → CLAUDE.md
- `Settings` --uses--> `Settings`  [INFERRED]
  scraper/cli.py → scraper/config.py
- `FakeFetcher` --uses--> `Settings`  [INFERRED]
  tests/test_pipeline.py → scraper/config.py
- `FakeRobots` --uses--> `Settings`  [INFERRED]
  tests/test_pipeline.py → scraper/config.py
- `FakeStore` --uses--> `Settings`  [INFERRED]
  tests/test_pipeline.py → scraper/config.py

## Import Cycles
- 1-file cycle: `scraper/csv_source.py -> scraper/csv_source.py`
- 1-file cycle: `scraper/extract/extractor.py -> scraper/extract/extractor.py`
- 1-file cycle: `scraper/extract/overrides/__init__.py -> scraper/extract/overrides/__init__.py`
- 1-file cycle: `scraper/store/db.py -> scraper/store/db.py`
- 1-file cycle: `scripts/checks_backtesting/ticker_candles.py -> scripts/checks_backtesting/ticker_candles.py`
- 1-file cycle: `scripts/ticker_scan/catalyst_returns.py -> scripts/ticker_scan/catalyst_returns.py`
- 1-file cycle: `scripts/ticker_scan/scan_ranges.py -> scripts/ticker_scan/scan_ranges.py`
- 1-file cycle: `scripts/search/search_articles_by_insights.py -> scripts/search/search_articles_by_insights.py`
- 3-file cycle: `scraper/extract/extractor.py -> scraper/extract/overrides/__init__.py -> scraper/extract/overrides/benzinga_com.py -> scraper/extract/extractor.py`

## Hyperedges (group relationships)
- **Pipeline stages reading/writing the shared public.articles table** — claude_md_scraper, claude_md_embedding, claude_md_enrichment, claude_md_search, claude_md_articles_table [EXTRACTED 1.00]
- **Real-publisher HTML fixture suite for extractor regression testing** — fixtures_benzinga_com, fixtures_fool_com, fixtures_globenewswire_com, fixtures_investing_com, fixtures_zacks_com, tests_test_extractor_real_test_extracts_real_article [EXTRACTED 1.00]

## Communities (29 total, 3 thin omitted)

### Community 0 - "Scraper Core Pipeline"
Cohesion: 0.09
Nodes (40): Settings, _parse_dt(), ArticleJob, datetime, read_jobs(), Fetcher, RawPage, Settings (+32 more)

### Community 1 - "Insight Extraction"
Cohesion: 0.06
Nodes (48): annotate_box(), articles_to_process(), _box_dict_to_text(), _clean_quote(), embed_missing(), _embed_texts(), ensure_schema(), extract_all() (+40 more)

### Community 2 - "Backtesting & Sentiment"
Cohesion: 0.10
Nodes (36): InsightHit, InsightHit, fetch_all_prices(), gain_for(), load_candidates(), main(), Judge every ticker the article names: one joint call (>=3 tickers) or one     c, Real-news articles published before 16:00 ET in the date range. (+28 more)

### Community 3 - "Candles & News Fetching"
Cohesion: 0.10
Nodes (33): _api_key(), CandleError, fetch_bars(), locate_candle(), main(), make_chart(), parse_timestamp(), Return the integer row position of the bar covering `ts`.      A bar labelled (+25 more)

### Community 4 - "Insight-Based Search"
Cohesion: 0.10
Nodes (31): datetime, _apply_ann_gucs(), _build_filters(), _cmd_article(), _cmd_text(), _fmt_meta(), InsightGroup, InsightHit (+23 more)

### Community 5 - "Semantic Article Search"
Cohesion: 0.12
Nodes (28): _apply_ann_gucs(), get_article(), _is_date_only(), load_statements(), main(), Semantic search over the embedded articles (text-embedding-3-small + pgvector)., One search result row., Execute the ANN query and map rows to SearchHit objects. (+20 more)

### Community 6 - "Article Classification"
Cohesion: 0.11
Nodes (25): articles_to_process(), classify_all(), classify_one(), ensure_schema(), get_conn(), _is_retryable_server_error(), load_gemini(), main() (+17 more)

### Community 7 - "Daily Range Scanner"
Cohesion: 0.14
Nodes (20): date, datetime, _api_key(), daily_ranges(), DayRange, fetch_bars(), _get(), index_ranges() (+12 more)

### Community 8 - "Article Embedding"
Cohesion: 0.13
Nodes (24): build_text(), create_index(), embed_all(), embed_query(), embed_texts(), ensure_schema(), fetch_rows(), get_conn() (+16 more)

### Community 9 - "Text Extraction & Overrides"
Cohesion: 0.16
Nodes (16): Article, extract(), generic_extract(), _parse_date(), register(), Benzinga.com extractor override.  Benzinga's article pages embed the full arti, datetime, RawPage (+8 more)

### Community 10 - "Catalyst Returns Analysis"
Cohesion: 0.16
Nodes (18): dtime, date, datetime, _api_key(), CatalystError, _fetch(), fetch_prices(), _get() (+10 more)

### Community 11 - "Ticker & Segment Tagging"
Cohesion: 0.14
Nodes (21): build_annotator(), build_matcher(), _build_patterns(), compute_row(), create_indexes(), ensure_schema(), load_ticker_data(), main() (+13 more)

### Community 12 - "Pipeline Architecture Concepts"
Cohesion: 0.12
Nodes (20): public.articles table (Postgres/pgvector), Autocommit one-row-per-transaction store so scrape runs are safe to kill and resume; already-ok URLs skipped unless --retry-errors, Two DB connection conventions: SCRAPER_DB_DSN vs NEWS_DB_DSN must resolve to the same news database, otherwise stages silently hit different databases, Embedding stage (BAAI/bge-m3, vector(1024), HNSW cosine index), Enrich/tag stage (tag_segments.py, ticker_data lookup), HTTP-first fetch with lazy Playwright Chromium fallback: only escalate to browser when http_looks_bad detects bad status, short body, or Cloudflare/JS challenge, Massive.com REST API, pgvector extension (CREATE EXTENSION vector) (+12 more)

### Community 13 - "Ticker Overview Loader"
Cohesion: 0.18
Nodes (17): ensure_schema(), existing_tickers(), fetch_description(), list_tickers(), load(), main(), Create and populate ``public.ticker_overview`` with Yahoo Finance company descr, Create the ticker_overview table if it doesn't already exist. (+9 more)

### Community 14 - "Ticker Data Loader"
Cohesion: 0.25
Nodes (10): ensure_schema(), load(), main(), Create and populate ``public.ticker_data`` from the market-universe CSV.  Read, Create the ticker_data table if it doesn't already exist., Pull (ticker, company_name, primary_ai_segment) from the universe CSV., Create the table and upsert every ticker from the CSV., read_rows() (+2 more)

### Community 15 - "Ticker Overview Tests"
Cohesion: 0.25
Nodes (6): FakeYfTicker, Offline tests for scripts/enrichment/load_ticker_overview.py., test_fetch_description_blank_summary(), test_fetch_description_missing_key(), test_fetch_description_none_info(), test_fetch_description_returns_summary()

### Community 16 - "Scraper CLI"
Cohesion: 0.42
Nodes (7): Namespace, build_settings(), main(), parse_args(), Settings, test_defaults_respect_robots(), test_ignore_robots_flag_overrides_settings()

### Community 18 - "HTTP Quality Checks"
Cohesion: 0.57
Nodes (7): http_looks_bad(), _raw(), test_blocked_statuses_are_bad(), test_challenge_page_is_bad(), test_good_page_is_fine(), test_none_is_bad(), test_tiny_body_is_bad()

### Community 19 - "Article-Day Attachment"
Cohesion: 0.47
Nodes (5): date, fetch_articles_by_ticker_day(), main(), Map (ticker, trading_date) -> [article dicts] for the scan's tickers.      An, read_rows()

### Community 20 - "Store Tests"
Cohesion: 0.70
Nodes (4): _save(), test_error_row_is_not_ok(), test_save_is_idempotent_on_url(), test_save_then_exists_ok()

### Community 21 - "Render All Tickers"
Cohesion: 0.67
Nodes (3): article_tickers(), main(), Map article id -> [primary_ticker, *more_tickers] (deduped, order-preserving).

## Knowledge Gaps
- **13 isolated node(s):** `Exception`, `Exception`, `Exception`, `Connection`, `date` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ZoneInfo` connect `Candles & News Fetching` to `Backtesting & Sentiment`, `Catalyst Returns Analysis`, `Daily Range Scanner`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `embed_query()` connect `Article Embedding` to `Insight-Based Search`, `Semantic Article Search`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `search_insights()` connect `Insight-Based Search` to `Article Embedding`, `Backtesting & Sentiment`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `RawPage` (e.g. with `Fetcher` and `RawPage`) actually correct?**
  _`RawPage` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `Settings` (e.g. with `Namespace` and `Settings`) actually correct?**
  _`Settings` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `ArticleJob` (e.g. with `ArticleJob` and `datetime`) actually correct?**
  _`ArticleJob` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `DomainLimiter` (e.g. with `Settings` and `Fetcher`) actually correct?**
  _`DomainLimiter` has 8 INFERRED edges - model-reasoned connections that need verification._