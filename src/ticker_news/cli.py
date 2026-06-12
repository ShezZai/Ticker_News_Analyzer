import asyncio
from dataclasses import replace
from pathlib import Path

import typer

app = typer.Typer(help="Ticker News Analyzer pipeline.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Ticker News Analyzer — collect, scrape, embed, classify, and judge stock news."""
    # Forces typer to treat this as a multi-command app even while `scrape`
    # is the only command, so the interface stays `ticker-news scrape ...`.


@app.command()
def scrape(
    csv: str = typer.Option(..., help="Path to the articles CSV."),
    limit: int | None = typer.Option(None, help="Process at most N rows."),
    retry_errors: bool = typer.Option(False, "--retry-errors", help="Re-process URLs already stored with an error."),
    ignore_robots: bool = typer.Option(False, "--ignore-robots", help="Skip robots.txt checks."),
    concurrency: int | None = typer.Option(None, min=1, help="Worker count override."),
) -> None:
    """Scrape article bodies for every URL in the CSV into the articles table."""
    from ticker_news.scraping import pipeline
    from ticker_news.scraping.config import Settings

    settings = Settings()
    if ignore_robots:
        settings = replace(settings, respect_robots=False)
    if concurrency is not None:
        settings = replace(settings, concurrency=concurrency)
    asyncio.run(pipeline.run(csv, settings, limit=limit, retry_errors=retry_errors))


@app.command()
def embed(
    batch_size: int = typer.Option(256, min=1, help="Rows per DB batch and inputs per embeddings API request."),
    limit: int | None = typer.Option(None, help="Only process the first N pending rows."),
    reembed: bool = typer.Option(False, "--reembed", help="Recompute embeddings for every row."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip building the HNSW index."),
) -> None:
    """Embed articles missing an embedding into pgvector (resumable)."""
    from ticker_news.embedding import pipeline

    pipeline.embed_all(
        batch_size=batch_size, limit=limit, reembed=reembed, build_index=not no_index
    )


@app.command()
def classify(
    limit: int | None = typer.Option(None, help="Only process the first N pending articles."),
    reprocess: bool = typer.Option(False, "--reprocess", help="Re-classify articles that already have a category."),
    ids: str | None = typer.Option(None, help="Comma-separated article ids to (re)classify."),
    workers: int = typer.Option(8, min=1, help="Concurrent Gemini requests."),
) -> None:
    """Classify articles into content categories (two-pass Gemini, resumable)."""
    from ticker_news.classification import pipeline

    id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else None
    pipeline.classify_all(reprocess=reprocess, limit=limit, ids=id_list, workers=workers)


@app.command()
def tag(
    not_only_missing: bool = typer.Option(False, "--not-only-missing", help="Recompute every row, not just untagged ones."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip building indexes."),
) -> None:
    """Tag articles with primary/secondary tickers and segments."""
    from ticker_news.enrichment import tagging

    tagging.tag_all(only_missing=not not_only_missing, build_index=not no_index)


@app.command(name="load-universe")
def load_universe(
    csv: Path = typer.Option(None, help="Universe CSV (default: repo-root consolidated CSV)."),
) -> None:
    """Load the ticker -> company/segment universe into ticker_data."""
    from ticker_news.enrichment import reference_data

    n = reference_data.load_universe(csv_path=csv or reference_data.DEFAULT_CSV)
    typer.echo(f"Upserted {n} tickers.")


@app.command(name="load-overviews")
def load_overviews(
    tickers: str | None = typer.Option(None, help="Comma-separated tickers (default: all in ticker_data)."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-fetch tickers that already have a row."),
    delay: float = typer.Option(0.5, help="Seconds between Yahoo requests."),
) -> None:
    """Fetch Yahoo Finance company descriptions into ticker_overview."""
    from ticker_news.enrichment import reference_data

    t = [x.strip().upper() for x in tickers.split(",") if x.strip()] if tickers else None
    reference_data.load_overviews(tickers=t, refresh=refresh, delay=delay)


@app.command()
def insights(
    limit: int | None = typer.Option(None, help="Only process the first N pending articles."),
    reprocess: bool = typer.Option(False, "--reprocess", help="Re-extract articles that already have insights."),
    ids: str | None = typer.Option(None, help="Comma-separated article ids (implies reprocess for those)."),
    workers: int = typer.Option(8, min=1, help="Concurrent Gemini requests."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Extract boxes only; skip embedding."),
    embed_only: bool = typer.Option(False, "--embed-only", help="Skip extraction; only fill missing embeddings."),
    fix_quotes_flag: bool = typer.Option(False, "--fix-quotes", help="Re-verbatimize existing rows (no LLM)."),
    quote_threshold: float = typer.Option(0.75, help="Min similarity for fuzzy quote match."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip building the HNSW index."),
) -> None:
    """Extract embedded insight boxes from articles (resumable)."""
    from ticker_news.enrichment import insights as mod

    if fix_quotes_flag:
        mod.fix_quotes(quote_threshold=quote_threshold)
        return
    id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else None
    if not embed_only:
        mod.extract_all(reprocess=reprocess, limit=limit,
                        quote_threshold=quote_threshold, ids=id_list, workers=workers)
    if embed_only or not no_embed:
        mod.embed_missing(build_index=not no_index)


@app.command()
def sentiment(
    limit: int | None = typer.Option(None, help="Judge at most N pending articles."),
    reprocess: bool = typer.Option(False, "--reprocess", help="Re-judge articles that already have a verdict."),
) -> None:
    """Run the analyst-panel sentiment over real-news articles missing a verdict."""
    from ticker_news.sentiment import batch

    n = batch.run_batch(limit=limit, reprocess=reprocess)
    typer.echo(f"judged {n} article(s)")


@app.command()
def search(
    query: str | None = typer.Argument(None, help="Free-text search query."),
    like: int | None = typer.Option(None, "--like", metavar="ID", help="Find articles similar to this article id instead of a text query."),
    like_url: str | None = typer.Option(None, "--like-url", metavar="URL", help="Find articles similar to the article at this URL."),
    statement: str | None = typer.Option(None, "--statement", metavar="TEXT|FILE", help="Search by similarity to a statement, dated relative to the --like/--like-url article. A literal string, or a path to a JSON array of {statement, scope, value} objects."),
    k: int = typer.Option(10, "-k", "--k", help="Number of results."),
    ticker: list[str] | None = typer.Option(None, "--ticker", metavar="SYM", help="Filter by ticker (repeatable)."),
    segment: str | None = typer.Option(None, "--segment", metavar="NAME", help="Filter by AI segment (matches primary_segment or more_segments)."),
    domain: str | None = typer.Option(None, "--domain", help="Filter by source_domain."),
    since: str | None = typer.Option(None, "--since", "--after", help="Earliest published_utc, inclusive (YYYY-MM-DD)."),
    until: str | None = typer.Option(None, "--until", "--before", help="Latest published_utc, inclusive (YYYY-MM-DD)."),
    months_before: int | None = typer.Option(None, "--months-before", metavar="N", help="With --like/--like-url, only search the N months before the seed article's own publish date."),
    min_similarity: float = typer.Option(0.7, "--min-similarity", help="Drop results below this cosine similarity (pass 0 to show all)."),
    exclusive: bool = typer.Option(False, "--exclusive", help="Make the upper date bound strict on the exact timestamp; with --months-before this anchors the window on the seed's exact publish time."),
    ef_search: int | None = typer.Option(None, "--ef-search", metavar="N", help="HNSW ANN candidate breadth (default 40, or HNSW_EF_SEARCH); raise it if a selective filter returns too few/no results."),
) -> None:
    """Semantic search over the embedded articles (pgvector ANN)."""
    from ticker_news.research import search as search_mod

    try:
        search_mod.run_cli(
            query,
            like=like,
            like_url=like_url,
            statement=statement,
            k=k,
            tickers=list(ticker) if ticker else None,
            segment=segment,
            domain=domain,
            since=since,
            until=until,
            months_before=months_before,
            min_similarity=min_similarity,
            exclusive=exclusive,
            ef_search=ef_search,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command(name="search-insights")
def search_insights(
    query: str | None = typer.Argument(None, help="Free-text insight search query."),
    like: int | None = typer.Option(None, "--like", metavar="ID", help="Search using the insights of this article id."),
    like_url: str | None = typer.Option(None, "--like-url", metavar="URL", help="Search using the insights of the article at this URL."),
    k: int = typer.Option(5, "-k", "--k", help="Top-k similar insights retrieved per seed insight (also total results in text mode)."),
    per_insight: bool = typer.Option(False, "--per-insight", help="With --like/--like-url, show the per-insight breakdown instead of the consolidated article list."),
    top_articles: int | None = typer.Option(None, "--top-articles", metavar="N", help="Cap how many consolidated articles to show (default: all)."),
    ticker: list[str] | None = typer.Option(None, "--ticker", metavar="SYM", help="Filter matches by ticker (repeatable)."),
    segment: str | None = typer.Option(None, "--segment", metavar="NAME", help="Filter matches by AI segment (primary or more_segments)."),
    domain: str | None = typer.Option(None, "--domain", help="Filter matches by source_domain."),
    since: str | None = typer.Option(None, "--since", "--after", help="Earliest published_utc of the matched article (YYYY-MM-DD)."),
    until: str | None = typer.Option(None, "--until", "--before", help="Latest published_utc of the matched article (YYYY-MM-DD)."),
    months_before: int | None = typer.Option(None, "--months-before", metavar="N", help="With --like/--like-url, only match insights from the N months before the seed article's own publish date."),
    min_similarity: float = typer.Option(0.7, "--min-similarity", help="Drop matches below this cosine similarity (pass 0 to show all)."),
    exclusive: bool = typer.Option(False, "--exclusive", help="Make the upper date bound strict on the exact timestamp; with --months-before this anchors the window on the seed's exact publish time."),
    ef_search: int | None = typer.Option(None, "--ef-search", metavar="N", help="HNSW ANN candidate breadth (default 40, or HNSW_EF_SEARCH); raise it if a selective filter returns too few/no results."),
) -> None:
    """Insight-level semantic search over article_insights (pgvector ANN)."""
    from ticker_news.research import insight_search as insight_mod

    try:
        insight_mod.run_cli(
            query,
            like=like,
            like_url=like_url,
            k=k,
            per_insight=per_insight,
            top_articles=top_articles,
            tickers=list(ticker) if ticker else None,
            segment=segment,
            domain=domain,
            since=since,
            until=until,
            months_before=months_before,
            min_similarity=min_similarity,
            exclusive=exclusive,
            ef_search=ef_search,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def serve(
    workers: int = typer.Option(4, min=1, help="Concurrent pipeline workers."),
    poll_interval: float = typer.Option(60.0, help="Feed poll interval, seconds."),
    lookback_hours: float = typer.Option(24.0, help="How far back the first poll reaches."),
    tickers: str | None = typer.Option(None, help="Comma-separated tickers (default: ticker_data table)."),
) -> None:
    """Run the live pipeline: poll the news feed, process articles end to end."""
    import asyncio
    from datetime import timedelta

    from ticker_news.ingestion.massive_rest import MassiveRestSource
    from ticker_news.service.worker import serve as run_service

    universe = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers else _universe_tickers()
    )
    source = MassiveRestSource(
        universe, poll_interval_s=poll_interval, lookback=timedelta(hours=lookback_hours)
    )
    try:
        asyncio.run(run_service(source, workers=workers,
                                poll_interval_s=5.0, drain=False))
    except KeyboardInterrupt:
        typer.echo("stopped.")


def _universe_tickers() -> list[str]:
    from ticker_news.shared import db

    conn = db.connect()
    try:
        rows = conn.execute("SELECT ticker FROM public.ticker_data ORDER BY ticker").fetchall()
    finally:
        conn.close()
    if not rows:
        raise SystemExit(
            "ticker_data is empty - run `ticker-news load-universe` first or pass --tickers."
        )
    return [r[0] for r in rows]


@app.command()
def backfill(
    csv: str = typer.Option(..., help="News CSV to enqueue and process."),
    limit: int | None = typer.Option(None, help="Enqueue at most N rows."),
    workers: int = typer.Option(4, min=1, help="Concurrent pipeline workers."),
) -> None:
    """Enqueue a news CSV and process it to completion (drain mode)."""
    import asyncio

    from ticker_news.ingestion.csv_backfill import CsvBackfillSource
    from ticker_news.service.worker import serve as run_service

    source = CsvBackfillSource(csv, limit=limit)
    try:
        counts = asyncio.run(run_service(source, workers=workers,
                                         poll_interval_s=1.0, drain=True))
    except KeyboardInterrupt:
        typer.echo("stopped.")
        return
    typer.echo(f"backfill complete: {counts}")


@app.command(name="fetch-news")
def fetch_news(
    tickers: str | None = typer.Option(None, help="Comma-separated tickers (default: ticker_data table)."),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: today)."),
    output: Path = typer.Option(None, "--output", "-o", help="Output CSV (default: news_<start>_<end>.csv)."),
) -> None:
    """Fetch news metadata + provider sentiment from Massive into a CSV."""
    from datetime import date

    from ticker_news.ingestion import news_history

    t = [x.strip().upper() for x in tickers.split(",") if x.strip()] if tickers else _universe_tickers()
    end_date = end or date.today().isoformat()
    out = str(output) if output else f"news_{start}_{end_date}.csv"
    path = news_history.fetch_news_csv(t, start, end_date, output_path=out)
    typer.echo(f"wrote {path}")


research_app = typer.Typer(help="On-demand research & backtesting tools.")
app.add_typer(research_app, name="research")


@research_app.command("chart")
def research_chart(
    ticker: str = typer.Argument(..., help="Ticker symbol."),
    timestamp: str = typer.Argument(..., help="Timestamp to mark (ISO, ET assumed if naive)."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output JPG path (default TICKER_DATE_HHMM.jpg)."),
    interval: int = typer.Option(1, help="Candle size in minutes."),
    tz: str = typer.Option("America/New_York", "--tz", help="Timezone for naive timestamps."),
    api_key: str | None = typer.Option(None, "--api-key", help="Override MASSIVE_API_KEY."),
) -> None:
    """Render an intraday candlestick chart with the timestamp marked."""
    from ticker_news.research.candles import make_chart

    try:
        typer.echo(make_chart(ticker, timestamp, out=output, interval=interval, tz=tz, key=api_key))
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


@research_app.command("scan-ranges")
def research_scan_ranges(
    threshold: float = typer.Option(5.0, "--threshold", "-t", help="Min low-to-high range in % to flag a day."),
    tickers: str | None = typer.Option(None, help="Comma-separated tickers (default: all from ticker_data)."),
    start: str = typer.Option("2024-11-01", help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: today)."),
    bar: str = typer.Option("hour", help="Intraday bar size: 'hour' (exact for daily extremes, cheap) or 'minute'."),
    index_ticker: str = typer.Option("I:COMP", "--index-ticker", help="Reference index symbol (default NASDAQ Composite)."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output CSV (default ticker_range_scan_<start>_<end>.csv)."),
    workers: int = typer.Option(8, min=1, help="Concurrent ticker fetches."),
    nas_t: float = typer.Option(1.2, "--nas-t", help="Max NASDAQ low-to-high % on a flagged day; only keeps days where NASDAQ moved LESS than this (set 999 to disable)."),
    api_key: str | None = typer.Option(None, "--api-key", help="Override MASSIVE_API_KEY."),
) -> None:
    """Scan tickers for big intraday-range days (calm-NASDAQ days only) into a CSV."""
    from datetime import date

    from ticker_news.research import ticker_scan as ts

    if bar not in ("hour", "minute"):
        raise typer.BadParameter("--bar must be 'hour' or 'minute'")
    end_date = end or date.today().isoformat()
    out = output or f"ticker_range_scan_{start}_{end_date}.csv"
    try:
        segment_map: dict[str, str] = {}
        if tickers:
            tlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            try:  # best-effort segment lookup even for explicit ticker lists
                _, segment_map = ts.load_universe()
            except ts.ScanError:
                pass
        else:
            tlist, segment_map = ts.load_universe()
        rows = ts.scan(
            tlist, start=start, end=end_date, threshold=threshold, bar=bar,
            index_ticker=index_ticker, nas_threshold=nas_t, workers=workers,
            key=api_key, segment_map=segment_map,
        )
        ts.write_scan_csv(rows, out)
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"Wrote {len(rows)} flagged day(s) across "
        f"{len({r['ticker'] for r in rows})} ticker(s) to {out}"
    )


@research_app.command("attach-articles")
def research_attach_articles(
    input_csv: str = typer.Argument(..., help="ticker_range_scan CSV to read."),
    output_csv: str = typer.Argument(..., help="Destination CSV (input + articles columns)."),
) -> None:
    """Attach each scan row's same-day and prev-after-hours articles from the DB."""
    from ticker_news.research import ticker_scan as ts

    try:
        ts.attach(input_csv, output_csv)
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


@research_app.command("catalyst-returns")
def research_catalyst_returns(
    start: str = typer.Option("2025-02-01", help="Start date YYYY-MM-DD."),
    end: str = typer.Option("2025-03-30", help="End date YYYY-MM-DD."),
    categories: str = typer.Option(
        "real news,legal solicitation,regulatory filing",
        help="Comma-separated catalyst categories.",
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output CSV (default catalyst_returns_<start>_<end>.csv)."),
    workers: int = typer.Option(8, min=1, help="Concurrent price fetches."),
    include_after_hours: bool = typer.Option(False, "--include-after-hours", help="Also keep extended-hours entries (sold at the next day's close); excluded by default."),
    all_tickers: bool = typer.Option(False, "--all-tickers", help="Emit a row for EVERY ticker each article names (primary + more_tickers); adds a ticker_role column."),
    api_key: str | None = typer.Option(None, "--api-key", help="Override MASSIVE_API_KEY."),
) -> None:
    """Buy-the-news returns per catalyst article: entry at publication, exit next regular close."""
    from ticker_news.research import ticker_scan as ts

    cats = [c.strip() for c in categories.split(",") if c.strip()]
    out = output or f"catalyst_returns_{start}_{end}.csv"
    try:
        rows = ts.catalyst_run(
            start=start, end=end, categories=cats, workers=workers,
            include_after_hours=include_after_hours, all_tickers=all_tickers,
            key=api_key,
        )
        ts.write_catalyst_csv(rows, out)
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Wrote {len(rows)} row(s) to {out}")
    for line in ts.catalyst_summary(rows):
        typer.echo(line)


@research_app.command("backtest")
def research_backtest(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD (article ET publication date)."),
    end: str = typer.Option(..., help="End date YYYY-MM-DD, inclusive."),
    include_hold: bool = typer.Option(False, "--include-hold", help="Also simulate hold verdicts (tracked unsigned; never scored)."),
    out: str | None = typer.Option(None, "--out", "-o", help="Output CSV (default backtest_verdicts_<start>_<end>.csv)."),
    workers: int = typer.Option(8, min=1, help="Concurrent price fetches."),
    api_key: str | None = typer.Option(None, "--api-key", help="Override MASSIVE_API_KEY."),
) -> None:
    """Backtest analyst-panel verdicts: entry at publication, exit next regular close."""
    from ticker_news.research import backtest as bt

    try:
        bt.run_backtest(
            start=start, end=end, include_hold=include_hold, out=out,
            workers=workers, key=api_key,
        )
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


@research_app.command("render-bombs")
def research_render_bombs(
    input_csv: str = typer.Argument(..., help="*_articled.csv from attach-articles."),
    out_dir: str = typer.Option("pics_bombs", "--out-dir", help="Output folder."),
) -> None:
    """Render a marked intraday chart for every article attached in an articled CSV."""
    from ticker_news.research import render as render_mod

    try:
        render_mod.render_bombs(input_csv, out_dir=out_dir)
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


@research_app.command("render-catalysts")
def research_render_catalysts(
    input_csv: str = typer.Argument(
        "catalyst_returns_2025-02-01_2025-03-30.csv",
        help="catalyst_returns CSV to read.",
    ),
    threshold: float = typer.Option(3.0, "--threshold", "-t", help="Min |gain_pct| to render, up or down."),
    out_dir: str = typer.Option("pics_bombs/catalysts", "--out-dir", help="Output folder."),
) -> None:
    """Render entry-point charts for the big movers in a catalyst_returns CSV."""
    from ticker_news.research import render as render_mod

    try:
        render_mod.render_catalysts(input_csv, threshold=threshold, out_dir=out_dir)
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


@research_app.command("render-all")
def research_render_all(
    csv_path: str = typer.Option(
        "catalyst_returns_2025-02-01_2025-03-30.csv", "--csv",
        help="catalyst_returns CSV the pics came from.",
    ),
    pics_dir: str = typer.Option(
        "pics_bombs/catalysts/other_related", "--pics-dir",
        help="Folder of <ticker>_<article_id>_<date>.jpg charts.",
    ),
    out_dir: str | None = typer.Option(None, "--out-dir", help="Output folder (default: --pics-dir)."),
) -> None:
    """Render the missing charts for every OTHER ticker each pictured article names."""
    from ticker_news.research import render as render_mod

    try:
        render_mod.render_all_tickers(csv_path, pics_dir, out_dir=out_dir)
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


prompts_app = typer.Typer(help="Manage Langfuse prompt versions.")
app.add_typer(prompts_app, name="prompts")


@prompts_app.command("push")
def prompts_push() -> None:
    """Upsert the in-repo prompts to Langfuse with the production label."""
    from ticker_news.shared import prompts as prompts_mod

    try:
        n = prompts_mod.push_all()
    except SystemExit as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"pushed {n} prompt(s)")


jobs_app = typer.Typer(help="Inspect and manage the pipeline job queue.")
app.add_typer(jobs_app, name="jobs")


@jobs_app.command("status")
def jobs_status() -> None:
    """Show queue counts by status."""
    from ticker_news.service import jobs as jobs_mod
    from ticker_news.shared import db

    conn = db.connect()
    try:
        jobs_mod.ensure_schema(conn)
        for status, n in sorted(jobs_mod.counts(conn).items()):
            typer.echo(f"{status:>8}  {n}")
    finally:
        conn.close()


@jobs_app.command("retry")
def jobs_retry(
    url: str | None = typer.Option(None, help="Requeue one failed URL (default: all failed)."),
) -> None:
    """Requeue failed jobs."""
    from ticker_news.service import jobs as jobs_mod
    from ticker_news.shared import db

    conn = db.connect()
    try:
        jobs_mod.ensure_schema(conn)
        n = jobs_mod.requeue_failed(conn, url)
        typer.echo(f"requeued {n} job(s).")
    finally:
        conn.close()


eval_app = typer.Typer(help="Pipeline quality evals (Langfuse experiments).")
app.add_typer(eval_app, name="eval")


def _echo_summary(summary: str) -> None:
    try:
        typer.echo(summary)
    except UnicodeEncodeError:  # Windows cp1252 console vs emoji in format()
        typer.echo(summary.encode("ascii", "backslashreplace").decode("ascii"))


@eval_app.command("pipeline")
def eval_pipeline(
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated article ids to force through the full eval pipeline."),
    ids_file: str | None = typer.Option(None, "--ids-file", help="File with one article id per line (unioned with --ids)."),
    dataset: str | None = typer.Option(None, "--dataset", help="Langfuse dataset name: upsert --ids as items, then run over the whole dataset."),
    dsn: str | None = typer.Option(None, "--dsn", help="Target DB DSN (default: DATABASE_URL)."),
    run_name: str | None = typer.Option(None, "--run-name", help="Experiment run name (default: auto-generated)."),
    skip_stages: str | None = typer.Option(None, "--skip-stages", help="Comma-separated stages whose stored outputs are reused instead of re-run: embed, classify, tag, insights."),
) -> None:
    """Re-run articles E2E through the pipeline; score verdicts against actual price moves."""
    from ticker_news.evals import pipeline_eval

    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else []
    except ValueError as exc:
        raise typer.BadParameter(f"--ids must be comma-separated integers: {exc}")
    if ids_file:
        try:
            id_list = sorted(set(id_list) | set(pipeline_eval.parse_ids_file(ids_file)))
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"--ids-file: {exc}")
    try:
        skip = pipeline_eval.parse_skip_stages(skip_stages)
    except ValueError as exc:
        raise typer.BadParameter(f"--skip-stages: {exc}")
    if not id_list and not dataset:
        raise typer.BadParameter("provide --ids/--ids-file, or --dataset with existing items")
    try:
        result = pipeline_eval.run_eval(
            id_list, dataset_name=dataset, dsn=dsn, run_name=run_name,
            skip_stages=skip,
        )
    except SystemExit as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    _echo_summary(result.format())


@eval_app.command("classify")
def eval_classify(
    variant: str = typer.Option("both", "--variant", help="both | binary | finegrained."),
    model: str = typer.Option("lite", "--model", help="lite (gemini-2.5-flash-lite) | flash (gemini-2.5-flash)."),
    dataset: str | None = typer.Option(None, "--dataset", help="Dataset override (single --variant only; default: the variant's own dataset)."),
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated article ids: run only this dataset subset (fast prompt iteration)."),
    dsn: str | None = typer.Option(None, "--dsn", help="Target DB DSN (default: DATABASE_URL)."),
    run_name: str | None = typer.Option(None, "--run-name", help="Experiment run name (default: <variant>-<model>-<timestamp>)."),
    concurrency: int = typer.Option(16, "--concurrency", help="Max concurrent dataset items."),
) -> None:
    """Single-pass classifier experiments: binary vs ACT labels, finegrained vs categories."""
    from ticker_news.classification.variants import MODEL_CHOICES, VARIANTS
    from ticker_news.evals import classify_eval

    if variant not in ("both", *VARIANTS):
        raise typer.BadParameter(f"--variant must be one of: both, {', '.join(VARIANTS)}")
    if model not in MODEL_CHOICES:
        raise typer.BadParameter(f"--model must be one of: {', '.join(MODEL_CHOICES)}")
    try:
        id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else None
    except ValueError as exc:
        raise typer.BadParameter(f"--ids must be comma-separated integers: {exc}")
    if dataset and variant == "both":
        raise typer.BadParameter("--dataset requires a single --variant")
    variants = VARIANTS if variant == "both" else (variant,)
    try:
        results = classify_eval.run_eval(
            variants, model=model, dataset_name=dataset,
            dsn=dsn, run_name=run_name, ids=id_list, concurrency=concurrency,
        )
    except SystemExit as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    for variant_name, result in results:
        typer.echo(f"=== {variant_name} ===")
        _echo_summary(result.format())
