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
