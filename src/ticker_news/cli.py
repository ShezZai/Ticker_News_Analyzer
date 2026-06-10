import asyncio
from dataclasses import replace

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
    batch_size: int = typer.Option(256, min=1, help="Inputs per embeddings API request."),
    limit: int | None = typer.Option(None, help="Only process the first N pending rows."),
    reembed: bool = typer.Option(False, "--reembed", help="Recompute embeddings for every row."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip building the HNSW index."),
) -> None:
    """Embed articles missing an embedding into pgvector (resumable)."""
    from ticker_news.embedding import pipeline

    pipeline.embed_all(
        batch_size=batch_size, limit=limit, reembed=reembed, build_index=not no_index
    )
