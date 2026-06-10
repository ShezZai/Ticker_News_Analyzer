from typer.testing import CliRunner

import ticker_news.cli as cli

runner = CliRunner()


def test_help_lists_scrape_command():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "scrape" in result.output


def test_scrape_passes_args_and_settings(monkeypatch):
    captured = {}

    async def fake_run(csv, settings, *, limit=None, retry_errors=False):
        captured.update(csv=csv, settings=settings, limit=limit, retry_errors=retry_errors)

    monkeypatch.setattr("ticker_news.scraping.pipeline.run", fake_run)
    result = runner.invoke(
        cli.app,
        ["scrape", "--csv", "x.csv", "--ignore-robots", "--concurrency", "4",
         "--limit", "10", "--retry-errors"],
    )
    assert result.exit_code == 0, result.output
    assert captured["csv"] == "x.csv"
    assert captured["limit"] == 10
    assert captured["retry_errors"] is True
    assert captured["settings"].respect_robots is False
    assert captured["settings"].concurrency == 4


def test_scrape_defaults_respect_robots(monkeypatch):
    from ticker_news.shared.config import get_settings
    get_settings.cache_clear()
    monkeypatch.delenv("SCRAPER_RESPECT_ROBOTS", raising=False)
    captured = {}

    async def fake_run(csv, settings, *, limit=None, retry_errors=False):
        captured["settings"] = settings

    monkeypatch.setattr("ticker_news.scraping.pipeline.run", fake_run)
    result = runner.invoke(cli.app, ["scrape", "--csv", "x.csv"])
    assert result.exit_code == 0, result.output
    assert captured["settings"].respect_robots is True


def test_scrape_rejects_zero_concurrency(monkeypatch):
    called = {}

    async def fake_run(csv, settings, *, limit=None, retry_errors=False):
        called["yes"] = True

    monkeypatch.setattr("ticker_news.scraping.pipeline.run", fake_run)
    result = runner.invoke(cli.app, ["scrape", "--csv", "x.csv", "--concurrency", "0"])
    assert result.exit_code != 0
    assert "yes" not in called


def test_embed_command_passes_args(monkeypatch):
    captured = {}

    def fake_embed_all(*, batch_size, limit, reembed, build_index):
        captured.update(batch_size=batch_size, limit=limit, reembed=reembed,
                        build_index=build_index)
        return 0

    monkeypatch.setattr("ticker_news.embedding.pipeline.embed_all", fake_embed_all)
    result = runner.invoke(
        cli.app, ["embed", "--batch-size", "64", "--limit", "5", "--reembed", "--no-index"]
    )
    assert result.exit_code == 0, result.output
    assert captured == {"batch_size": 64, "limit": 5, "reembed": True, "build_index": False}


def test_classify_command_passes_args(monkeypatch):
    captured = {}

    def fake_classify_all(*, reprocess, limit, ids, workers):
        captured.update(reprocess=reprocess, limit=limit, ids=ids, workers=workers)
        return 0

    monkeypatch.setattr(
        "ticker_news.classification.pipeline.classify_all", fake_classify_all
    )
    result = runner.invoke(
        cli.app, ["classify", "--limit", "20", "--ids", "78,79,80", "--workers", "4"]
    )
    assert result.exit_code == 0, result.output
    assert captured == {"reprocess": False, "limit": 20, "ids": [78, 79, 80], "workers": 4}


def test_tag_command(monkeypatch):
    captured = {}

    def fake_tag_all(*, only_missing, build_index):
        captured.update(only_missing=only_missing, build_index=build_index)
        return 0

    monkeypatch.setattr("ticker_news.enrichment.tagging.tag_all", fake_tag_all)
    result = runner.invoke(cli.app, ["tag", "--not-only-missing", "--no-index"])
    assert result.exit_code == 0, result.output
    assert captured == {"only_missing": False, "build_index": False}


def test_tag_command_defaults(monkeypatch):
    captured = {}

    def fake_tag_all(*, only_missing, build_index):
        captured.update(only_missing=only_missing, build_index=build_index)
        return 0

    monkeypatch.setattr("ticker_news.enrichment.tagging.tag_all", fake_tag_all)
    result = runner.invoke(cli.app, ["tag"])
    assert result.exit_code == 0, result.output
    assert captured == {"only_missing": True, "build_index": True}


def test_load_universe_command(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "ticker_news.enrichment.reference_data.load_universe",
        lambda csv_path: captured.update(csv=str(csv_path)) or 7,
    )
    result = runner.invoke(cli.app, ["load-universe", "--csv", "u.csv"])
    assert result.exit_code == 0, result.output
    assert captured["csv"].endswith("u.csv")


def test_load_overviews_command(monkeypatch):
    captured = {}

    def fake_load(*, tickers, refresh, delay):
        captured.update(tickers=tickers, refresh=refresh, delay=delay)

    monkeypatch.setattr("ticker_news.enrichment.reference_data.load_overviews", fake_load)
    result = runner.invoke(cli.app, ["load-overviews", "--tickers", "nvda,AMD", "--refresh"])
    assert result.exit_code == 0, result.output
    assert captured == {"tickers": ["NVDA", "AMD"], "refresh": True, "delay": 0.5}


def test_insights_command_passes_args(monkeypatch):
    captured = {}

    def fake_extract_all(*, reprocess, limit, quote_threshold, ids, workers):
        captured.update(reprocess=reprocess, limit=limit,
                        quote_threshold=quote_threshold, ids=ids, workers=workers)
        return 0

    monkeypatch.setattr("ticker_news.enrichment.insights.extract_all", fake_extract_all)
    monkeypatch.setattr("ticker_news.enrichment.insights.embed_missing", lambda **kw: 0)
    result = runner.invoke(cli.app, ["insights", "--limit", "3", "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert captured["limit"] == 3 and captured["workers"] == 2


def test_insights_fix_quotes_early_returns(monkeypatch):
    called = {}
    monkeypatch.setattr(
        "ticker_news.enrichment.insights.fix_quotes",
        lambda *, quote_threshold: called.setdefault("fix", quote_threshold),
    )
    monkeypatch.setattr(
        "ticker_news.enrichment.insights.extract_all",
        lambda **kw: called.setdefault("extract", True),
    )
    monkeypatch.setattr(
        "ticker_news.enrichment.insights.embed_missing",
        lambda **kw: called.setdefault("embed", True),
    )
    result = runner.invoke(cli.app, ["insights", "--fix-quotes", "--quote-threshold", "0.8"])
    assert result.exit_code == 0, result.output
    assert called == {"fix": 0.8}


def test_insights_embed_only_skips_extraction(monkeypatch):
    called = {}
    monkeypatch.setattr(
        "ticker_news.enrichment.insights.extract_all",
        lambda **kw: called.setdefault("extract", True),
    )
    monkeypatch.setattr(
        "ticker_news.enrichment.insights.embed_missing",
        lambda **kw: called.setdefault("embed", True),
    )
    result = runner.invoke(cli.app, ["insights", "--embed-only"])
    assert result.exit_code == 0, result.output
    assert called == {"embed": True}


def test_insights_no_embed_skips_embedding(monkeypatch):
    called = {}
    monkeypatch.setattr(
        "ticker_news.enrichment.insights.extract_all",
        lambda **kw: called.setdefault("extract", True),
    )
    monkeypatch.setattr(
        "ticker_news.enrichment.insights.embed_missing",
        lambda **kw: called.setdefault("embed", True),
    )
    result = runner.invoke(cli.app, ["insights", "--no-embed"])
    assert result.exit_code == 0, result.output
    assert called == {"extract": True}


def test_backfill_command_enqueues_csv(monkeypatch):
    captured = {}

    class FakeSource:
        def __init__(self, csv_path, limit=None):
            captured["csv"] = csv_path
            captured["limit"] = limit

    async def fake_serve(source, *, workers, poll_interval_s, drain):
        captured["drain"] = drain
        captured["workers"] = workers
        return {"done": 0, "failed": 0}

    monkeypatch.setattr("ticker_news.ingestion.csv_backfill.CsvBackfillSource", FakeSource)
    monkeypatch.setattr("ticker_news.service.worker.serve", fake_serve)
    result = runner.invoke(cli.app, ["backfill", "--csv", "x.csv", "--workers", "2"])
    assert result.exit_code == 0, result.output
    assert captured == {"csv": "x.csv", "limit": None, "drain": True, "workers": 2}


def test_serve_command_uses_massive_source(monkeypatch):
    captured = {}

    class FakeSource:
        def __init__(self, tickers, *, poll_interval_s, lookback, **kw):
            captured["tickers"] = list(tickers)
            captured["poll"] = poll_interval_s

    async def fake_serve(source, *, workers, poll_interval_s, drain):
        captured["drain"] = drain
        return {"done": 0, "failed": 0}

    monkeypatch.setattr("ticker_news.ingestion.massive_rest.MassiveRestSource", FakeSource)
    monkeypatch.setattr("ticker_news.service.worker.serve", fake_serve)
    monkeypatch.setattr(
        "ticker_news.cli._universe_tickers", lambda: ["NVDA", "AMD"]
    )
    result = runner.invoke(cli.app, ["serve", "--poll-interval", "30"])
    assert result.exit_code == 0, result.output
    assert captured["tickers"] == ["NVDA", "AMD"]
    assert captured["poll"] == 30.0
    assert captured["drain"] is False


def test_jobs_status_command(monkeypatch):
    monkeypatch.setattr("ticker_news.service.jobs.ensure_schema", lambda conn: None)
    monkeypatch.setattr("ticker_news.service.jobs.counts", lambda conn: {"pending": 2, "done": 5})
    monkeypatch.setattr("ticker_news.shared.db.connect", lambda **kw: _FakeConn())
    result = runner.invoke(cli.app, ["jobs", "status"])
    assert result.exit_code == 0, result.output
    assert "pending" in result.output and "2" in result.output


class _FakeConn:
    def close(self):
        pass
