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


def test_sentiment_command(monkeypatch):
    captured = {}

    def fake_run(*, limit, reprocess):
        captured.update(limit=limit, reprocess=reprocess)
        return 3

    monkeypatch.setattr("ticker_news.sentiment.batch.run_batch", fake_run)
    result = runner.invoke(cli.app, ["sentiment", "--limit", "10"])
    assert result.exit_code == 0, result.output
    assert captured == {"limit": 10, "reprocess": False}


def test_prompts_push_command(monkeypatch):
    monkeypatch.setattr("ticker_news.shared.prompts.push_all", lambda: 6)
    result = runner.invoke(cli.app, ["prompts", "push"])
    assert result.exit_code == 0, result.output
    assert "6" in result.output


def test_research_chart_wraps_runtime_errors(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("MASSIVE_API_KEY is not set")

    monkeypatch.setattr("ticker_news.research.candles.make_chart", boom)
    result = runner.invoke(cli.app, ["research", "chart", "NVDA", "2025-01-07 10:30"])
    assert result.exit_code == 1
    assert "MASSIVE_API_KEY" in result.output
    assert "Traceback" not in result.output


def test_research_backtest_passes_args(monkeypatch):
    captured = {}

    def fake_run(*, start, end, include_hold, out, workers, key):
        captured.update(start=start, end=end, include_hold=include_hold,
                        out=out, workers=workers, key=key)
        return {"total": 0}

    monkeypatch.setattr("ticker_news.research.backtest.run_backtest", fake_run)
    result = runner.invoke(
        cli.app,
        ["research", "backtest", "--start", "2025-02-01", "--end", "2025-03-30",
         "--include-hold", "--workers", "4", "--out", "v.csv"],
    )
    assert result.exit_code == 0, result.output
    assert captured == {
        "start": "2025-02-01", "end": "2025-03-30", "include_hold": True,
        "out": "v.csv", "workers": 4, "key": None,
    }


def test_research_backtest_wraps_runtime_errors(monkeypatch):
    def boom(**kw):
        raise RuntimeError("MASSIVE_API_KEY is not set")

    monkeypatch.setattr("ticker_news.research.backtest.run_backtest", boom)
    result = runner.invoke(
        cli.app, ["research", "backtest", "--start", "2025-02-01", "--end", "2025-03-30"]
    )
    assert result.exit_code == 1
    assert "MASSIVE_API_KEY" in result.output
    assert "Traceback" not in result.output


def test_research_render_bombs_command(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "ticker_news.research.render.render_bombs",
        lambda input_csv, *, out_dir: captured.update(csv=input_csv, out=out_dir) or 2,
    )
    result = runner.invoke(
        cli.app, ["research", "render-bombs", "scan_articled.csv", "--out-dir", "p"]
    )
    assert result.exit_code == 0, result.output
    assert captured == {"csv": "scan_articled.csv", "out": "p"}


def test_research_render_catalysts_defaults(monkeypatch):
    captured = {}

    def fake(input_csv, *, threshold, out_dir):
        captured.update(csv=input_csv, threshold=threshold, out=out_dir)
        return 0

    monkeypatch.setattr("ticker_news.research.render.render_catalysts", fake)
    result = runner.invoke(cli.app, ["research", "render-catalysts"])
    assert result.exit_code == 0, result.output
    assert captured == {
        "csv": "catalyst_returns_2025-02-01_2025-03-30.csv",
        "threshold": 3.0,
        "out": "pics_bombs/catalysts",
    }


def test_research_render_all_command(monkeypatch):
    captured = {}

    def fake(csv_path, pics_dir, *, out_dir):
        captured.update(csv=csv_path, pics=pics_dir, out=out_dir)
        return 0

    monkeypatch.setattr("ticker_news.research.render.render_all_tickers", fake)
    result = runner.invoke(
        cli.app,
        ["research", "render-all", "--csv", "c.csv", "--pics-dir", "pics", "--out-dir", "o"],
    )
    assert result.exit_code == 0, result.output
    assert captured == {"csv": "c.csv", "pics": "pics", "out": "o"}


def test_research_render_all_defaults(monkeypatch):
    captured = {}

    def fake(csv_path, pics_dir, *, out_dir):
        captured.update(csv=csv_path, pics=pics_dir, out=out_dir)
        return 0

    monkeypatch.setattr("ticker_news.research.render.render_all_tickers", fake)
    result = runner.invoke(cli.app, ["research", "render-all"])
    assert result.exit_code == 0, result.output
    assert captured == {
        "csv": "catalyst_returns_2025-02-01_2025-03-30.csv",
        "pics": "pics_bombs/catalysts/other_related",
        "out": None,
    }


class _FakeConn:
    def close(self):
        pass


def test_eval_classify_passes_args(monkeypatch):
    captured = {}

    def fake_run_eval(variants, **kwargs):
        captured["variants"] = variants
        captured.update(kwargs)
        return []

    from ticker_news.evals import classify_eval
    monkeypatch.setattr(classify_eval, "run_eval", fake_run_eval)
    result = runner.invoke(cli.app, [
        "eval", "classify", "--variant", "binary", "--mode", "lite",
        "--gt-csv", "gt.csv", "--ids", "595,14682", "--dsn", "postgresql://x",
        "--run-name", "tuning-1",
    ])
    assert result.exit_code == 0, result.output
    assert captured["variants"] == ("binary",)
    assert captured["mode"] == "lite"
    assert captured["gt_csv"] == "gt.csv"
    assert captured["ids"] == [595, 14682]
    assert captured["dsn"] == "postgresql://x"
    assert captured["run_name"] == "tuning-1"


def test_eval_classify_defaults_to_both_two_pass(monkeypatch):
    captured = {}

    def fake_run_eval(variants, **kwargs):
        captured["variants"] = variants
        captured.update(kwargs)
        return []

    from ticker_news.evals import classify_eval
    monkeypatch.setattr(classify_eval, "run_eval", fake_run_eval)
    result = runner.invoke(cli.app, ["eval", "classify"])
    assert result.exit_code == 0, result.output
    assert captured["variants"] == ("binary", "finegrained")
    assert captured["mode"] == "two-pass"
    assert captured["dataset_name"] == "classify-ground-truth"
    assert captured["ids"] is None


def test_eval_classify_rejects_bad_variant():
    result = runner.invoke(cli.app, ["eval", "classify", "--variant", "ternary"])
    assert result.exit_code != 0


def test_eval_classify_rejects_bad_mode():
    result = runner.invoke(cli.app, ["eval", "classify", "--mode", "warp"])
    assert result.exit_code != 0


def test_eval_classify_rejects_bad_ids():
    result = runner.invoke(cli.app, ["eval", "classify", "--ids", "1,x"])
    assert result.exit_code != 0
