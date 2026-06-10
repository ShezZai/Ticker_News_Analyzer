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

    monkeypatch.setattr(cli, "pipeline_run", fake_run)
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

    monkeypatch.setattr(cli, "pipeline_run", fake_run)
    result = runner.invoke(cli.app, ["scrape", "--csv", "x.csv"])
    assert result.exit_code == 0, result.output
    assert captured["settings"].respect_robots is True


def test_scrape_rejects_zero_concurrency(monkeypatch):
    called = {}

    async def fake_run(csv, settings, *, limit=None, retry_errors=False):
        called["yes"] = True

    monkeypatch.setattr(cli, "pipeline_run", fake_run)
    result = runner.invoke(cli.app, ["scrape", "--csv", "x.csv", "--concurrency", "0"])
    assert result.exit_code != 0
    assert "yes" not in called
