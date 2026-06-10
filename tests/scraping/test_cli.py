from ticker_news.scraping.cli import build_settings, parse_args


def test_ignore_robots_flag_overrides_settings():
    args = parse_args(["--csv", "x.csv", "--ignore-robots", "--concurrency", "4"])
    settings = build_settings(args)
    assert settings.respect_robots is False
    assert settings.concurrency == 4
    assert args.csv == "x.csv"


def test_defaults_respect_robots():
    args = parse_args(["--csv", "x.csv"])
    settings = build_settings(args)
    assert settings.respect_robots is True
