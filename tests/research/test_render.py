"""Offline tests for the chart-rendering batch tools (monkeypatched make_chart)."""

import csv
import json

from ticker_news.research import candles, render


# --------------------------------------------------------------------------- #
# bomb_jobs — articles-JSON-column parsing
# --------------------------------------------------------------------------- #
def test_bomb_jobs_parses_articles_json_column():
    row = {
        "ticker": " nvda ",
        "date": "2025-01-07",
        "articles": json.dumps([
            {"id": 11, "published_et": "2025-01-07 10:30", "when": "same-day",
             "title": "t1", "url": "u1"},
            {"id": 12, "published_et": "2025-01-06 17:05", "when": "prev-after-hours",
             "title": "t2", "url": "u2"},
        ]),
    }
    jobs = render.bomb_jobs([row])
    assert len(jobs) == 2

    same_day, prev_ah = jobs
    # same-day article -> marked at its own published_et
    assert same_day.ticker == "NVDA"
    assert same_day.ts == "2025-01-07 10:30"
    assert same_day.filename == "NVDA_11_2025-01-07.jpg"
    # prev-after-hours article -> marked at 04:00 ET premarket open of the
    # trading day (the row's date), NOT its own published time
    assert prev_ah.ts == "2025-01-07 04:00"
    assert prev_ah.filename == "NVDA_12_2025-01-07.jpg"


def test_bomb_jobs_missing_when_defaults_to_published_et():
    row = {
        "ticker": "AMD",
        "date": "2025-01-07",
        "articles": json.dumps([{"id": 3, "published_et": "2025-01-07 09:31"}]),
    }
    (job,) = render.bomb_jobs([row])
    assert job.ts == "2025-01-07 09:31"


def test_bomb_jobs_empty_articles_column():
    assert render.bomb_jobs([{"ticker": "AMD", "date": "2025-01-07", "articles": ""}]) == []
    assert render.bomb_jobs([{"ticker": "AMD", "date": "2025-01-07", "articles": "[]"}]) == []


# --------------------------------------------------------------------------- #
# catalyst_jobs — threshold filter
# --------------------------------------------------------------------------- #
def _cat_row(aid, gain, ticker="nvda", buy_et="2025-02-03 10:30"):
    return {"article_id": str(aid), "ticker": ticker, "buy_et": buy_et,
            "gain_pct": gain}


def test_catalyst_jobs_threshold_is_strictly_greater():
    rows = [
        _cat_row(1, "3.0"),    # exactly at threshold -> excluded (legacy: strict >)
        _cat_row(2, "3.01"),   # just above -> included
        _cat_row(3, "-4.2"),   # magnitude counts: big down moves render too
        _cat_row(4, "1.0"),    # below -> excluded
    ]
    jobs = render.catalyst_jobs(rows, threshold=3.0)
    assert [j.article_id for j in jobs] == ["2", "3"]
    assert jobs[0].ticker == "NVDA"
    assert jobs[0].ts == "2025-02-03 10:30"
    # filename date comes from buy_et's date part
    assert jobs[0].filename == "NVDA_2_2025-02-03.jpg"


def test_catalyst_jobs_skips_bad_or_missing_gain():
    rows = [
        {"article_id": "1", "ticker": "A", "buy_et": "2025-02-03 10:30"},  # no gain_pct
        _cat_row(2, "n/a"),                                                # unparsable
        _cat_row(3, "10.0"),
    ]
    jobs = render.catalyst_jobs(rows, threshold=3.0)
    assert [j.article_id for j in jobs] == ["3"]


# --------------------------------------------------------------------------- #
# skip-existing + render loop
# --------------------------------------------------------------------------- #
def _write_articled_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", "ticker", "articles"])
        writer.writeheader()
        writer.writerows(rows)


def test_render_bombs_skips_existing_files(tmp_path, monkeypatch):
    csv_path = tmp_path / "scan_articled.csv"
    _write_articled_csv(csv_path, [{
        "date": "2025-01-07",
        "ticker": "NVDA",
        "articles": json.dumps([
            {"id": 11, "published_et": "2025-01-07 10:30", "when": "same-day"},
            {"id": 12, "published_et": "2025-01-06 17:05", "when": "prev-after-hours"},
        ]),
    }])
    out_dir = tmp_path / "pics"
    out_dir.mkdir()
    (out_dir / "NVDA_11_2025-01-07.jpg").write_bytes(b"existing")

    calls = []
    monkeypatch.setattr(candles, "make_chart",
                        lambda ticker, ts, *, out=None, **kw: calls.append((ticker, ts, out)))

    n = render.render_bombs(str(csv_path), out_dir=str(out_dir))
    assert n == 1
    assert calls == [("NVDA", "2025-01-07 04:00", str(out_dir / "NVDA_12_2025-01-07.jpg"))]
    # the pre-existing chart was not touched
    assert (out_dir / "NVDA_11_2025-01-07.jpg").read_bytes() == b"existing"


def test_render_bombs_counts_failures_separately(tmp_path, monkeypatch):
    csv_path = tmp_path / "scan_articled.csv"
    _write_articled_csv(csv_path, [{
        "date": "2025-01-07",
        "ticker": "NVDA",
        "articles": json.dumps([
            {"id": 1, "published_et": "2025-01-07 10:30", "when": "same-day"},
            {"id": 2, "published_et": "2025-01-07 11:30", "when": "same-day"},
        ]),
    }])

    def fake_chart(ticker, ts, *, out=None, **kw):
        if ts.endswith("10:30"):
            raise RuntimeError("no bars")
        return out

    monkeypatch.setattr(candles, "make_chart", fake_chart)
    n = render.render_bombs(str(csv_path), out_dir=str(tmp_path / "pics"))
    assert n == 1  # the failed chart is not counted as written


def test_render_catalysts_filters_and_skips(tmp_path, monkeypatch):
    csv_path = tmp_path / "catalyst_returns.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["article_id", "ticker", "buy_et", "gain_pct"])
        writer.writeheader()
        writer.writerows([
            _cat_row(1, "5.0", buy_et="2025-02-03 10:30"),
            _cat_row(2, "-6.0", buy_et="2025-02-04 09:31"),
            _cat_row(3, "2.0"),  # under threshold
        ])
    out_dir = tmp_path / "cats"
    out_dir.mkdir()
    (out_dir / "NVDA_1_2025-02-03.jpg").write_bytes(b"x")  # already rendered

    calls = []
    monkeypatch.setattr(candles, "make_chart",
                        lambda ticker, ts, *, out=None, **kw: calls.append((ticker, ts, out)))

    n = render.render_catalysts(str(csv_path), threshold=3.0, out_dir=str(out_dir))
    assert n == 1
    assert calls == [("NVDA", "2025-02-04 09:31", str(out_dir / "NVDA_2_2025-02-04.jpg"))]


# --------------------------------------------------------------------------- #
# render_all_tickers — DB seam + existing-pics skip
# --------------------------------------------------------------------------- #
def test_render_all_tickers_renders_only_missing_tickers(tmp_path, monkeypatch):
    csv_path = tmp_path / "catalyst_returns.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["article_id", "ticker", "buy_et", "gain_pct"])
        writer.writeheader()
        writer.writerows([
            _cat_row(5, "5.0", buy_et="2025-01-07 10:30"),
            _cat_row(9, "4.0", buy_et="2025-01-08 11:00", ticker="tsm"),
        ])
    pics = tmp_path / "pics"
    pics.mkdir()
    (pics / "NVDA_5_2025-01-07.jpg").write_bytes(b"primary")  # the existing primary chart

    looked_up = {}

    def fake_article_tickers(aids):
        looked_up["aids"] = list(aids)
        return {5: ["NVDA", "AMD", "TSM"]}

    calls = []
    monkeypatch.setattr(render, "article_tickers", fake_article_tickers)
    monkeypatch.setattr(candles, "make_chart",
                        lambda ticker, ts, *, out=None, **kw: calls.append((ticker, ts, out)))

    n = render.render_all_tickers(str(csv_path), str(pics))
    assert n == 2
    assert looked_up["aids"] == [5]  # only ids parsed from the pics folder
    # NVDA chart already exists -> only the other named tickers, same buy moment,
    # written next to the originals
    assert calls == [
        ("AMD", "2025-01-07 10:30", str(pics / "AMD_5_2025-01-07.jpg")),
        ("TSM", "2025-01-07 10:30", str(pics / "TSM_5_2025-01-07.jpg")),
    ]


def test_render_all_tickers_skips_articles_without_buy_et(tmp_path, monkeypatch):
    csv_path = tmp_path / "catalyst_returns.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["article_id", "ticker", "buy_et", "gain_pct"])
        writer.writeheader()
        writer.writerow(_cat_row(5, "5.0", buy_et="2025-01-07 10:30"))
    pics = tmp_path / "pics"
    pics.mkdir()
    (pics / "NVDA_5_2025-01-07.jpg").write_bytes(b"x")
    (pics / "TSM_9_2025-01-08.jpg").write_bytes(b"x")  # aid 9 has no CSV row

    monkeypatch.setattr(render, "article_tickers",
                        lambda aids: {5: ["NVDA"], 9: ["TSM", "AMD"]})
    calls = []
    monkeypatch.setattr(candles, "make_chart",
                        lambda ticker, ts, *, out=None, **kw: calls.append(ticker))

    n = render.render_all_tickers(str(csv_path), str(pics), out_dir=str(tmp_path / "other"))
    # aid 9 lacks a buy_et -> dropped; aid 5's NVDA goes to the separate out_dir
    # where no chart exists yet, so it renders there
    assert n == 1
    assert calls == ["NVDA"]


def test_render_all_tickers_empty_pics_dir(tmp_path):
    csv_path = tmp_path / "c.csv"
    with open(csv_path, "w", newline="") as fh:
        fh.write("article_id,ticker,buy_et,gain_pct\n")
    pics = tmp_path / "pics"
    pics.mkdir()
    assert render.render_all_tickers(str(csv_path), str(pics)) == 0
