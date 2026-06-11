"""Chart-rendering batch tools over scan / catalyst CSVs.

Consolidates the legacy ticker-scan ``{render_bombs,
render_catalyst_bombs,render_all_tickers}.py`` scripts into one module. All chart
drawing delegates to :func:`ticker_news.research.candles.make_chart`, whose
pandas / mplfinance dependencies (the ``charts`` extra) are imported lazily —
importing this module and planning jobs needs none of them. DB access (the
:func:`render_all_tickers` ticker lookup) goes through
:func:`ticker_news.shared.db.connect`.

Three tools, all writing ``<ticker>_<article_id>_<date>.jpg``:

* :func:`render_bombs` — one marked chart per article attached in an
  ``attach-articles`` output CSV. Same-day articles are marked at their own
  ``published_et``; prev-after-hours articles broke after the previous close,
  so they are marked at the trading day's 04:00 ET premarket open instead.
* :func:`render_catalysts` — entry-point charts for every catalyst_returns
  row whose ``|gain_pct|`` is strictly above the threshold, marked at
  ``buy_et``.
* :func:`render_all_tickers` — given a pics folder of already-rendered
  charts plus the catalyst CSV behind them, look up each article's full
  ticker set (``primary_ticker`` + ``more_tickers``) in the DB and render
  the missing non-primary-ticker charts at the same ``buy_et`` moment.

Conscious deviation from legacy: :func:`render_bombs` and
:func:`render_catalysts` now skip charts whose output file already exists
(legacy re-rendered them on every run); :func:`render_all_tickers` always
behaved this way. Each function returns the number of charts written.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from contextlib import redirect_stdout
from typing import Iterable, NamedTuple, Sequence

from ticker_news.research import candles

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

PREMARKET_START = "04:00"  # ET pre-market open
DEFAULT_CATALYST_CSV = "catalyst_returns_2025-02-01_2025-03-30.csv"
DEFAULT_PICS_DIR = "pics_bombs/catalysts/other_related"
_FNAME = re.compile(r"^[A-Z0-9.]+_(\d+)_\d{4}-\d{2}-\d{2}\.jpg$")
_CSV_FIELD_LIMIT = 10**7  # the articles JSON column can be large


class RenderJob(NamedTuple):
    ticker: str
    article_id: int | str  # int (attach JSON) or str (catalyst CSV); used verbatim
    ts: str             # timestamp make_chart should mark
    filename: str       # <ticker>_<article_id>_<date>.jpg


def chart_filename(ticker: str, article_id: int | str, day: str) -> str:
    return f"{ticker}_{article_id}_{day}.jpg"


# --------------------------------------------------------------------------- #
# Pure job planning
# --------------------------------------------------------------------------- #
def bomb_jobs(rows: Iterable[dict]) -> list[RenderJob]:
    """Articled-CSV rows -> one job per attached article.

    Same-day articles are marked at their own ``published_et``;
    prev-after-hours ones at the trading day's 04:00 ET premarket open.
    """
    jobs: list[RenderJob] = []
    for r in rows:
        arts = json.loads(r.get("articles") or "[]")
        ticker, row_date = r["ticker"].strip().upper(), r["date"]
        for a in arts:
            if a.get("when") == "prev-after-hours":
                ts = f"{row_date} {PREMARKET_START}"
            else:
                ts = a["published_et"]
            jobs.append(RenderJob(ticker, a["id"], ts,
                                  chart_filename(ticker, a["id"], row_date)))
    return jobs


def catalyst_jobs(rows: Iterable[dict], threshold: float) -> list[RenderJob]:
    """Catalyst-CSV rows with |gain_pct| strictly above `threshold` -> jobs."""
    jobs: list[RenderJob] = []
    for r in rows:
        try:
            gain = float(r["gain_pct"])
        except (KeyError, ValueError):
            continue
        if abs(gain) > threshold:
            ticker, buy_et = r["ticker"].strip().upper(), r["buy_et"]
            jobs.append(RenderJob(ticker, r["article_id"], buy_et,
                                  chart_filename(ticker, r["article_id"],
                                                 buy_et.split(" ")[0])))
    return jobs


def split_existing(
    jobs: Sequence[RenderJob], out_dir: str
) -> tuple[list[tuple[RenderJob, str]], int]:
    """(job, out_path) pairs still to render + count skipped as already present."""
    pending: list[tuple[RenderJob, str]] = []
    skipped = 0
    for job in jobs:
        out = os.path.join(out_dir, job.filename)
        if os.path.exists(out):
            skipped += 1
        else:
            pending.append((job, out))
    return pending, skipped


# --------------------------------------------------------------------------- #
# Render loop
# --------------------------------------------------------------------------- #
def _render(pending: Sequence[tuple[RenderJob, str]]) -> tuple[int, int]:
    """Render each (job, out_path); return (written, failed). Keeps going on errors."""
    done = failed = 0
    iterator = tqdm(pending, unit="chart") if tqdm else pending
    for job, out in iterator:
        try:
            with open(os.devnull, "w") as null, redirect_stdout(null):
                candles.make_chart(job.ticker, job.ts, out=out)
            done += 1
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failed += 1
            print(f"  {job.ticker} {job.article_id} ({job.ts}): {exc}", file=sys.stderr)
    return done, failed


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def render_bombs(input_csv: str, out_dir: str = "pics_bombs") -> int:
    """Render a chart for every article attached in an articled CSV."""
    os.makedirs(out_dir, exist_ok=True)
    csv.field_size_limit(_CSV_FIELD_LIMIT)
    with open(input_csv, newline="") as fh:
        jobs = bomb_jobs(csv.DictReader(fh))
    if not jobs:
        print("No article attachments found in the CSV.")
        return 0

    pending, skipped = split_existing(jobs, out_dir)
    print(f"Rendering {len(pending)} chart(s) to {out_dir}/ "
          f"({skipped} already present) ...")
    done, failed = _render(pending)
    print(f"Done. Wrote {done} chart(s); {failed} failed.")
    return done


def render_catalysts(
    input_csv: str, *, threshold: float = 3.0, out_dir: str = "pics_bombs/catalysts"
) -> int:
    """Render entry-point charts for the big movers in a catalyst_returns CSV."""
    os.makedirs(out_dir, exist_ok=True)
    csv.field_size_limit(_CSV_FIELD_LIMIT)
    with open(input_csv, newline="") as fh:
        jobs = catalyst_jobs(csv.DictReader(fh), threshold)
    if not jobs:
        print(f"No rows with |gain_pct| > {threshold:g} in {input_csv}.")
        return 0

    pending, skipped = split_existing(jobs, out_dir)
    print(f"Rendering {len(pending)} mover(s) (|gain| > {threshold:g}%) to {out_dir}/ "
          f"({skipped} already present) ...")
    done, failed = _render(pending)
    print(f"Done. Wrote {done} chart(s); {failed} failed.")
    return done


def article_tickers(aids: Sequence[int]) -> dict[int, list[str]]:
    """Map article id -> [primary_ticker, *more_tickers] (deduped, order-preserving)."""
    from ticker_news.shared import db

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, primary_ticker, more_tickers FROM public.articles "
            "WHERE id = ANY(%s)",
            (list(aids),),
        )
        rows = cur.fetchall()
    out: dict[int, list[str]] = {}
    for aid, primary, more in rows:
        seen: set[str] = set()
        tickers: list[str] = []
        for t in [primary, *(more or [])]:
            if t and t.upper() not in seen:
                seen.add(t.upper())
                tickers.append(t.upper())
        out[aid] = tickers
    return out


def all_ticker_jobs(
    aids: Sequence[int],
    buy_et: dict[int, str],
    tickers_by_aid: dict[int, list[str]],
) -> tuple[list[RenderJob], int]:
    """Every ticker of every article at its buy_et -> jobs + missing-buy_et count."""
    jobs: list[RenderJob] = []
    no_buyet = 0
    for aid in aids:
        ts = buy_et.get(aid)
        if not ts:
            no_buyet += 1
            continue
        bdate = ts.split(" ")[0]
        for tk in tickers_by_aid.get(aid, []):
            jobs.append(RenderJob(tk, aid, ts, chart_filename(tk, aid, bdate)))
    return jobs, no_buyet


def render_all_tickers(csv_path: str, pics_dir: str, out_dir: str | None = None) -> int:
    """For each article behind `pics_dir`, render the charts of EVERY ticker it names."""
    out_dir = out_dir or pics_dir
    os.makedirs(out_dir, exist_ok=True)
    csv.field_size_limit(_CSV_FIELD_LIMIT)

    # aid -> buy_et, from the CSV (the shared entry moment for all of an article's tickers)
    buy_et: dict[int, str] = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            buy_et[int(r["article_id"])] = r["buy_et"]

    # article ids present in the pics folder
    aids = sorted({int(m.group(1)) for f in os.listdir(pics_dir)
                   if (m := _FNAME.match(f))})
    if not aids:
        print(f"No <ticker>_<id>_<date>.jpg charts found in {pics_dir}.")
        return 0
    tickers_by_aid = article_tickers(aids)

    jobs, no_buyet = all_ticker_jobs(aids, buy_et, tickers_by_aid)
    pending, skipped = split_existing(jobs, out_dir)
    print(f"{len(aids)} article(s); {len(pending)} new ticker-chart(s) to render "
          f"({skipped} already present"
          f"{f', {no_buyet} missing buy_et' if no_buyet else ''}).")
    if not pending:
        return 0

    done, failed = _render(pending)
    print(f"Done. Wrote {done} chart(s); {failed} failed.")
    return done
