"""E2E pipeline eval: re-run articles through the real stage chain against the
shared DB, then score the verdict against the actual price move as a Langfuse
experiment.

Scoring is directional agreement: buy+up = 1, sell+down = 1, wrong direction
= 0; hold / no-verdict / no-price-data are excluded (value None) with an
explanatory comment. The raw entry->close move is recorded as a second score.

Design: docs/superpowers/specs/2026-06-11-e2e-pipeline-eval-design.md
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import psycopg
from langfuse import Evaluation

# Stages whose outputs may be preserved across eval runs. Sentiment is never
# skippable - the verdict is what the experiment scores.
SKIPPABLE_STAGES = ("embed", "classify", "tag", "insights")

# stage -> articles columns nulled when the stage is NOT kept.
_STAGE_COLUMNS = {
    "embed": ("embedding",),
    "classify": ("category", "category_reason"),
    "tag": ("primary_ticker", "primary_segment", "more_tickers", "more_segments"),
    "insights": ("insights_extracted_at",),
}


def parse_skip_stages(raw: str | None) -> frozenset[str]:
    """Validate a comma-separated --skip-stages value. ValueError on unknowns.

    The sentinel ``all`` expands to every skippable stage (reuse all stable
    upstream outputs, re-run only sentiment); it must stand alone.
    """
    if not raw:
        return frozenset()
    stages = {s.strip() for s in raw.split(",") if s.strip()}
    if "all" in stages:
        if stages != {"all"}:
            raise ValueError("'all' cannot be combined with other stages")
        return frozenset(SKIPPABLE_STAGES)
    unknown = stages - set(SKIPPABLE_STAGES)
    if unknown:
        raise ValueError(
            f"unknown stage(s): {', '.join(sorted(unknown))} "
            f"(skippable: {', '.join(SKIPPABLE_STAGES)})"
        )
    return frozenset(stages)


def parse_ids_file(path: str | Path) -> list[int]:
    """One article id per line (utf-8, BOM/trailing-comma tolerant)."""
    ids: list[int] = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for lineno, line in enumerate(text.splitlines(), start=1):
        token = line.strip().rstrip(",").strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"{path} line {lineno}: not an article id: {token!r}")
        ids.append(int(token))
    return ids


def score_directional(
    action: str | None, gain_pct: float | None, *, skip_reason: str | None = None
) -> tuple[float | None, str]:
    """Directional-agreement score for one verdict vs the realized move.

    Returns (value, comment) where value is 1.0 / 0.0, or None when the item
    cannot be scored (hold, no verdict, no price data) — None scores are
    excluded from Langfuse aggregates by the run evaluator.
    """
    if action is None:
        return None, f"no verdict ({skip_reason or 'sentiment skipped'})"
    if action == "hold":
        return None, "hold verdict - no direction to verify"
    if action not in ("buy", "sell"):
        return None, f"unknown action '{action}'"
    if gain_pct is None:
        return None, f"no price data ({skip_reason or 'unknown'})"
    correct = gain_pct > 0 if action == "buy" else gain_pct < 0
    verdict = "agree" if correct else "disagree"
    return (1.0 if correct else 0.0), f"{action} with {gain_pct:+.2f}% by close -> {verdict}"


def upsert_dataset_items(client, dataset_name: str, items: list[dict]) -> None:
    """Idempotently seed dataset items keyed on input.article_id.

    Langfuse item ids are PROJECT-scoped, so a bare deterministic id like
    "article-595" collides when the same article appears in two datasets.
    Items that already exist in this dataset (matched on article_id) are
    updated via their existing id - whatever scheme it was created under;
    new items get a dataset-scoped deterministic id, so re-seeding never
    duplicates and never collides across datasets.
    """
    try:
        existing = {
            (it.input or {}).get("article_id"): it.id
            for it in client.get_dataset(dataset_name).items
        }
    except Exception:  # noqa: BLE001 - dataset empty/just created
        existing = {}
    for it in items:
        article_id = it["input"]["article_id"]
        kwargs = dict(
            dataset_name=dataset_name,
            id=existing.get(article_id) or f"{dataset_name}:article-{article_id}",
            input=it["input"],
            metadata=it.get("metadata"),
        )
        if "expected_output" in it:
            kwargs["expected_output"] = it["expected_output"]
        client.create_dataset_item(**kwargs)


def connect_eval(dsn: str | None = None) -> psycopg.Connection:
    """Fresh transactional connection to the eval target DB (DSN overridable).

    pgvector registration matches the worker convention (db.connect(vector=True));
    a separate helper because db.connect() cannot take an explicit DSN.
    """
    from pgvector.psycopg import register_vector

    from ticker_news.shared.config import get_settings

    conn = psycopg.connect(dsn or get_settings().database_url)
    register_vector(conn)
    return conn


def ensure_eval_schema(conn: psycopg.Connection) -> None:
    """Additively heal an older shared schema; safe to run every time.

    Checks before ALTERing: even a no-op ADD COLUMN IF NOT EXISTS takes an
    ACCESS EXCLUSIVE lock on the shared articles table, which can queue behind
    the production service's long read transactions.
    """
    from ticker_news.sentiment import store as sentiment_store

    existing = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'articles'"
        ).fetchall()
    }
    wanted = {"insights_extracted_at": "timestamptz", "provider_sentiments": "jsonb"}
    for column, pg_type in wanted.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS {column} {pg_type}"
            )
    conn.commit()
    sentiment_store.ensure_schema(conn)


def build_items(conn: psycopg.Connection, ids: list[int]) -> list[dict]:
    """Load articles as Langfuse local experiment items; reject unusable ones.

    The input payload is JSON-only (datetimes as ISO strings) so the same dict
    can be upserted as a Langfuse dataset item.
    """
    rows = conn.execute(
        "SELECT id, url, primary_ticker, published_utc, title, status, "
        "coalesce(content, '') <> '' "
        "FROM public.articles WHERE id = ANY(%s) ORDER BY id",
        (ids,),
    ).fetchall()
    missing = sorted(set(ids) - {r[0] for r in rows})
    if missing:
        raise ValueError(f"article ids not found: {missing}")
    items = []
    for aid, url, ticker, published, title, status, has_content in rows:
        if status != "ok" or not has_content:
            raise ValueError(f"article {aid} has no scraped content (status={status})")
        if published is None:
            raise ValueError(f"article {aid} has no published_utc; cannot price the entry")
        items.append({
            "input": {
                "article_id": aid,
                "url": url,
                "published_utc": published.astimezone(timezone.utc).isoformat(),
                "title": title or "",
            },
            "metadata": {"seed_ticker": ticker or ""},
        })
    return items


def reset_article(
    conn: psycopg.Connection, article_id: int, keep: frozenset[str] = frozenset()
) -> None:
    """Clear derived fields so the idempotent stage adapters re-run.

    Stages named in `keep` retain their outputs, so the corresponding
    adapters no-op naturally (no LLM/API cost). Scraped content is never
    touched; the sentiment verdict is always cleared (it is what the eval
    scores). One transaction: an eval article is never left half-reset.
    """
    conn.execute(
        "DELETE FROM public.article_sentiment WHERE article_id = %s", (article_id,)
    )
    if "insights" not in keep:
        conn.execute(
            "DELETE FROM public.article_insights WHERE article_id = %s", (article_id,)
        )
    columns = [
        col for stage in SKIPPABLE_STAGES if stage not in keep
        for col in _STAGE_COLUMNS[stage]
    ]
    if columns:
        assignments = ", ".join(f"{col} = NULL" for col in columns)
        conn.execute(
            f"UPDATE public.articles SET {assignments} WHERE id = %s",
            (article_id,),
        )
    conn.commit()


def realized_move(ticker: str, published_utc: datetime) -> tuple[float | None, str | None]:
    """Entry->close move per the backtest rule. Returns (gain_pct, error).

    Entry = first tradeable minute bar at/after publication; exit = that day's
    regular close, or the next session's close for extended-hours entries
    (include_after_hours=True so evening articles still get scored).
    """
    from ticker_news.research import ticker_scan as ts

    pub_et = published_utc.astimezone(ts.MARKET_TZ)
    frm = (pub_et.date() - timedelta(days=1)).isoformat()
    to = (pub_et.date() + timedelta(days=7)).isoformat()
    try:
        prices = ts.fetch_prices(ticker, frm, to)
    except RuntimeError as exc:  # covers ScanError; missing MASSIVE_API_KEY etc.
        return None, str(exc)
    sim = ts.simulate(
        {"id": 0, "ticker": ticker, "published_et": pub_et}, prices,
        include_after_hours=True,
    )
    if sim is None:
        return None, "no tradeable entry/exit bar"
    return sim["gain_pct"], None


@lru_cache(maxsize=256)
def _cached_move(ticker: str, published_iso: str) -> tuple[float | None, str | None]:
    """Both item evaluators need the move; fetch Massive bars only once."""
    return realized_move(ticker, datetime.fromisoformat(published_iso))


def directional_agreement_evaluator(*, input, output, **kwargs) -> Evaluation:
    """Langfuse item evaluator (input/output names fixed by its contract).

    Langfuse rejects value=None scores, so excluded items (hold, no verdict,
    no price data) are recorded as a categorical sibling score instead —
    visible and filterable rather than silently absent.
    """
    out = output or {}
    action, ticker = out.get("action"), out.get("ticker")
    gain_pct, price_err = None, "no primary ticker"
    if ticker:
        gain_pct, price_err = _cached_move(ticker, input["published_utc"])
    value, comment = score_directional(
        action, gain_pct, skip_reason=out.get("skip_reason") or price_err
    )
    if value is None:
        return Evaluation(name="directional_agreement_skip", value=comment)
    return Evaluation(name="directional_agreement", value=value, comment=comment)


def price_move_evaluator(*, input, output, **kwargs) -> Evaluation:
    """Langfuse item evaluator: raw entry->close move, recorded even when unscored."""
    ticker = (output or {}).get("ticker")
    if not ticker:
        return Evaluation(name="price_move_pct_skip", value="no primary ticker")
    gain_pct, price_err = _cached_move(ticker, input["published_utc"])
    if gain_pct is None:
        return Evaluation(name="price_move_pct_skip", value=price_err or "unknown")
    return Evaluation(
        name="price_move_pct", value=gain_pct,
        comment=f"entry->close move {gain_pct:+.2f}%",
    )


def avg_directional_agreement(*, item_results, **kwargs) -> Evaluation:
    values = [
        e.value
        for r in item_results
        for e in r.evaluations
        if e.name == "directional_agreement" and e.value is not None
    ]
    if not values:
        return Evaluation(
            name="avg_directional_agreement_skip", value="no scorable items"
        )
    avg = sum(values) / len(values)
    return Evaluation(
        name="avg_directional_agreement", value=avg,
        comment=f"scored {len(values)}/{len(item_results)} items, avg {avg:.2f}",
    )


def _warn_failed_items(result, requested_ids: list[int]) -> None:
    """Failed items vanish from the result (SDK logs only); make them loud.

    A failed item is an article left reset-but-not-rebuilt in the shared DB -
    re-run it through the eval, or let the batch CLIs re-process it.
    """
    done: set[int] = set()
    for r in result.item_results:
        item = r.item
        data = item["input"] if isinstance(item, dict) else item.input
        done.add(data["article_id"])
    failed = sorted(set(requested_ids) - done)
    if failed:
        print(
            f"WARNING: {len(failed)} item(s) errored and were left reset in the DB: "
            f"{failed}. Re-run them (ticker-news eval pipeline --ids "
            f"{','.join(map(str, failed))}) or heal via the batch CLIs."
        )


def make_task(
    dsn: str | None,
    skip_stages: frozenset[str] = frozenset(),
    precedent_source: str | None = None,
):
    """Experiment task: reset the article, run the real stage chain, return the verdict.

    A fresh connection per invocation — sync psycopg connections must not be
    shared across the runner's concurrent task calls (same rule as the worker).
    `precedent_source` selects the sentiment precedent flow for this run.
    """

    def run_pipeline_task(*, item, **kwargs) -> dict:
        from langfuse import propagate_attributes

        from ticker_news.service import stages
        from ticker_news.shared import observability as obs

        data = item["input"] if isinstance(item, dict) else item.input
        article_id, url = data["article_id"], data["url"]
        conn = connect_eval(dsn)
        # The SDK names every experiment-item trace AND its root span
        # "experiment-item-run"; rename both — the trace via
        # propagate_attributes, the root span in place.
        with propagate_attributes(
            trace_name=f"{EXPERIMENT_NAME}:article-{article_id}"
        ):
            if (lf := obs.client()) is not None:
                lf.update_current_span(name=f"{EXPERIMENT_NAME}:article-{article_id}")
            try:
                reset_article(conn, article_id, keep=skip_stages)
                tag_ctx = stages.TagContext.load(conn)
                with obs.stage_span("embed"):
                    stages.embed_stage(conn, url)
                with obs.stage_span("classify"):
                    category = stages.classify_stage(conn, url)
                if category is None:
                    # classify no-ops when category is already set (e.g. the stage
                    # was kept via --skip-stages); report the stored value.
                    row = conn.execute(
                        "SELECT category FROM public.articles WHERE id = %s",
                        (article_id,),
                    ).fetchone()
                    category = row[0] if row else None
                with obs.stage_span("tag"):
                    stages.tag_stage(conn, url, tag_ctx)
                with obs.stage_span("insights"):
                    stages.insights_stage(conn, url, tag_ctx)
                with obs.stage_span("sentiment"):
                    verdict = stages.sentiment_stage(
                        conn, url, precedent_source=precedent_source
                    )
                if verdict is None:
                    row = conn.execute(
                        "SELECT primary_ticker FROM public.articles WHERE id = %s",
                        (article_id,),
                    ).fetchone()
                    ticker = row[0] if row else None
                    reason = (
                        f"category={category}" if category != "real news"
                        else "no primary ticker" if not ticker
                        else "sentiment skipped"
                    )
                    return {"action": None, "confidence": None, "category": category,
                            "ticker": ticker, "skip_reason": reason}
                return {"action": verdict["action"], "confidence": verdict["confidence"],
                        "category": category, "ticker": verdict["ticker"],
                        "skip_reason": None}
            finally:
                conn.close()

    return run_pipeline_task


EXPERIMENT_NAME = "pipeline-e2e"
_DESCRIPTION = (
    "Full post-scrape pipeline re-run per article; verdict scored against the "
    "realized entry->close price move (Massive)."
)


def run_eval(
    ids: list[int],
    *,
    dataset_name: str | None = None,
    dsn: str | None = None,
    run_name: str | None = None,
    skip_stages: frozenset[str] = frozenset(),
    precedent_source: str | None = None,
):
    """Run the E2E pipeline experiment; returns the langfuse ExperimentResult.

    Local-data mode runs exactly `ids`. Dataset mode upserts `ids` as items
    (deterministic id => idempotent) and runs over the WHOLE dataset, so the
    dataset acts as the growing eval suite.
    """
    from ticker_news.shared import observability as obs
    from ticker_news.shared.config import get_settings

    client = obs.client()
    if client is None:
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are required - "
            "eval results live in Langfuse."
        )
    s = get_settings()
    missing = [
        name
        for name, val in (
            ("MASSIVE_API_KEY", s.massive_api_key),
            ("GOOGLE_API_KEY", s.google_api_key),
            ("OPENAI_API_KEY", s.openai_api_key),
        )
        if not val
    ]
    if missing:
        raise SystemExit(f"missing required keys: {', '.join(missing)}")

    conn = connect_eval(dsn)
    try:
        ensure_eval_schema(conn)
        try:
            items = build_items(conn, ids) if ids else []
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    finally:
        conn.close()

    metadata = {"entrypoint": "eval"}
    if skip_stages:
        metadata["skipped_stages"] = sorted(skip_stages)
    metadata["precedent_source"] = precedent_source or s.precedent_source
    common = dict(
        name=EXPERIMENT_NAME,
        run_name=run_name,
        description=_DESCRIPTION,
        task=make_task(dsn, skip_stages, precedent_source),
        evaluators=[directional_agreement_evaluator, price_move_evaluator],
        run_evaluators=[avg_directional_agreement],
        # Sync tasks run serially in langfuse 4.7.1 (no to_thread); this only
        # caps the async evaluators. Kept low deliberately - each item already
        # fans out ~7 LLM calls inside the stages.
        max_concurrency=2,
        metadata=metadata,
    )
    try:
        if dataset_name:
            try:
                client.create_dataset(name=dataset_name, description=_DESCRIPTION)
            except Exception:  # noqa: BLE001 - already exists is fine
                pass
            upsert_dataset_items(client, dataset_name, items)
            dataset = client.get_dataset(dataset_name)
            if not dataset.items:
                raise SystemExit(f"dataset '{dataset_name}' has no items")
            dataset_ids = [it.input["article_id"] for it in dataset.items]
            result = dataset.run_experiment(**common)
            _warn_failed_items(result, dataset_ids)
            return result
        if not items:
            raise SystemExit("no article ids given (use --ids, or --dataset with items)")
        result = client.run_experiment(data=items, **common)
        _warn_failed_items(result, [it["input"]["article_id"] for it in items])
        return result
    finally:
        obs.flush()
