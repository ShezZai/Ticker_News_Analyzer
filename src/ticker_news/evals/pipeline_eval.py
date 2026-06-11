"""E2E pipeline eval: re-run articles through the real stage chain against the
shared DB, then score the verdict against the actual price move as a Langfuse
experiment.

Scoring is directional agreement: buy+up = 1, sell+down = 1, wrong direction
= 0; hold / no-verdict / no-price-data are excluded (value None) with an
explanatory comment. The raw entry->close move is recorded as a second score.

Design: docs/superpowers/specs/2026-06-11-e2e-pipeline-eval-design.md
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg


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
    """Additively heal an older shared schema; safe to run every time."""
    from ticker_news.sentiment import store as sentiment_store

    conn.execute(
        "ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS insights_extracted_at timestamptz"
    )
    conn.execute(
        "ALTER TABLE public.articles ADD COLUMN IF NOT EXISTS provider_sentiments jsonb"
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


def reset_article(conn: psycopg.Connection, article_id: int) -> None:
    """Clear every derived field so the idempotent stage adapters re-run.

    Scraped content is untouched. One transaction: an eval article is never
    left half-reset.
    """
    conn.execute(
        "DELETE FROM public.article_sentiment WHERE article_id = %s", (article_id,)
    )
    conn.execute(
        "DELETE FROM public.article_insights WHERE article_id = %s", (article_id,)
    )
    conn.execute(
        "UPDATE public.articles SET embedding = NULL, category = NULL, "
        "category_reason = NULL, primary_ticker = NULL, primary_segment = NULL, "
        "more_tickers = NULL, more_segments = NULL, insights_extracted_at = NULL "
        "WHERE id = %s",
        (article_id,),
    )
    conn.commit()
