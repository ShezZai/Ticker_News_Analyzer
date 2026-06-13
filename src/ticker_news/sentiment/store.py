"""Persistence for sentiment verdicts: public.article_sentiment."""

from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from ticker_news.sentiment.schemas import Verdict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS article_sentiment (
    article_id  bigint NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    ticker      text NOT NULL,
    action      text NOT NULL,
    confidence  real NOT NULL,
    reasoning   text,
    analyses    jsonb NOT NULL DEFAULT '[]'::jsonb,
    model       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, ticker)
);
CREATE INDEX IF NOT EXISTS article_sentiment_ticker_idx ON article_sentiment (ticker)
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    for statement in (s.strip() for s in _SCHEMA.split(";")):
        if statement:
            conn.execute(statement)
    conn.commit()


def save_verdict(
    conn: psycopg.Connection,
    article_id: int,
    ticker: str,
    verdict: Verdict,
    analyses: list[dict],
    model: str,
) -> None:
    conn.execute(
        "INSERT INTO article_sentiment "
        "(article_id, ticker, action, confidence, reasoning, analyses, model) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (article_id, ticker) DO NOTHING",
        (article_id, ticker, verdict.action, verdict.confidence,
         verdict.reasoning or None, Jsonb(analyses), model),
    )
    conn.commit()


def has_verdict(conn: psycopg.Connection, article_id: int, ticker: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM article_sentiment WHERE article_id = %s AND ticker = %s",
        (article_id, ticker),
    ).fetchone()
    return row is not None
