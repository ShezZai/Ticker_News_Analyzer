"""Postgres-backed pipeline job queue.

One row per article URL. Workers claim with FOR UPDATE SKIP LOCKED, drive the
article through the stage chain, and advance `stage` after each step — a crash
resumes mid-article. Failures back off exponentially; over-cap jobs park as
'failed' and are requeueable via the CLI. NOTIFY 'pipeline_jobs' wakes the
service instantly on enqueue; polling is the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg

STAGES = ["scrape", "embed", "classify", "tag", "insights"]
DONE = "done"

MAX_ATTEMPTS = 5
BASE_BACKOFF_S = 30.0
BACKOFF_CAP_S = 3600.0

NOTIFY_CHANNEL = "pipeline_jobs"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    article_url     text PRIMARY KEY,
    stage           text NOT NULL DEFAULT 'scrape',
    status          text NOT NULL DEFAULT 'pending',
    attempts        int  NOT NULL DEFAULT 0,
    last_error      text,
    tickers         text[] NOT NULL DEFAULT '{}',
    published_utc   timestamptz,
    publisher       text,
    enqueued_at     timestamptz NOT NULL DEFAULT now(),
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pipeline_jobs_claim_idx
    ON pipeline_jobs (status, next_attempt_at);
"""


@dataclass
class Job:
    article_url: str
    stage: str
    attempts: int
    tickers: list[str]
    published_utc: datetime | None
    publisher: str | None


def ensure_schema(conn: psycopg.Connection) -> None:
    for statement in (s.strip() for s in _SCHEMA.split(";")):
        if statement:
            conn.execute(statement)
    conn.commit()


def next_stage(stage: str) -> str:
    idx = STAGES.index(stage)
    return STAGES[idx + 1] if idx + 1 < len(STAGES) else DONE


def backoff_delay(attempts: int) -> float:
    return min(BASE_BACKOFF_S * (2 ** attempts), BACKOFF_CAP_S)


def enqueue(conn: psycopg.Connection, item) -> bool:
    """Insert a job for a FeedItem; returns True if it was new."""
    cur = conn.execute(
        "INSERT INTO pipeline_jobs (article_url, tickers, published_utc, publisher) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (article_url) DO NOTHING",
        (item.url, item.tickers, item.published_utc, item.publisher),
    )
    new = cur.rowcount == 1
    if new:
        conn.execute(f"NOTIFY {NOTIFY_CHANNEL}")
    conn.commit()
    return new


def claim(conn: psycopg.Connection) -> Job | None:
    row = conn.execute(
        """
        UPDATE pipeline_jobs SET status = 'running', updated_at = now()
        WHERE article_url = (
            SELECT article_url FROM pipeline_jobs
            WHERE status = 'pending' AND next_attempt_at <= now()
            ORDER BY enqueued_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING article_url, stage, attempts, tickers, published_utc, publisher
        """
    ).fetchone()
    conn.commit()
    return Job(*row) if row else None


def advance(conn: psycopg.Connection, article_url: str, to_stage: str) -> None:
    """Record stage completion. to_stage == DONE finishes the job."""
    status = "done" if to_stage == DONE else "running"
    conn.execute(
        "UPDATE pipeline_jobs SET stage = %s, status = %s, attempts = 0, "
        "last_error = NULL, updated_at = now() WHERE article_url = %s",
        (to_stage, status, article_url),
    )
    conn.commit()


def fail(conn: psycopg.Connection, article_url: str, error: str,
         *, max_attempts: int = MAX_ATTEMPTS, permanent: bool = False) -> None:
    row = conn.execute(
        "UPDATE pipeline_jobs SET attempts = attempts + 1, last_error = %s, "
        "updated_at = now() WHERE article_url = %s RETURNING attempts",
        (error[:2000], article_url),
    ).fetchone()
    attempts = row[0] if row else max_attempts
    if permanent or attempts >= max_attempts:
        conn.execute(
            "UPDATE pipeline_jobs SET status = 'failed' WHERE article_url = %s",
            (article_url,),
        )
    else:
        conn.execute(
            "UPDATE pipeline_jobs SET status = 'pending', "
            "next_attempt_at = now() + %s * interval '1 second' "
            "WHERE article_url = %s",
            (backoff_delay(attempts), article_url),
        )
    conn.commit()


def recover_orphans(conn: psycopg.Connection) -> int:
    """Reset 'running' jobs to 'pending' (call once at service startup —
    single-service assumption: any running row is an orphan of a dead run)."""
    cur = conn.execute(
        "UPDATE pipeline_jobs SET status = 'pending' WHERE status = 'running'"
    )
    conn.commit()
    return cur.rowcount


def requeue_failed(conn: psycopg.Connection, article_url: str | None = None) -> int:
    where, params = ("AND article_url = %s", (article_url,)) if article_url else ("", ())
    cur = conn.execute(
        f"UPDATE pipeline_jobs SET status = 'pending', attempts = 0, "
        f"next_attempt_at = now() WHERE status = 'failed' {where}",
        params,
    )
    conn.commit()
    return cur.rowcount


def counts(conn: psycopg.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, count(*) FROM pipeline_jobs GROUP BY status"
    ).fetchall()
    return {status: n for status, n in rows}


def queue_drained(conn: psycopg.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pipeline_jobs WHERE status IN ('pending', 'running') LIMIT 1"
    ).fetchone()
    return row is None
