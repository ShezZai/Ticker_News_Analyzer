"""Embedding pipeline: fetch unembedded articles, vectorise, write back.

Resumable: only rows with ``embedding IS NULL`` are processed unless
``reembed=True``. Uses the shared ``ticker_news.shared.db.connect`` and the
shared ``embed_texts`` from ``ticker_news.embedding.embedder``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import psycopg

from ticker_news.embedding.embedder import build_text, embed_texts
from ticker_news.shared.llm import EMBED_DIM, EMBED_MODEL

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

DEFAULT_BATCH = 256  # inputs per embeddings request


def ensure_schema(conn: psycopg.Connection) -> None:
    """Verify pgvector is present and add the embedding column if missing."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone() is None:
            raise SystemExit(
                "pgvector extension is not installed in this database.\n"
                "Create it once as a superuser:\n"
                "    sudo -u postgres psql -d news -c 'CREATE EXTENSION vector;'"
            )
        cur.execute(
            f"ALTER TABLE public.articles "
            f"ADD COLUMN IF NOT EXISTS embedding vector({EMBED_DIM});"
        )
    conn.commit()


def ids_to_process(
    conn: psycopg.Connection, reembed: bool, limit: Optional[int]
) -> List[int]:
    """Return the ordered list of article ids that need an embedding."""
    where = "" if reembed else "WHERE embedding IS NULL"
    lim = f"LIMIT {int(limit)}" if limit else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM public.articles {where} ORDER BY id {lim}")
        return [row[0] for row in cur.fetchall()]


def fetch_rows(conn: psycopg.Connection, ids: Sequence[int]) -> List[tuple]:
    """Fetch (id, title, content) for the given ids."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, content FROM public.articles WHERE id = ANY(%s) "
            "ORDER BY id",
            (list(ids),),
        )
        return cur.fetchall()


def create_index(conn: psycopg.Connection) -> None:
    """Build an HNSW cosine index over the embeddings (idempotent)."""
    print("Building HNSW cosine index (articles_embedding_idx)...")
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_embedding_idx "
            "ON public.articles USING hnsw (embedding vector_cosine_ops);"
        )
    conn.commit()


def embed_all(
    batch_size: int = DEFAULT_BATCH,
    limit: Optional[int] = None,
    reembed: bool = False,
    build_index: bool = True,
) -> int:
    """Embed articles and write vectors back. Returns the number embedded."""
    from ticker_news.shared.db import connect

    conn = connect(vector=True)
    try:
        ensure_schema(conn)
        ids = ids_to_process(conn, reembed=reembed, limit=limit)
        if not ids:
            print("Nothing to embed.")
            return 0

        print(f"Embedding {len(ids)} article(s) via {EMBED_MODEL} "
              f"in batches of {batch_size} ...")
        batches = range(0, len(ids), batch_size)
        pbar = tqdm(batches, unit="batch") if tqdm else batches

        done = 0
        for start in pbar:
            chunk = ids[start : start + batch_size]
            rows = fetch_rows(conn, chunk)

            texts, row_ids = [], []
            for rid, title, content in rows:
                text = build_text(title, content)
                if not text:
                    continue
                texts.append(text)
                row_ids.append(rid)

            if not texts:
                continue
            embeddings = embed_texts(texts, batch_size=batch_size)
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE public.articles SET embedding = %s WHERE id = %s",
                    list(zip(embeddings, row_ids)),
                )
            conn.commit()
            done += len(row_ids)

        print(f"Embedded {done} article(s).")
        if build_index and done:
            create_index(conn)
        return done
    finally:
        conn.close()
