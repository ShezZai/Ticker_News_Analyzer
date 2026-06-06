"""Embed every article with OpenAI text-embedding-3-small and store the vector.

Adds an `embedding vector(1536)` column to ``public.articles`` (pgvector) and
fills it in for every article by calling the OpenAI embeddings API. Re-running
only processes rows whose embedding is still NULL, so the job is fully resumable.

The text fed to the model is ``title`` + ``content`` (truncated to the model's
8191-token window). text-embedding-3 vectors are unit-normalized by OpenAI, so
cosine similarity equals inner product — an HNSW cosine index is created at the
end for fast ANN search.

Usage:
    python embed_articles.py                  # embed all rows missing an embedding
    python embed_articles.py --batch-size 256 # inputs per API request
    python embed_articles.py --limit 100      # only the first N (smoke test)
    python embed_articles.py --reembed        # recompute for every row
    python embed_articles.py --no-index       # skip building the HNSW index

Requires OPENAI_API_KEY in the environment / .env.
Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``
(peer auth over the local socket).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

MODEL_NAME = "text-embedding-3-small"
EMBED_DIM = 1536
# text-embedding-3-small hard-caps inputs at 8192 tokens; truncate just under it.
MAX_INPUT_TOKENS = 8000
DEFAULT_BATCH = 256  # inputs per embeddings request
EMBED_COST = 0.02 / 1e6  # approx text-embedding-3-small $/token (verify rates)

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"

_client = None
_encoder = None


def _truncate_tokens(text: str) -> str:
    """Trim text to <= MAX_INPUT_TOKENS tokens (cl100k_base, as used by the model).

    A token is always >= 1 character, so anything no longer than MAX_INPUT_TOKENS
    characters is already safe and skips the tokenizer entirely.
    """
    global _encoder
    text = (text or "").strip()
    if len(text) <= MAX_INPUT_TOKENS:
        return text or " "
    try:
        if _encoder is None:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        ids = _encoder.encode(text)
        if len(ids) > MAX_INPUT_TOKENS:
            text = _encoder.decode(ids[:MAX_INPUT_TOKENS])
    except ImportError:  # no tiktoken: fall back to a conservative char cap
        text = text[: MAX_INPUT_TOKENS * 3]
    return text or " "


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(DB_DSN)
    register_vector(conn)
    return conn


def _openai():
    """Lazily construct a singleton OpenAI client (reads OPENAI_API_KEY)."""
    global _client
    if _client is None:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (put it in .env).")
        _client = OpenAI()
    return _client


def embed_texts(
    texts: Sequence[str], batch_size: int = DEFAULT_BATCH, return_tokens: bool = False
):
    """Return one embedding (np.float32 array) per input text, in sub-batches.

    Empty/whitespace inputs are sent as a single space (the API rejects "") so
    the result stays index-aligned with ``texts``. numpy arrays are returned so
    the pgvector psycopg adapter can bind them directly. With
    ``return_tokens=True`` returns ``(embeddings, total_tokens)`` instead.
    """
    import numpy as np

    client = _openai()
    cleaned = [_truncate_tokens(t) for t in texts]
    out: List = []
    tokens = 0
    for start in range(0, len(cleaned), batch_size):
        chunk = cleaned[start : start + batch_size]
        resp = client.embeddings.create(model=MODEL_NAME, input=chunk)
        tokens += getattr(resp.usage, "total_tokens", 0) or 0
        # API preserves input order, but sort by index to be safe.
        for d in sorted(resp.data, key=lambda d: d.index):
            out.append(np.asarray(d.embedding, dtype=np.float32))
    return (out, tokens) if return_tokens else out


def embed_query(text: str) -> list:
    """Return the embedding for a single query string (used by search)."""
    if not text or not text.strip():
        raise ValueError("query text is empty")
    return embed_texts([text])[0]


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


def build_text(title: Optional[str], content: Optional[str]) -> str:
    """Combine title + content into a single passage for embedding.

    A generous char pre-cut bounds tokenizer work on pathologically long rows;
    embed_texts() then trims precisely to the token limit.
    """
    parts = [p.strip() for p in (title, content) if p and p.strip()]
    return "\n\n".join(parts)[: MAX_INPUT_TOKENS * 8]


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
    conn = get_conn()
    try:
        ensure_schema(conn)
        ids = ids_to_process(conn, reembed=reembed, limit=limit)
        if not ids:
            print("Nothing to embed.")
            return 0

        print(f"Embedding {len(ids)} article(s) via {MODEL_NAME} "
              f"in batches of {batch_size} ...")
        batches = range(0, len(ids), batch_size)
        pbar = tqdm(batches, unit="batch") if tqdm else batches

        done = 0
        tokens = 0
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
            embeddings, used = embed_texts(texts, batch_size=batch_size, return_tokens=True)
            tokens += used
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE public.articles SET embedding = %s WHERE id = %s",
                    list(zip(embeddings, row_ids)),
                )
            conn.commit()
            done += len(row_ids)
            if tqdm:
                pbar.set_postfix_str(f"${tokens * EMBED_COST:.2f} | {tokens/1e6:.2f}M tok")

        print(f"Embedded {done} article(s) "
              f"({tokens/1e6:.2f}M tokens ~ ${tokens * EMBED_COST:.2f}).")
        if build_index and done:
            create_index(conn)
        return done
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed articles with OpenAI text-embedding-3-small into pgvector."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help=f"inputs per embeddings request (default {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N pending rows (for testing)",
    )
    parser.add_argument(
        "--reembed", action="store_true",
        help="recompute embeddings for every row, not just NULL ones",
    )
    parser.add_argument(
        "--no-index", dest="build_index", action="store_false",
        help="skip building the HNSW index afterwards",
    )
    args = parser.parse_args()

    embed_all(
        batch_size=args.batch_size,
        limit=args.limit,
        reembed=args.reembed,
        build_index=args.build_index,
    )


if __name__ == "__main__":
    main()
