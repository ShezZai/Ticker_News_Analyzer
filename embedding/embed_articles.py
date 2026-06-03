"""Embed every article with BAAI/bge-m3 and store the vector on each row.

Adds an `embedding vector(1024)` column to ``public.articles`` (pgvector) and
fills it in for every article. Re-running only processes rows whose embedding is
still NULL, so the job is fully resumable / incremental.

The text fed to the model is ``title`` + ``content`` (truncated to the model's
8192-token window). Embeddings are L2-normalized, so cosine similarity equals
inner product — an HNSW cosine index is created at the end for fast ANN search.

Usage:
    python embed_articles.py                  # embed all rows missing an embedding
    python embed_articles.py --batch-size 32  # tune throughput / VRAM
    python embed_articles.py --limit 100      # only the first N (smoke test)
    python embed_articles.py --reembed        # recompute for every row
    python embed_articles.py --no-index       # skip building the HNSW index

    # upgrade only the long articles to bge-m3's full 8192-token window:
    python embed_articles.py --min-tokens 2048 --max-seq-len 8192 --batch-size 2

Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``
(peer auth over the local socket).
"""

from __future__ import annotations

import argparse
import os

# Reduce CUDA fragmentation OOMs on long sequences (must be set before torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import List, Optional, Sequence

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
# bge-m3 supports up to 8192 tokens, but attention memory grows with seq_len^2.
# 2048 tokens (~1500 words) covers the body of virtually every news article and
# keeps a single GPU comfortably within memory; raise with --max-seq-len if needed.
DEFAULT_SEQ_LEN = 2048
DEFAULT_BATCH = 8

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(DB_DSN)
    register_vector(conn)
    return conn


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


def build_text(
    title: Optional[str], content: Optional[str], max_chars: Optional[int]
) -> str:
    """Combine title + content into a single passage for embedding.

    ``max_chars=None`` keeps the full text (used with chunk-pooling, where the
    whole article is split into windows rather than truncated).
    """
    parts = [p.strip() for p in (title, content) if p and p.strip()]
    text = "\n\n".join(parts)
    return text if max_chars is None else text[:max_chars]


def encode_texts(model, texts, max_seq_len, batch_size, chunk_pool):
    """Return one L2-normalized embedding per text.

    Without ``chunk_pool`` each text is embedded once (truncated to max_seq_len).
    With ``chunk_pool`` any text longer than max_seq_len tokens is split into
    consecutive window-sized chunks; the chunk embeddings are mean-pooled and
    re-normalized so the whole article is represented, not just its head.
    """
    import numpy as np

    def _encode(batch):
        return model.encode(
            batch, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )

    if not chunk_pool:
        return list(_encode(texts))

    tokenizer = model.tokenizer
    results: List[object] = [None] * len(texts)
    short_idx, short_texts = [], []
    chunk_texts, chunk_owner = [], []  # flattened chunks across all long texts

    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_seq_len:
            short_idx.append(i)
            short_texts.append(text)
            continue
        for start in range(0, len(token_ids), max_seq_len):
            piece = tokenizer.decode(token_ids[start : start + max_seq_len])
            chunk_texts.append(piece)
            chunk_owner.append(i)

    if short_texts:
        for k, emb in zip(short_idx, _encode(short_texts)):
            results[k] = emb

    if chunk_texts:
        chunk_embs = _encode(chunk_texts)
        groups: dict[int, list] = {}
        for owner, emb in zip(chunk_owner, chunk_embs):
            groups.setdefault(owner, []).append(emb)
        for owner, vecs in groups.items():
            pooled = np.mean(np.vstack(vecs), axis=0)
            norm = np.linalg.norm(pooled)
            results[owner] = pooled / norm if norm else pooled

    return results


def ids_to_process(
    conn: psycopg.Connection,
    reembed: bool,
    limit: Optional[int],
    min_chars: Optional[int] = None,
) -> List[int]:
    """Return the ordered list of article ids that need an embedding.

    ``min_chars`` keeps only rows whose title+content is at least that many
    characters. Since a token is always >= 1 character, this is a cheap, safe
    superset prefilter for a later exact token-count filter.
    """
    clauses = [] if reembed else ["embedding IS NULL"]
    if min_chars:
        clauses.append(
            "char_length(coalesce(title, '') || coalesce(content, '')) >= "
            f"{int(min_chars)}"
        )
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    lim = f"LIMIT {int(limit)}" if limit else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM public.articles {where} ORDER BY id {lim}")
        return [row[0] for row in cur.fetchall()]


def filter_long_ids(
    conn: psycopg.Connection, model, ids: Sequence[int], min_tokens: int, max_chars: int
) -> List[int]:
    """Narrow ``ids`` to articles whose text tokenizes to > min_tokens tokens."""
    tokenizer = model.tokenizer
    long_ids: List[int] = []
    step = 512
    batches = range(0, len(ids), step)
    iterator = tqdm(batches, desc="scanning length", unit="batch") if tqdm else batches
    for start in iterator:
        for rid, title, content in fetch_rows(conn, ids[start : start + step]):
            text = build_text(title, content, max_chars)
            if not text:
                continue
            n_tokens = len(tokenizer.encode(text, add_special_tokens=True))
            if n_tokens > min_tokens:
                long_ids.append(rid)
    return long_ids


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
    max_seq_len: int = DEFAULT_SEQ_LEN,
    limit: Optional[int] = None,
    reembed: bool = False,
    build_index: bool = True,
    min_tokens: Optional[int] = None,
    chunk_pool: bool = False,
) -> int:
    """Embed articles and write vectors back. Returns the number embedded."""
    # Imported lazily so --help and connection errors don't pay the import cost.
    import torch
    from sentence_transformers import SentenceTransformer

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Loading {MODEL_NAME} on {device} (seq_len={max_seq_len}) ...")
    # fp16 on GPU halves activation memory with no meaningful quality loss.
    model_kwargs = {"torch_dtype": torch.float16} if use_cuda else None
    model = SentenceTransformer(MODEL_NAME, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = max_seq_len
    # With chunk-pooling the whole article is split into windows, so keep the
    # full text; otherwise cap at ~6 chars/token to bound tokenizer work.
    max_chars = None if chunk_pool else max_seq_len * 6

    conn = get_conn()
    try:
        ensure_schema(conn)
        # min_tokens targets already-embedded long articles, so it implies reembed.
        ids = ids_to_process(
            conn,
            reembed=reembed or min_tokens is not None,
            limit=limit,
            min_chars=min_tokens,
        )
        if min_tokens is not None:
            ids = filter_long_ids(conn, model, ids, min_tokens, max_chars)
            print(f"{len(ids)} article(s) exceed {min_tokens} tokens.")
        if not ids:
            print("Nothing to embed.")
            return 0

        print(f"Embedding {len(ids)} article(s) in batches of {batch_size} ...")
        batches = range(0, len(ids), batch_size)
        iterator = tqdm(batches, unit="batch") if tqdm else batches

        done = 0
        for start in iterator:
            chunk = ids[start : start + batch_size]
            rows = fetch_rows(conn, chunk)

            payload = []  # (id, embedding) for rows with non-empty text
            texts, row_ids = [], []
            for rid, title, content in rows:
                text = build_text(title, content, max_chars)
                if not text:
                    continue
                texts.append(text)
                row_ids.append(rid)

            if texts:
                embeddings = encode_texts(
                    model, texts, max_seq_len, batch_size, chunk_pool
                )
                payload = list(zip(embeddings, row_ids))

            if payload:
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE public.articles SET embedding = %s WHERE id = %s",
                        payload,
                    )
                conn.commit()
                done += len(payload)

        print(f"Embedded {done} article(s).")
        if build_index and done:
            create_index(conn)
        return done
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed articles with BAAI/bge-m3 into a pgvector column."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help=f"rows per encode/update batch (default {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=DEFAULT_SEQ_LEN,
        help=f"max tokens per article, up to 8192 (default {DEFAULT_SEQ_LEN})",
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
        "--min-tokens", type=int, default=None,
        help="only (re)embed articles longer than this many tokens; pair with "
             "--max-seq-len 8192 to upgrade long articles to the full window",
    )
    parser.add_argument(
        "--chunk-pool", action="store_true",
        help="for articles longer than --max-seq-len, split into window-sized "
             "chunks and mean-pool their embeddings instead of truncating",
    )
    parser.add_argument(
        "--no-index", dest="build_index", action="store_false",
        help="skip building the HNSW index afterwards",
    )
    args = parser.parse_args()

    embed_all(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        limit=args.limit,
        reembed=args.reembed,
        build_index=args.build_index,
        min_tokens=args.min_tokens,
        chunk_pool=args.chunk_pool,
    )


if __name__ == "__main__":
    main()
