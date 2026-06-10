"""Text → vector for articles and queries.

Same model and truncation for stored vectors and search queries on purpose —
do not fork the config. (text-embedding-3-small, unit-normalized, 1536 dims.)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ticker_news.shared.llm import EMBED_DIM, EMBED_MODEL, openai_embeddings

MAX_INPUT_TOKENS = 8000  # model hard-caps at 8192; trim just under

_encoder = None


def truncate_tokens(text: str) -> str:
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


def build_text(title: Optional[str], content: Optional[str]) -> str:
    """Combine title + content into a single passage for embedding.

    A generous char pre-cut bounds tokenizer work on pathologically long rows;
    embed_texts() then trims precisely to the token limit.
    """
    parts = [p.strip() for p in (title, content) if p and p.strip()]
    return "\n\n".join(parts)[: MAX_INPUT_TOKENS * 8]


def embed_texts(texts: Sequence[str], *, embeddings=None) -> List[np.ndarray]:
    """One float32 vector per input, index-aligned (empty inputs sent as ' ')."""
    client = embeddings if embeddings is not None else openai_embeddings()
    cleaned = [truncate_tokens(t) for t in texts]
    vectors = client.embed_documents(cleaned)
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def embed_query(text: str, *, embeddings=None) -> np.ndarray:
    """Embedding for a single search query (same code path as stored vectors)."""
    if not text or not text.strip():
        raise ValueError("query text is empty")
    return embed_texts([text], embeddings=embeddings)[0]
