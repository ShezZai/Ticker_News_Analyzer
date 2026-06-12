"""Single factory for every LLM client in the pipeline.

All chat models and embedding models are built here so retries, rate limits,
and (in a later phase) Langfuse instrumentation live in exactly one place.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.rate_limiters import InMemoryRateLimiter

from ticker_news.shared.config import get_settings

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import OpenAIEmbeddings

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_FLASH = "gemini-2.5-flash"

# One limiter shared by every Gemini model instance: concurrent stages must not
# blow the per-project quota between them.
_GEMINI_RPS = 8.0


@lru_cache(maxsize=1)
def gemini_rate_limiter() -> InMemoryRateLimiter:
    return InMemoryRateLimiter(
        requests_per_second=_GEMINI_RPS, max_bucket_size=_GEMINI_RPS
    )


def gemini_chat(model: str, *, timeout_s: float = 60.0) -> ChatGoogleGenerativeAI:
    """A deterministic (temperature 0) Gemini chat model with shared rate limit.

    google-genai has no default request timeout — without one a single stuck
    call freezes a whole worker pool, so timeout_s is mandatory here.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    if not s.google_api_key:
        raise SystemExit("GOOGLE_API_KEY is not set (put it in .env).")
    return ChatGoogleGenerativeAI(
        model=model,
        # Runnable name: generation spans read "gemini-2.5-flash-lite" instead
        # of the class name. Invoke-time run_name (analyst:<role>, synthesize)
        # still takes precedence, so contract observation names are unaffected.
        name=model,
        temperature=0.0,
        timeout=timeout_s,
        google_api_key=s.google_api_key,
        rate_limiter=gemini_rate_limiter(),
    )


def openai_embeddings(*, batch_size: int = 256) -> OpenAIEmbeddings:
    """text-embedding-3-small client. Inputs must be pre-truncated to the model
    window by the caller (ticker_news.embedding); LangChain's own
    chunk-and-average path for long inputs is disabled because it changes
    vector semantics versus plain truncation."""
    from langchain_openai import OpenAIEmbeddings

    s = get_settings()
    if not s.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in .env).")
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=s.openai_api_key,
        chunk_size=batch_size,
        check_embedding_ctx_length=False,
    )
