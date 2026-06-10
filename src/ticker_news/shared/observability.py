"""Langfuse Cloud wiring. Every helper degrades to a no-op when keys are absent.

One trace per article; stage spans; LLM generations nest via the LangChain
CallbackHandler. Stable observation names are an eval contract:
process-article, scrape, embed, classify, tag, insights, sentiment,
analyst:<role>, synthesize.

SDK notes (langfuse >=4,<5):
- create_trace_id is a @staticmethod on Langfuse; callable on instance or class.
- Spans expose .update(input=...) — there is no .update_trace() on span objects.
- propagate_attributes is a standalone function from `langfuse`, not a client method.
- Langfuse() construction does NOT make network calls; it is safe to construct with
  fake keys in tests.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache

from ticker_news.shared.config import get_settings

logger = logging.getLogger(__name__)


def enabled() -> bool:
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


@lru_cache(maxsize=1)
def client():
    """Singleton Langfuse client, or None when disabled."""
    if not enabled():
        return None
    from langfuse import Langfuse

    s = get_settings()
    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )


def chain_config(run_name: str | None = None) -> dict:
    """Per-invoke config for chains/graphs: Langfuse callbacks + optional run_name.

    Returns {} (or just the run_name) when disabled, so call sites can pass it
    unconditionally: chain.invoke(x, config=chain_config() or None).
    """
    cfg: dict = {}
    if enabled():
        from langfuse.langchain import CallbackHandler

        cfg["callbacks"] = [CallbackHandler()]
    if run_name:
        cfg["run_name"] = run_name
    return cfg


@contextmanager
def article_trace(url: str, *, ticker: str | None = None):
    """Root trace for one article moving through the pipeline.

    Deterministic trace id seeded from the URL so re-runs correlate.
    Yields the root span (or None when disabled).

    Adaptation: the plan sketch calls root.update_trace(input=...) but in the
    installed SDK (v4.7.1) there is no update_trace() on span objects. The correct
    call is root.update(input=...) which sets the span's input attribute.
    """
    c = client()
    if c is None:
        yield None
        return
    from langfuse import Langfuse, propagate_attributes

    trace_id = Langfuse.create_trace_id(seed=url)
    with c.start_as_current_observation(
        as_type="span",
        name="process-article",
        trace_context={"trace_id": trace_id},
    ) as root:
        root.update(input={"url": url})
        metadata = {"url": url[:200]}
        if ticker:
            metadata["ticker"] = ticker
        with propagate_attributes(tags=["pipeline-v2"], metadata=metadata):
            yield root


@contextmanager
def stage_span(name: str):
    c = client()
    if c is None:
        yield None
        return
    with c.start_as_current_observation(as_type="span", name=name) as span:
        yield span


def flush() -> None:
    c = client()
    if c is not None:
        c.flush()
