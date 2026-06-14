"""Sentiment orchestration: a single historical-precedent verdict node, with an
optional cheap pre-verdict summarization node.

Simplified from the old three-analyst fan-out + synthesis judge: one
structured-output call weighs the historical precedents and emits the
buy/sell/hold Verdict directly. The model is configurable
(SENTIMENT_VERDICT_MODEL, default gemini-2.5-flash-lite).

When a summary model is configured (SENTIMENT_SUMMARY_MODEL, default off) a
`summarize` node runs first: a cheap model distills the raw article body into a
decision-relevant brief that REPLACES the body fed to the verdict, stripping the
boilerplate (legal disclaimers, "About" sections, market-size filler) that
otherwise dilutes the verdict prompt.

The nodes are sync on purpose: the stage runs inside asyncio.to_thread.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ticker_news.sentiment.analysts import render_summary, render_verdict
from ticker_news.sentiment.schemas import Verdict
from ticker_news.shared.config import get_settings
from ticker_news.shared.llm import gemini_chat

GEMINI_TIMEOUT_S = 90.0
RETRIES = 3


class SentimentState(TypedDict):
    article: dict
    analyses: Annotated[list[dict], operator.add]
    verdict: Optional[Verdict]


def _verdict_llm(model: str):
    """Structured-output Gemini for the verdict, using the given model id."""
    return (
        gemini_chat(model, timeout_s=GEMINI_TIMEOUT_S)
        .with_structured_output(Verdict)
        .with_retry(stop_after_attempt=RETRIES, wait_exponential_jitter=True)
    )


def _summary_llm(model: str):
    """Plain-text Gemini for the pre-verdict article summary (no structured output)."""
    return (
        gemini_chat(model, timeout_s=GEMINI_TIMEOUT_S)
        .with_retry(stop_after_attempt=RETRIES, wait_exponential_jitter=True)
    )


def _default_verdict_llm():
    """Structured-output Gemini for the verdict, model from settings."""
    return _verdict_llm(get_settings().sentiment_verdict_model)


def build_graph(*, verdict_llm=None, summary_llm=None):
    """Compile the sentiment graph. With `summary_llm`, a `summarize` node runs
    before the verdict and rewrites the article body to a distilled brief; without
    it the verdict sees the raw body (the historical default, topology unchanged).
    """
    verdict_llm = verdict_llm if verdict_llm is not None else _default_verdict_llm()

    def summarize_node(state: SentimentState) -> dict:
        article = state["article"]
        content = (article.get("content") or "").strip()
        if not content:
            return {}
        prompt = render_summary(article)
        # run_name "summarize-article": the stable name for the cheap pre-verdict
        # distillation generation (see CLAUDE.md observation contract).
        msg = summary_llm.invoke(prompt, config={"run_name": "summarize-article"})
        summary = (getattr(msg, "content", None) or "").strip()
        if not summary:
            return {}
        # The verdict's ARTICLE BODY becomes the distilled brief; the raw body is
        # kept under content_full for provenance (never shown to the verdict).
        return {"article": {**article, "content": summary, "content_full": content}}

    def verdict_node(state: SentimentState) -> dict:
        prompt = render_verdict(state["article"])
        # run_name kept as "synthesize" - the stable verdict-producing
        # observation name in the eval contract.
        verdict = verdict_llm.invoke(prompt, config={"run_name": "synthesize"})
        # Preserve the (verdict, analyses) contract: the single historical-
        # precedent rationale is recorded as the lone analysis for provenance.
        return {
            "verdict": verdict,
            "analyses": [
                {"role": "historical_precedent", "analysis": verdict.reasoning or ""}
            ],
        }

    g = StateGraph(SentimentState)
    g.add_node("verdict", verdict_node)
    if summary_llm is not None:
        g.add_node("summarize", summarize_node)
        g.add_edge(START, "summarize")
        g.add_edge("summarize", "verdict")
    else:
        g.add_edge(START, "verdict")
    g.add_edge("verdict", END)
    return g.compile()


@lru_cache(maxsize=16)
def _graph_for_models(verdict_model: str | None, summary_model: str | None):
    """A sentiment graph bound to specific model ids (cached per pair). `None` on
    the verdict slot means 'use the configured default'; `None` on the summary
    slot means 'no summarization pass'."""
    return build_graph(
        verdict_llm=_verdict_llm(verdict_model) if verdict_model else None,
        summary_llm=_summary_llm(summary_model) if summary_model else None,
    )


def judge_article(
    article: dict, *, graph=None, config=None,
    model: str | None = None, summary_model: str | None = None,
) -> tuple[Verdict, list[dict]]:
    """Run the historical-precedent verdict for one article/ticker.

    `model` overrides the configured verdict model. `summary_model` enables the
    cheap pre-verdict summarization pass with the given model id; when it is None
    the configured `sentiment_summary_model` applies (itself defaulting to off)."""
    if graph is None:
        if summary_model is None:
            summary_model = get_settings().sentiment_summary_model
        graph = _graph_for_models(model, summary_model)
    result = graph.invoke({"article": article, "analyses": [], "verdict": None}, config=config)
    return result["verdict"], result["analyses"]
