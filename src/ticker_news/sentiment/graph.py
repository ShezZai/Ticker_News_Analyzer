"""Sentiment orchestration: a single historical-precedent verdict node.

Simplified from the old three-analyst fan-out + synthesis judge: one
structured-output call weighs the historical precedents and emits the
buy/sell/hold Verdict directly. The model is configurable
(SENTIMENT_VERDICT_MODEL, default gemini-2.5-flash-lite).

The node is sync on purpose: the stage runs inside asyncio.to_thread.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ticker_news.sentiment.analysts import render_verdict
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


def _default_verdict_llm():
    """Structured-output Gemini for the verdict, model from settings."""
    return _verdict_llm(get_settings().sentiment_verdict_model)


def build_graph(*, verdict_llm=None):
    verdict_llm = verdict_llm if verdict_llm is not None else _default_verdict_llm()

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
    g.add_edge(START, "verdict")
    g.add_edge("verdict", END)
    return g.compile()


@lru_cache(maxsize=1)
def _default_graph():
    return build_graph()


@lru_cache(maxsize=8)
def _graph_for_model(model: str):
    """A verdict graph bound to a specific model id (cached per model)."""
    return build_graph(verdict_llm=_verdict_llm(model))


def judge_article(
    article: dict, *, graph=None, config=None, model: str | None = None
) -> tuple[Verdict, list[dict]]:
    """Run the historical-precedent verdict for one article/ticker.

    `model` overrides the configured verdict model for this call (the default
    graph, read from settings, is used when it is None)."""
    if graph is None:
        graph = _graph_for_model(model) if model else _default_graph()
    result = graph.invoke({"article": article, "analyses": [], "verdict": None}, config=config)
    return result["verdict"], result["analyses"]
