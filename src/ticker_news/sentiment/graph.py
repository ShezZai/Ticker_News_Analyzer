"""Sentiment orchestration: Send fan-out to fixed-role analysts, then a
structured-output synthesis. No supervisor — roles are static, always all run.

Nodes are sync on purpose: the stage runs inside asyncio.to_thread, and
LangGraph executes a superstep's parallel Send tasks on its background
executor, so the three analysts still run concurrently.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ticker_news.sentiment.analysts import (
    ANALYST_PROMPTS,
    render_analyst,
    render_synthesis,
)
from ticker_news.sentiment.schemas import Verdict
from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE, gemini_chat

GEMINI_TIMEOUT_S = 90.0
RETRIES = 3


class SentimentState(TypedDict):
    article: dict
    analyses: Annotated[list[dict], operator.add]
    verdict: Optional[Verdict]


def _default_analyst_llm():
    return gemini_chat(GEMINI_FLASH_LITE, timeout_s=GEMINI_TIMEOUT_S).with_retry(
        stop_after_attempt=RETRIES, wait_exponential_jitter=True
    )


def _default_judge():
    return (
        gemini_chat(GEMINI_FLASH, timeout_s=GEMINI_TIMEOUT_S)
        .with_structured_output(Verdict)
        .with_retry(stop_after_attempt=RETRIES, wait_exponential_jitter=True)
    )


def build_graph(*, analyst_llm=None, judge=None):
    analyst_llm = analyst_llm if analyst_llm is not None else _default_analyst_llm()
    judge = judge if judge is not None else _default_judge()

    def fan_out(state: SentimentState):
        return [
            Send("analyst", {"article": state["article"], "role": role})
            for role in ANALYST_PROMPTS
        ]

    def analyst(payload: dict) -> dict:
        prompt = render_analyst(payload["role"], payload["article"])
        message = analyst_llm.invoke(prompt, config={"run_name": f"analyst:{payload['role']}"})
        text = getattr(message, "content", None) or str(message)
        return {"analyses": [{"role": payload["role"], "analysis": text}]}

    def synthesize(state: SentimentState) -> dict:
        prompt = render_synthesis(state["article"], state["analyses"])
        return {"verdict": judge.invoke(prompt, config={"run_name": "synthesize"})}

    g = StateGraph(SentimentState)
    g.add_node("analyst", analyst)
    g.add_node("synthesize", synthesize)
    g.add_conditional_edges(START, fan_out, ["analyst"])
    g.add_edge("analyst", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


@lru_cache(maxsize=1)
def _default_graph():
    return build_graph()


def judge_article(article: dict, *, graph=None, config=None) -> tuple[Verdict, list[dict]]:
    """Run the full analyst panel + synthesis for one article/ticker."""
    graph = graph if graph is not None else _default_graph()
    result = graph.invoke({"article": article, "analyses": [], "verdict": None}, config=config)
    return result["verdict"], result["analyses"]
