from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ticker_news.sentiment.analysts import render_summary, render_verdict
from ticker_news.sentiment.graph import build_graph, judge_article
from ticker_news.sentiment.schemas import Verdict

ARTICLE = {
    "ticker": "NVDA",
    "title": "NVDA beats and raises",
    "content": "Data center revenue grew 90%.",
    "published_utc": "2026-06-09T10:00:00Z",
    "precedents": ["2026-05-01 NVDA: prior beat"],
}


def _fake_verdict_llm():
    """A structured-output stub: records the prompt it saw, returns a Verdict."""
    prompts = []
    llm = RunnableLambda(
        lambda p: (prompts.append(p)
                   or Verdict(action="buy", confidence=0.9, reasoning="distinct beat"))
    )
    return llm, prompts


def _fake_summary_llm(summary_text="DISTILLED: NVDA beat, DC rev +90%."):
    """A plain-text stub: records the prompt it saw, returns an AIMessage."""
    prompts = []
    llm = RunnableLambda(lambda p: (prompts.append(p) or AIMessage(content=summary_text)))
    return llm, prompts


def test_single_node_produces_verdict_from_precedents():
    llm, prompts = _fake_verdict_llm()
    graph = build_graph(verdict_llm=llm)
    result = graph.invoke({"article": ARTICLE, "analyses": [], "verdict": None})
    assert result["verdict"].action == "buy"
    # exactly one model call, and it saw the article + the precedent
    assert len(prompts) == 1
    assert "NVDA beats and raises" in prompts[0]
    assert "prior beat" in prompts[0]


def test_analyses_records_the_historical_precedent_rationale():
    llm, _ = _fake_verdict_llm()
    graph = build_graph(verdict_llm=llm)
    result = graph.invoke({"article": ARTICLE, "analyses": [], "verdict": None})
    assert result["analyses"] == [
        {"role": "historical_precedent", "analysis": "distinct beat"}
    ]


def test_judge_article_returns_verdict_and_analyses():
    llm, _ = _fake_verdict_llm()
    graph = build_graph(verdict_llm=llm)
    verdict, analyses = judge_article(ARTICLE, graph=graph)
    assert isinstance(verdict, Verdict)
    assert len(analyses) == 1
    assert analyses[0]["role"] == "historical_precedent"


def test_config_reaches_graph():
    llm, _ = _fake_verdict_llm()
    graph = build_graph(verdict_llm=llm)
    verdict, _ = judge_article(ARTICLE, graph=graph, config={"metadata": {"k": "v"}})
    assert verdict.action == "buy"


def test_verdict_prompt_shows_own_insights_when_present():
    article = {**ARTICLE, "own_insights": ["DC revenue +90%", "guidance raised"]}
    prompt = render_verdict(article)
    assert "THIS ARTICLE'S INSIGHTS" in prompt
    assert "- DC revenue +90%" in prompt
    assert "- guidance raised" in prompt


def test_verdict_prompt_omits_own_insights_when_absent():
    prompt = render_verdict(ARTICLE)  # no own_insights key
    assert "THIS ARTICLE'S INSIGHTS" not in prompt
    assert "SIMILAR PAST ARTICLES" in prompt


def test_verdict_prompt_asks_for_buy_sell_hold():
    prompt = render_verdict(ARTICLE)
    assert "buy / sell / hold" in prompt
    assert "confidence" in prompt


def test_precedents_header_is_mode_aware():
    # default (no precedent_kind) -> the cosine-similarity header
    default = render_verdict(ARTICLE)
    assert "SIMILAR PAST ARTICLES" in default
    assert "cosine-nearest" in default
    # ticker-history -> recency header for the ticker, no false "similar" claim
    recency = render_verdict({**ARTICLE, "precedent_kind": "ticker-recency"})
    assert "RECENT NEWS ON NVDA" in recency
    assert "newest first" in recency
    assert "cosine-nearest" not in recency
    # ticker-relevant -> relevance header for the ticker
    relevance = render_verdict({**ARTICLE, "precedent_kind": "ticker-relevance"})
    assert "RELATED PAST NEWS ON NVDA" in relevance
    assert "most relevant to this story first" in relevance


def test_label_legend_only_when_precedents_are_labelled():
    plain = render_verdict(ARTICLE)
    assert "CERTAINTY / CLOSURE" not in plain
    labelled = render_verdict({**ARTICLE, "precedent_labelled": True})
    assert "CERTAINTY / CLOSURE" in labelled
    assert "evidance-event" in labelled


# --- pre-verdict summarization node ----------------------------------------

LONG_ARTICLE = {
    **ARTICLE,
    "content": (
        "NVDA reported data center revenue up 90%. " + "Safe Harbor boilerplate. " * 40
    ),
}


def test_no_summary_node_when_disabled():
    """summary_llm=None keeps the historical topology: verdict sees the raw body."""
    vllm, vprompts = _fake_verdict_llm()
    sllm, sprompts = _fake_summary_llm()
    graph = build_graph(verdict_llm=vllm)  # no summary_llm
    graph.invoke({"article": LONG_ARTICLE, "analyses": [], "verdict": None})
    # summary stub never invoked; verdict prompt carries the raw (boilerplate) body
    assert sprompts == []
    assert "Safe Harbor boilerplate." in vprompts[0]


def test_summary_node_replaces_body_fed_to_verdict():
    """With a summary_llm, the verdict's ARTICLE BODY is the distilled brief, not
    the raw text — and the summarizer saw the original body."""
    vllm, vprompts = _fake_verdict_llm()
    sllm, sprompts = _fake_summary_llm("DISTILLED: NVDA beat, DC rev +90%.")
    graph = build_graph(verdict_llm=vllm, summary_llm=sllm)
    graph.invoke({"article": LONG_ARTICLE, "analyses": [], "verdict": None})
    # the summarizer saw the raw body...
    assert len(sprompts) == 1
    assert "Safe Harbor boilerplate." in sprompts[0]
    # ...and the verdict saw the summary instead of the raw boilerplate body
    assert "DISTILLED: NVDA beat" in vprompts[0]
    assert "Safe Harbor boilerplate." not in vprompts[0]


def test_summary_node_passthrough_on_empty_summary():
    """A blank summary must not blank the verdict's body — keep the original."""
    vllm, vprompts = _fake_verdict_llm()
    sllm, _ = _fake_summary_llm("   ")  # whitespace-only -> treated as no summary
    graph = build_graph(verdict_llm=vllm, summary_llm=sllm)
    graph.invoke({"article": LONG_ARTICLE, "analyses": [], "verdict": None})
    assert "data center revenue up 90%" in vprompts[0]


def test_summary_prompt_carries_article_and_strips_instructions():
    prompt = render_summary(LONG_ARTICLE)
    assert "TICKER: NVDA" in prompt
    assert "DISCARD boilerplate" in prompt
    assert "120 words maximum" in prompt
    assert "data center revenue up 90%" in prompt
