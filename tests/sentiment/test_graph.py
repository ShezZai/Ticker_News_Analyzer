from langchain_core.runnables import RunnableLambda

from ticker_news.sentiment.analysts import render_verdict
from ticker_news.sentiment.graph import build_graph, judge_article
from ticker_news.sentiment.schemas import Verdict

ARTICLE = {
    "ticker": "NVDA",
    "title": "NVDA beats and raises",
    "content": "Data center revenue grew 90%.",
    "published_utc": "2026-06-09T10:00:00Z",
    "provider_sentiment": "positive",
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


def test_label_legend_only_when_precedents_are_labelled():
    plain = render_verdict(ARTICLE)
    assert "CERTAINTY / CLOSURE" not in plain
    labelled = render_verdict({**ARTICLE, "precedent_labelled": True})
    assert "CERTAINTY / CLOSURE" in labelled
    assert "evidance-event" in labelled
