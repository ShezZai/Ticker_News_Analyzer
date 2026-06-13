from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ticker_news.sentiment.analysts import ANALYST_PROMPTS, render_analyst
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


def _fakes():
    prompts = []
    analyst = RunnableLambda(
        lambda p: (prompts.append(p) or AIMessage(content=f"take #{len(prompts)}"))
    )
    judge_prompts = []
    judge = RunnableLambda(
        lambda p: (judge_prompts.append(p)
                   or Verdict(action="buy", confidence=0.9, reasoning="beat"))
    )
    return analyst, judge, prompts, judge_prompts


def test_fans_out_to_every_role_and_synthesizes():
    analyst, judge, prompts, judge_prompts = _fakes()
    graph = build_graph(analyst_llm=analyst, judge=judge)
    result = graph.invoke({"article": ARTICLE, "analyses": [], "verdict": None})
    assert len(result["analyses"]) == len(ANALYST_PROMPTS)
    assert {a["role"] for a in result["analyses"]} == set(ANALYST_PROMPTS)
    assert result["verdict"].action == "buy"
    # every analyst saw the article; the judge saw every analysis
    assert all("NVDA beats and raises" in p for p in prompts)
    assert len(judge_prompts) == 1
    for a in result["analyses"]:
        assert a["analysis"] in judge_prompts[0]


def test_precedents_reach_only_the_historical_analyst():
    analyst, judge, prompts, _ = _fakes()
    graph = build_graph(analyst_llm=analyst, judge=judge)
    graph.invoke({"article": ARTICLE, "analyses": [], "verdict": None})
    with_precedent = [p for p in prompts if "prior beat" in p]
    assert len(with_precedent) == 1


def test_historical_prompt_shows_own_insights_when_present():
    article = {**ARTICLE, "own_insights": ["DC revenue +90%", "guidance raised"]}
    prompt = render_analyst("historical_precedent", article)
    assert "THIS ARTICLE'S INSIGHTS" in prompt
    assert "- DC revenue +90%" in prompt
    assert "- guidance raised" in prompt
    # other roles never get the section
    assert "THIS ARTICLE'S INSIGHTS" not in render_analyst("fundamentals", article)


def test_historical_prompt_omits_own_insights_when_absent():
    prompt = render_analyst("historical_precedent", ARTICLE)  # no own_insights key
    assert "THIS ARTICLE'S INSIGHTS" not in prompt
    assert "SIMILAR PAST ARTICLES" in prompt


def test_label_legend_only_when_precedents_are_labelled():
    # unlabelled (article/insights) modes: no [label] legend
    plain = render_analyst("historical_precedent", ARTICLE)
    assert "CERTAINTY / CLOSURE" not in plain
    # labelled (distilled) modes: legend present
    labelled = render_analyst(
        "historical_precedent", {**ARTICLE, "precedent_labelled": True}
    )
    assert "CERTAINTY / CLOSURE" in labelled
    assert "evidance-event" in labelled


def test_judge_article_returns_verdict_and_analyses():
    analyst, judge, _, _ = _fakes()
    graph = build_graph(analyst_llm=analyst, judge=judge)
    verdict, analyses = judge_article(ARTICLE, graph=graph)
    assert isinstance(verdict, Verdict)
    assert len(analyses) == len(ANALYST_PROMPTS)


def test_config_reaches_graph_and_run_names_set():
    analyst, judge, prompts, _ = _fakes()
    graph = build_graph(analyst_llm=analyst, judge=judge)
    verdict, _ = judge_article(ARTICLE, graph=graph, config={"metadata": {"k": "v"}})
    assert verdict.action == "buy"
