"""Fixed-role analyst registry for the sentiment graph.

Each analyst examines one aspect of a news article for a specific ticker at
its publication moment. Roles are intentionally diverse — the synthesizer
weighs their (possibly conflicting) takes into one verdict.
"""

MAX_CONTENT_CHARS = 24_000

ARTICLE_BLOCK = """\
TICKER UNDER ANALYSIS: {ticker}
PUBLISHED (UTC): {published_utc}
HEADLINE: {title}

ARTICLE BODY:
\"\"\"
{content}
\"\"\"

NEWS-PROVIDER SENTIMENT FOR {ticker} (third-party, may be wrong): {provider_sentiment}
"""

ANALYST_PROMPTS: dict[str, str] = {
    "fundamentals": ARTICLE_BLOCK + """
You are an equity fundamentals analyst. Assess ONLY the impact of this news on
{ticker}'s business fundamentals: revenue, margins, guidance, demand/supply,
competitive position, capital structure. Quantify magnitude where the article
allows it; say explicitly when the article carries no fundamental information.
3-6 sentences, no preamble.""",
    "market_context": ARTICLE_BLOCK + """
You are a market/sector strategist. Assess this news in its market context at
publication time: sector backdrop, what is likely already priced in, how
similar names trade on such news, crowd positioning, and whether the headline
overstates or understates the substance. 3-6 sentences, no preamble.""",
    "historical_precedent": ARTICLE_BLOCK + """
SIMILAR PAST ARTICLES (same corpus, published earlier; cosine-nearest first):
{precedents}

You are a quantitative analyst of historical precedent. Judge how distinctive
this news actually is versus the precedents above: is it a recurring news
pattern for this name/sector or genuinely new information? If the precedent
list is empty or weak, say so. 3-6 sentences, no preamble.""",
}

SYNTHESIS_PROMPT = ARTICLE_BLOCK + """
Three analysts examined this article for {ticker}:

{analyses}

You are the senior portfolio manager. Weigh the analyses (they may conflict),
decide buy / sell / hold for {ticker} at the moment of publication, and give a
confidence in [0,1] proportional to the strength and agreement of the evidence.
Cite which analyst arguments drove your decision in the reasoning.
"""


def render_article(article: dict) -> dict:
    """Common .format kwargs for every prompt."""
    return {
        "ticker": article.get("ticker") or "?",
        "published_utc": str(article.get("published_utc") or "unknown"),
        "title": (article.get("title") or "").strip(),
        "content": (article.get("content") or "")[:MAX_CONTENT_CHARS],
        "provider_sentiment": article.get("provider_sentiment") or "none given",
    }


def render_analyst(role: str, article: dict) -> str:
    kwargs = render_article(article)
    if role == "historical_precedent":
        precedents = article.get("precedents") or []
        kwargs["precedents"] = (
            "\n".join(f"- {p}" for p in precedents) if precedents else "(none found)"
        )
    return ANALYST_PROMPTS[role].format(**kwargs)


def render_synthesis(article: dict, analyses: list[dict]) -> str:
    blocks = "\n\n".join(
        f"[{a['role']}]\n{a['analysis']}" for a in analyses
    )
    return SYNTHESIS_PROMPT.format(analyses=blocks, **render_article(article))
