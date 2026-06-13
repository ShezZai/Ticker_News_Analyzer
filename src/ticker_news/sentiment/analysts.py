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
{own_insights}SIMILAR PAST ARTICLES (same corpus, published earlier; cosine-nearest first).
Each entry is one prior article; any indented "- " lines beneath it are the
specific insight excerpts from that article that matched this news:
{precedents}

{label_legend}You are a quantitative analyst of historical precedent. Judge how distinctive
this news actually is versus the precedents above: is it a recurring news
pattern for this name/sector or genuinely new information? When this article's
own insights are listed above, treat the indented precedent excerpts as the
concrete points of overlap with them. If the precedent list is empty or weak,
say so. 3-6 sentences, no preamble.""",
}

# Inserted (via {label_legend}) only when the precedent excerpts carry "[label]"
# classification tags - i.e. the distilled precedent-source modes. Omitted for
# the article/insights modes, whose excerpts have no tags, so the analyst is
# never shown a legend for tags it will not see.
LABEL_LEGEND = """\
A leading "[label]" on an insight is its classification tag; the split is about
CERTAINTY / CLOSURE, not topic:
- "evidance-event" = it definitively HAPPENED or was reported - a settled,
  quotable fact (reported earnings/guidance/margins, completed M&A, a signed deal
  with terms, a suit filed, a recall, layoffs, a regulatory ruling, a macro
  release, a disclosed position, a real price move on news). Bullish or bearish,
  but settled.
- "informative" = it was ANNOUNCED but isn't settled yet - real but soft/open/
  unverified (an MoU/LoI/proposed deal with no terms, a product launch, a funding
  round, a planned/future action, a self-reported milestone not independently
  confirmed). In motion, but terms/close/materiality unsettled.
("DROP"-labelled low-signal insights are excluded from the lists above.)
Weight settled evidance-event overlaps more heavily than soft informative ones.

"""

SYNTHESIS_PROMPT = ARTICLE_BLOCK + """
The following analysts examined this article for {ticker}:

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
    """Render an analyst prompt, fetching the current version from Langfuse if
    available (falls back to the in-repo template when Langfuse is disabled or
    unavailable). Fetches per call; the Langfuse client-side TTL cache absorbs
    the overhead.
    """
    from ticker_news.shared.prompts import get_prompt, safe_format

    kwargs = render_article(article)
    if role == "historical_precedent":
        precedents = article.get("precedents") or []
        kwargs["precedents"] = (
            "\n".join(f"- {p}" for p in precedents) if precedents else "(none found)"
        )
        # The target article's own insights render as a self-contained section
        # (header + bullets) when present, or empty so the prompt collapses to
        # just the precedents block in article-similarity mode.
        own = article.get("own_insights") or []
        kwargs["own_insights"] = (
            "THIS ARTICLE'S INSIGHTS (the news under analysis, distilled):\n"
            + "\n".join(f"- {o}" for o in own)
            + "\n\n"
            if own else ""
        )
        # Only show the [label] legend when the excerpts are actually tagged
        # (distilled modes set precedent_labelled); article/insights have no tags.
        kwargs["label_legend"] = (
            LABEL_LEGEND if article.get("precedent_labelled") else ""
        )
    template = get_prompt(f"analyst-{role}", ANALYST_PROMPTS[role])
    return safe_format(template, ANALYST_PROMPTS[role], **kwargs)


def render_synthesis(article: dict, analyses: list[dict]) -> str:
    """Render the synthesis prompt, fetching the current version from Langfuse
    if available (falls back to in-repo template). Fetches per call; the
    Langfuse client-side TTL cache absorbs the overhead.
    """
    from ticker_news.shared.prompts import get_prompt, safe_format

    blocks = "\n\n".join(
        f"[{a['role']}]\n{a['analysis']}" for a in analyses
    )
    template = get_prompt("synthesize-verdict", SYNTHESIS_PROMPT)
    return safe_format(template, SYNTHESIS_PROMPT, analyses=blocks, **render_article(article))
