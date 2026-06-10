"""Two-pass article classifier as LangChain chains.

Pass 1: gemini-2.5-flash-lite labels every article.
Pass 2: gemini-2.5-flash re-runs the same prompt only when pass 1 said
"real news" — it confirms or overturns. If the confirmation call fails
entirely, the lite verdict stands (same behavior as the legacy script).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from ticker_news.classification.schemas import Classification
from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE, gemini_chat

logger = logging.getLogger(__name__)

GEMINI_TIMEOUT_S = 60.0
MAX_ARTICLE_CHARS = 6_000  # classification needs the lede, not the whole article
RETRIES = 4

PROMPT_TEMPLATE = """You are an equity-research editor sorting financial news articles by type.

Classify the article below into EXACTLY ONE of these categories:

- "conference-PR": announcement of a company sponsoring, speaking at, exhibiting
  at, or being recognized at a conference, trade show, summit, or awards event;
  event promotion or registration notices.
- "marketing fluff": promotional / advertorial / vendor self-promotion, product
  marketing copy, newsletter or subscription promos, "about the company" filler --
  no decision-useful financial substance.
- "real news": first-hand reporting of a concrete, newly-occurring company or
  market event -- earnings results, guidance updates, demand/supply signals, deals
  (M&A, partnerships, contracts), product launches, management changes, regulatory
  or legal events, analyst rating/price-target actions. The article must BE the
  primary source breaking the event. Do NOT use this for articles that merely
  explain or analyse a price move that has already happened.
- "recap/review": opinion, educational, or retrospective content -- this includes
  "why X stock rose/fell/plummeted/skyrocketed today/this week", performance
  explainers written after the fact, "should you buy X?", "3 stocks to buy now",
  "here's what [event] means for investors", market-wrap or listicle articles.
  Any piece whose primary purpose is to explain a move that already happened, rather
  than to report a new event, belongs here even if it contains factual detail.
- "market speculation": forward-looking conjecture -- predictions, "will X hit $Y",
  speculative outlooks, scenario pieces driven mainly by guesswork.
- "legal solicitation": law-firm / class-action / securities-fraud solicitations,
  "investor alert" / "deadline alert" / "shareholder alert" notices inviting
  shareholders to join or lead a lawsuit, attorney advertising, "contact the firm"
  / lead-plaintiff-deadline reminders.
- "regulatory filing": routine corporate or regulatory disclosures and notices --
  e.g. Form 8.3 dealing disclosures, annual-meeting / record-date notices,
  shareholder-meeting resolutions, proxy notices, exchange/registration filings --
  reported as a notice, not as analyzed news.
- "book PR": press releases announcing or promoting a book launch (novels, poetry,
  children's, faith-based, self-help, memoirs), author promotion or book awards.
- "politics/macro": general political, policy, geopolitical, or macroeconomic news
  (elections, tariffs/trade, legislation, war/sanctions, central-bank/fiscal policy)
  that is not specific company financial news.
- "other": anything that genuinely fits none of the above, e.g. obituaries,
  celebrity / human-interest content, navigation or boilerplate.

Pick the single best fit. Return ONLY a JSON object:
{{"category": "<one category>", "reason": "<one short clause justifying the label>"}}

ARTICLE TITLE: {title}

ARTICLE BODY:
\"\"\"
{body}
\"\"\""""


def build_classifier(model_name: str):
    """prompt | structured-output Gemini, with retry on transient failures."""
    llm = gemini_chat(model_name, timeout_s=GEMINI_TIMEOUT_S)
    structured = llm.with_structured_output(Classification).with_retry(
        stop_after_attempt=RETRIES, wait_exponential_jitter=True
    )
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | structured


@lru_cache(maxsize=1)
def _default_lite():
    return build_classifier(GEMINI_FLASH_LITE)


@lru_cache(maxsize=1)
def _default_confirm():
    return build_classifier(GEMINI_FLASH)


def classify_article(
    title: Optional[str],
    content: str,
    *,
    lite=None,
    confirm=None,
    config=None,
) -> Tuple[Classification, bool]:
    """Returns (verdict, confirmation_ran). Two-pass semantics preserved."""
    lite = lite if lite is not None else _default_lite()
    inputs = {
        "title": (title or "").strip()[:300],
        "body": (content or "")[:MAX_ARTICLE_CHARS],
    }
    first = lite.invoke(inputs, config=config)
    if first.category != "real news":
        return first, False

    confirm = confirm if confirm is not None else _default_confirm()
    try:
        return confirm.invoke(inputs, config=config), True
    except Exception as exc:
        # confirmation exhausted its retries: keep the lite "real news" verdict
        logger.warning("confirmation pass failed (%r); keeping lite verdict", exc)
        return first, True
