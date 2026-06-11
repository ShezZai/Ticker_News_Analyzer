"""Experimental classification prompt variants: binary and fine-grained.

Candidate replacements for the production classifier, evaluated by
`ticker-news eval classify` against the hand-labeled ground-truth set
(spec: docs/superpowers/specs/2026-06-12-classify-eval-design.md).
Not wired into the pipeline; promoting a winner stays a human decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple, get_args

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

logger = logging.getLogger(__name__)

GEMINI_TIMEOUT_S = 60.0
RETRIES = 4

MODES = ("lite", "flash", "two-pass")
VARIANTS = ("binary", "finegrained")

# ---------------------------------------------------------------------------
# Schemas, category constants, helpers
# ---------------------------------------------------------------------------

BinaryLabel = Literal["real news", "none news"]
BINARY_LABELS: list[str] = list(get_args(BinaryLabel))


class BinaryClassification(BaseModel):
    """Structured verdict of the binary ACT / DON'T-ACT classifier."""

    label: BinaryLabel
    confidence: Optional[float] = None
    reason: str = ""


FinegrainedCategory = Literal[
    # --- a real, newly-occurring event is being reported (NEWS) ---
    "earnings-reporting",
    "dividend-reporting",
    "merger/investment/funding",
    "legal-event",
    "MACRO-investment",
    "news-event",
    "news-report",
    # --- not a new event (NOT-NEWS) ---
    "recap/review",
    "market speculation",
    "MACRO-political",
    "legal-call",
    "conference-PR",
    "marketing fluff",
    "book PR",
    "Other-filing-reporting",
    "other",
]
FINEGRAINED_CATEGORIES: list[str] = list(get_args(FinegrainedCategory))

# Fine-grained categories that count as ACT/YES for the binary ground truth.
NEWS_SUBTYPES: frozenset[str] = frozenset({
    "earnings-reporting",
    "dividend-reporting",
    "merger/investment/funding",
    "legal-event",
    "MACRO-investment",
    "news-event",
    "news-report",
})


class FinegrainedClassification(BaseModel):
    """Structured verdict of the fine-grained taxonomy classifier."""

    category: FinegrainedCategory
    reason: str = ""


def is_act_binary(label: str) -> bool:
    # Unknown labels are treated as NOT-ACT (don't trade on what we can't parse).
    return label == "real news"


def is_act_finegrained(category: str) -> bool:
    return category in NEWS_SUBTYPES

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BINARY_PROMPT_TEMPLATE = """You are an equity-research editor deciding whether an article is genuine market NEWS or not.

Label "real news" ONLY if the article is FIRST-HAND reporting of a CONCRETE, NEWLY-OCCURRING, market-relevant EVENT -- something that just happened that a trader could act on. Qualifying events include:
- earnings / results / guidance changes; a dividend declared or changed;
- M&A, a strategic equity investment / stake, or a funding round;
- a product / service launch, a partnership / contract / agreement, a leadership or
  board change, an expansion or new entity, a listing;
- a supply-chain disruption, a major customer win or loss, a regulatory action, or
  another operational development that materially changes the business;
- a lawsuit / class action actually FILED, an investigation opened, a ruling / settlement;
- a GOVERNMENT equity / investment action (taking a stake, a national investment program);
- a bankruptcy, restructuring, or insolvency event; a major contract win or loss; or a
  significant technological breakthrough.
The article must BE the primary report of the event.

Label "none news" for everything else, including:
- opinion / recap -- "why X moved", "should you buy X?", predictions, listicles, market-wraps;
- PR / marketing / advertorials / product copy / market-research-report promos / book PR;
- conference / trade-show / awards announcements;
- law-firm SOLICITATIONS -- "investor alert", "deadline alert", "opportunity to lead",
  "contact the firm", lead-plaintiff reminders;
- general politics / macro / policy commentary -- tariffs, Fed / rates, elections,
  inflation or jobs data, broad market direction;
- routine filings reported as notices -- share buybacks, insider / managers' transactions,
  annual reports / proxy notices;
- rehashed summaries of already-known events, pure speculation / commentary with no new
  facts, generic market chatter, or content whose primary purpose is engagement not info;
- lifestyle / entertainment / human-interest / obituaries / charity / university events /
  boilerplate / anything else.

Two quick tests for "real news":
(1) Did a CONCRETE event just happen (not a prediction, not a recap of an old move)?
(2) Is THIS article the PRIMARY source breaking it (not a law firm's pitch, not an
    opinion column, not a marketing release)?
Both YES -> "real news". Otherwise -> "none news".
Borderline guidance: a law-firm "lawsuit filed" report and a government taking a company
stake ARE real news; a law-firm deadline/solicitation and a tariff/Fed commentary are NOT.

Judge by MATERIALITY, not sentiment or tone. Positive, negative, and neutral developments
can ALL be "real news" if the event could plausibly affect a company's revenue,
profitability, growth prospects, competitive position, market share, regulatory /
operational risk, strategic direction, or investor perception. Conversely, a highly
emotional or dramatic article that carries NO concrete new information is "none news".
Ignore sentiment, and ignore WHICH ticker or company is involved -- judge only whether the
article carries genuinely NEW, investment-relevant factual information. New facts outweigh
opinion; materiality outweighs popularity or drama. When genuinely uncertain, prefer
"none news".

Return ONLY a JSON object:
{{"label": "<real news | none news>", "confidence": 0.0-1.0, "reason": "<one short clause>"}}

ARTICLE TITLE: {title}

ARTICLE BODY:
\"\"\"
{body}
\"\"\""""


FINEGRAINED_PROMPT_TEMPLATE = """You are an equity-research editor sorting financial news articles into a fine-grained taxonomy.

First decide: is the article FIRST-HAND reporting of a CONCRETE, NEWLY-OCCURRING, market-relevant EVENT (a thing that just happened -- results, a deal, a filing, a launch, a government action)? Or is it promotion / opinion / solicitation / a recap of an already-happened move?

Then assign EXACTLY ONE label.

=== A real new event is being reported (NEWS) ===
- "earnings-reporting": first-hand reporting of a company's earnings / financial
  results, revenue, a guidance change, or an earnings-call transcript.
  (e.g. "X reports Q4 results", "X beats on EPS", "<co> earnings call transcript".)
- "dividend-reporting": a company DECLARING or CHANGING a cash dividend, or a fund
  announcing distributions. (e.g. "declares quarterly dividend", "raises dividend 10%".)
  NOT share buybacks -> "Other-filing-reporting".
- "merger/investment/funding": an M&A acquisition/merger, a strategic equity
  investment or stake, or a funding round. (e.g. "X acquires Y", "secures investment
  from Z".) A purely commercial supply/partnership deal with NO equity -> "news-event".
- "legal-event": an actual legal action that OCCURRED -- a lawsuit / class action
  FILED, an investigation opened, a ruling or settlement. The hook is the filing itself.
- "MACRO-investment": a GOVERNMENT acting on equity / investment -- the government
  taking or weighing a stake in a company, or a government-driven investment
  program/initiative. (e.g. "White House eyes Intel stake", "Trump announces $500B AI
  investment".)
- "news-event": any OTHER first-hand corporate event announced by the parties --
  product/service launch, partnership / contract / agreement, leadership or board
  appointment, expansion / new entity, listing, capacity / pricing change. The article
  IS the announcement (typically a company press release).
- "news-report": SECONDARY journalism about a development -- sourced from "a report" /
  Bloomberg / Fortune, rumors, layoffs / restructuring plans, multi-item deal roundups,
  or reporting on a third-party study. Reportage ABOUT events, not a primary release.

=== Not a new event (NOT-NEWS) ===
- "recap/review": opinion / educational / retrospective -- "why X rose/fell today",
  "should you buy X?", "3 stocks to buy now", "what [event] means for investors",
  performance explainers, market-wraps, listicles. Explaining an already-happened move.
- "market speculation": forward-looking conjecture -- predictions, "will X hit $Y",
  price-target / scenario guesswork.
- "MACRO-political": general political / policy / geopolitical / macro news NOT tied to
  a government equity action -- tariffs, Fed / rate decisions, elections, legislation,
  war / sanctions, inflation / jobs data, broad market commentary.
- "legal-call": a law-firm SOLICITATION -- "investor / deadline / shareholder alert",
  "opportunity to lead", "secure counsel before deadline", "contact the firm",
  lead-plaintiff reminders. A call to action, not a new filing.
- "conference-PR": a company sponsoring / speaking / exhibiting / being recognized at a
  conference, trade show, summit, or AWARDS event; event promotion / registration.
- "marketing fluff": promotional / advertorial / vendor self-promotion, product
  marketing copy, market-research-report promos, newsletter / subscription promos,
  "about the company" filler -- no decision-useful substance.
- "book PR": a press release announcing / promoting a book (novel, poetry, children's,
  memoir, self-help) or an author / book award.
- "Other-filing-reporting": a routine corporate / exchange filing reported as a NOTICE
  -- share buybacks / repurchases, managers' / insider transactions, annual reports /
  Form 20-F, proxy / meeting notices. (A dividend goes to "dividend-reporting".)
- "other": genuinely none of the above -- obituaries, charity, university
  commencements, human-interest, retraction notices, boilerplate.

If an article fits more than one NEWS subtype, prefer in this order:
earnings-reporting > dividend-reporting > merger/investment/funding > MACRO-investment >
legal-event > (news-event if it is a primary press release, else news-report).

Pick the single best fit. Return ONLY a JSON object:
{{"category": "<one category>", "reason": "<one short clause justifying the label>"}}

ARTICLE TITLE: {title}

ARTICLE BODY:
\"\"\"
{body}
\"\"\""""

# ---------------------------------------------------------------------------
# Chain builders
# ---------------------------------------------------------------------------


def _build(model_name: str, prompt_name: str, fallback: str, schema: type):
    """prompt | structured-output Gemini, with the same Langfuse-template
    guard as the production classifier. NOT cached — the eval rebuilds per
    run so Langfuse prompt edits apply without a process restart."""
    from ticker_news.shared.llm import gemini_chat
    from ticker_news.shared.prompts import get_prompt

    llm = gemini_chat(model_name, timeout_s=GEMINI_TIMEOUT_S)
    structured = llm.with_structured_output(schema).with_retry(
        stop_after_attempt=RETRIES, wait_exponential_jitter=True
    )
    template = get_prompt(prompt_name, fallback)
    try:
        prompt = ChatPromptTemplate.from_template(template)
        if set(prompt.input_variables) != {"title", "body"}:
            raise ValueError(f"unexpected variables: {prompt.input_variables}")
    except Exception as exc:
        logger.warning("%s prompt invalid (%r); using in-repo fallback", prompt_name, exc)
        prompt = ChatPromptTemplate.from_template(fallback)
    return prompt | structured


def build_binary_classifier(model_name: str):
    return _build(model_name, "classify-binary",
                  BINARY_PROMPT_TEMPLATE, BinaryClassification)


def build_finegrained_classifier(model_name: str):
    return _build(model_name, "classify-finegrained",
                  FINEGRAINED_PROMPT_TEMPLATE, FinegrainedClassification)

# ---------------------------------------------------------------------------
# VariantRunner
# ---------------------------------------------------------------------------


@dataclass
class VariantRunner:
    """One variant/mode pairing. Two-pass mirrors production semantics:
    `lite` labels everything, `confirm` re-runs only verdicts `is_act` calls
    ACT; a failed confirmation keeps the lite verdict. Single-pass modes set
    exactly one chain."""

    lite: Optional[Any]
    confirm: Optional[Any]
    is_act: Callable[[str], bool]
    label_of: Callable[[Any], str]

    async def classify(self, title: Optional[str], body: str,
                       config=None) -> Tuple[Any, bool]:
        from ticker_news.classification.chain import MAX_ARTICLE_CHARS

        inputs = {
            "title": (title or "").strip()[:300],
            "body": (body or "")[:MAX_ARTICLE_CHARS],
        }
        first_chain = self.lite if self.lite is not None else self.confirm
        first = await first_chain.ainvoke(inputs, config=config)
        two_pass = self.lite is not None and self.confirm is not None
        if not two_pass or not self.is_act(self.label_of(first)):
            return first, False
        try:
            return await self.confirm.ainvoke(inputs, config=config), True
        except Exception as exc:
            logger.warning("confirmation pass failed (%r); keeping lite verdict", exc)
            return first, True

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_runner(variant: str, mode: str) -> VariantRunner:
    """Build the chains for a variant/mode pair (fresh — no lru_cache)."""
    from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of {VARIANTS})")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")
    if variant == "binary":
        build, is_act, label_of = (
            build_binary_classifier, is_act_binary, lambda v: v.label)
    else:
        build, is_act, label_of = (
            build_finegrained_classifier, is_act_finegrained, lambda v: v.category)
    lite = build(GEMINI_FLASH_LITE) if mode in ("lite", "two-pass") else None
    confirm = build(GEMINI_FLASH) if mode in ("flash", "two-pass") else None
    return VariantRunner(lite=lite, confirm=confirm, is_act=is_act, label_of=label_of)


def models_for_mode(mode: str) -> dict[str, str]:
    """Model names per mode, for experiment-run metadata."""
    from ticker_news.shared.llm import GEMINI_FLASH, GEMINI_FLASH_LITE

    if mode == "lite":
        return {"lite": GEMINI_FLASH_LITE}
    if mode == "flash":
        return {"flash": GEMINI_FLASH}
    return {"lite": GEMINI_FLASH_LITE, "confirm": GEMINI_FLASH}
