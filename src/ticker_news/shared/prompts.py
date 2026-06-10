"""Prompt management: Langfuse-versioned with committed in-repo fallbacks.

The fallback IS the source of truth in the repo; Langfuse holds versioned,
labeled copies for A/B and eval runs. The service boots fine with Langfuse
down or disabled.

NOTE: chains that call get_prompt() are lru_cached — prompt updates in
Langfuse require a process restart before the new text takes effect in those
chains. Renderers that call get_prompt() directly (e.g. analysts.render_analyst,
analysts.render_synthesis) do fetch per call, but Langfuse's client-side prompt
cache (TTL-based, default ~60s) absorbs the overhead.
"""

from __future__ import annotations

import logging

from ticker_news.shared.observability import client

logger = logging.getLogger(__name__)

PROMPT_LABEL = "production"


def safe_format(template: str, fallback: str, **kwargs) -> str:
    """Format a (possibly Langfuse-edited) template; fall back to the in-repo
    template when the remote copy has unknown/typo'd placeholders, instead of
    crashing every article until a restart."""
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("prompt template format failed (%r); using in-repo fallback", exc)
        return fallback.format(**kwargs)


def get_prompt(name: str, fallback: str) -> str:
    """Return Langfuse prompt text (label=production) or the in-repo fallback.

    When Langfuse is disabled (no keys) or the prompt is unavailable (e.g.
    not yet pushed, network error), returns the fallback unchanged so the
    service stays fully operational.
    """
    c = client()
    if c is None:
        return fallback
    try:
        return c.get_prompt(name, label=PROMPT_LABEL).prompt
    except Exception as exc:
        logger.warning("langfuse prompt %r unavailable (%r); using fallback", name, exc)
        return fallback


def registry() -> dict[str, str]:
    """name -> in-repo fallback text, for `ticker-news prompts push`."""
    from ticker_news.classification.chain import PROMPT_TEMPLATE as classify_prompt
    from ticker_news.enrichment.insights import PROMPT_TEMPLATE as insights_prompt
    from ticker_news.sentiment.analysts import ANALYST_PROMPTS, SYNTHESIS_PROMPT

    reg = {
        "classify-article": classify_prompt,
        "extract-insights": insights_prompt,
        "synthesize-verdict": SYNTHESIS_PROMPT,
    }
    for role, prompt in ANALYST_PROMPTS.items():
        reg[f"analyst-{role}"] = prompt
    return reg


def push_all() -> int:
    """Upsert every registry prompt to Langfuse with the production label.

    Raises SystemExit if Langfuse keys are not configured.
    """
    c = client()
    if c is None:
        raise SystemExit("Langfuse keys not configured (LANGFUSE_PUBLIC_KEY/SECRET_KEY).")
    count = 0
    for name, text in registry().items():
        c.create_prompt(name=name, prompt=text, labels=[PROMPT_LABEL], type="text")
        count += 1
    return count
