"""Pure text helpers for insight-box parsing and quote verbatimization.

No LLM, no DB. These functions are shared between the extraction pipeline
and the fix-quotes pass, and are independently unit-testable offline.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import List, Optional, Tuple


HEADLINE_PREFIX = "ARTICLE_HEADLINE:"

# Surrounding quote marks the model often adds; stripped before matching.
_WRAP_CHARS = " \t\"\'“”‘’"
DEFAULT_QUOTE_THRESHOLD = 0.75


def with_headline(box_text: str, headline: Optional[str]) -> str:
    """Prefix a box with an ``ARTICLE_HEADLINE:`` line (idempotent)."""
    if box_text.startswith(HEADLINE_PREFIX):
        return box_text  # already prefixed
    head = (headline or "").strip()
    return f"{HEADLINE_PREFIX} {head}\n{box_text}"


def split_box(box_text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Parse a box string into (topic, insight, quotes[]).

    A leading ``ARTICLE_HEADLINE:`` line (if present) is ignored here; parsing
    starts at the TOPIC/INSIGHT/QUOTES labels.
    """
    topic = insight = None
    quotes: List[str] = []
    section = None
    for line in box_text.splitlines():
        if line.startswith("TOPIC:"):
            topic = line[len("TOPIC:"):].strip()
            section = "topic"
        elif line.startswith("INSIGHT:"):
            insight = line[len("INSIGHT:"):].strip()
            section = "insight"
        elif line.startswith("QUOTES:"):
            section = "quotes"
        elif section == "quotes" and line.strip():
            quotes.append(line.rstrip())
    return topic, insight, quotes


def _clean_quote(q: str) -> str:
    return q.strip().strip(_WRAP_CHARS).strip()


def fuzzy_find_in_source(
    quote: str, article: str, threshold: float = DEFAULT_QUOTE_THRESHOLD
) -> Optional[str]:
    """Return the verbatim span of ``article`` that best matches ``quote``.

    Exact substrings are returned as-is. Otherwise the closest contiguous span
    of the article is located (char-level diff) and returned *only* if it is
    similar enough — so the result is always a real substring of the article.
    Returns None when nothing matches well (a likely hallucination/paraphrase).
    """
    q = _clean_quote(quote)
    if not q:
        return None
    if q in article:
        return q

    # Anchor on the matching blocks between article and quote, then take the
    # article span those blocks cover (extended to span the whole quote).
    sm = difflib.SequenceMatcher(None, article, q, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    a_start = max(0, blocks[0].a - blocks[0].b)
    last = blocks[-1]
    a_end = min(len(article), last.a + last.size + (len(q) - (last.b + last.size)))
    if a_end <= a_start:
        return None
    cand = article[a_start:a_end]
    if difflib.SequenceMatcher(None, q, cand).ratio() >= threshold:
        return cand
    return None


def verbatimize_quotes(
    quotes: Optional[List[str]], article: str, threshold: float = DEFAULT_QUOTE_THRESHOLD
) -> Tuple[List[str], int]:
    """Map each quote to its verbatim source span; drop ones with no match.

    Returns (verbatim_quotes, n_dropped). De-duplicates while preserving order.
    """
    out: List[str] = []
    dropped = 0
    seen = set()
    for q in quotes or []:
        match = fuzzy_find_in_source(q, article, threshold)
        if match is None:
            dropped += 1
            continue
        if match not in seen:
            seen.add(match)
            out.append(match)
    return out, dropped


def _box_dict_to_text(b: dict) -> str:
    """Render a {TOPIC, INSIGHT, QUOTES} object into the canonical box string."""
    lower = {k.lower(): v for k, v in b.items()}
    topic = (lower.get("topic") or "").strip()
    insight = (lower.get("insight") or "").strip()
    quotes = lower.get("quotes") or []
    if isinstance(quotes, str):
        quotes = [quotes]
    quote_lines = [str(q).strip() for q in quotes if str(q).strip()]
    if not (topic or insight or quote_lines):
        return ""
    lines = [f"TOPIC: {topic}", f"INSIGHT: {insight}", "QUOTES:"]
    lines.extend(quote_lines)
    return "\n".join(lines)


def _first_json_object(text: str) -> Optional[dict]:
    """Return the first top-level JSON object in ``text``, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _extract_boxes(completion: str) -> Optional[List[str]]:
    """Pull the ``boxes`` array out of the model's JSON completion.

    Returns the list of box strings (possibly empty — a valid result for
    boilerplate-only articles), or ``None`` when no valid ``{"boxes": [...]}``
    object can be found (an empty/blocked/garbled response worth retrying).
    Tolerates ```json fences, a reasoning preamble, or trailing text.
    """
    text = completion.strip()
    if not text:
        return None
    # Strip <think>...</think> if the model emitted one.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

    obj = _first_json_object(text)
    if obj is None or not isinstance(obj.get("boxes"), list):
        # Fallback for gemini JSON mode emitting each box as a bare single-string
        # object with no value -- {"TOPIC: ...\nINSIGHT: ...\nQUOTES:\n..."} --
        # which is invalid JSON. Collapse those wrappers back to the string and
        # re-parse once before giving up.
        repaired = re.sub(r'\{\s*("(?:[^"\\]|\\.)*")\s*\}', r"\1", text)
        obj = _first_json_object(repaired) if repaired != text else obj
    if obj is None or not isinstance(obj.get("boxes"), list):
        return None

    out: List[str] = []
    for b in obj["boxes"]:
        if isinstance(b, str) and b.strip():
            out.append(b)
        elif isinstance(b, dict):
            # In JSON mode the model returns structured boxes; rebuild the
            # canonical TOPIC/INSIGHT/QUOTES text that split_box() expects.
            box = _box_dict_to_text(b)
            if box:
                out.append(box)
    return out


def parse_boxes(completion: str) -> List[str]:
    """Backward-compatible wrapper: returns boxes, or [] when none are found."""
    return _extract_boxes(completion) or []
