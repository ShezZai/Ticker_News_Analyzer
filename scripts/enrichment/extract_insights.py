"""Chunk each article into logical "insight boxes" with Gemini and store them.

For every article this sends the analyst prompt below to Gemini 2.5 Flash-Lite
(via the google-genai API). The model returns a JSON object
``{"boxes": [ ... ]}`` where each box is a short text block:

    TOPIC: <short label>
    INSIGHT: <1-2 sentence takeaway>
    QUOTES:
    <verbatim quote 1>
    <verbatim quote 2>

Each box is parsed and written as one row of ``public.article_insights``, with a
foreign key back to its source article plus denormalized ``source_url`` /
``article_headline`` so a RAG hit can say which original article it came from.
The article's headline is also baked into ``box_text`` as a leading
``ARTICLE_HEADLINE:`` line, so it travels with the box and is embedded for RAG.

Stored ``quotes`` are forced to be verbatim: each model quote is matched back to
the source article (exact, else fuzzy) and replaced with the real source span;
quotes with no good match are dropped.

Each box's text is embedded with OpenAI text-embedding-3-small into a
``vector(1536)`` column — the *same* model/space as ``articles.embedding`` — so
insights and full articles are mutually searchable. An HNSW cosine index is
built at the end.

Re-runnable: by default only articles with no insight rows yet are processed
(``--only-missing`` is the default). Use ``--reprocess`` to delete and rebuild an
article's insights. Embeddings fill any rows whose embedding is still NULL, so
the embed pass is independently resumable.

Usage:
    python extract_insights.py --limit 5        # smoke test on a few articles
    python extract_insights.py                  # all un-processed articles
    python extract_insights.py --workers 16     # more concurrent Gemini requests
    python extract_insights.py --ids 78,79,80   # (re)process specific article ids
    python extract_insights.py --reprocess --limit 5
    python extract_insights.py --no-embed       # extract only, embed later
    python extract_insights.py --embed-only     # only fill missing embeddings
    python extract_insights.py --fix-quotes     # re-verbatimize existing rows (no LLM)

Requires GOOGLE_API_KEY (Gemini) and OPENAI_API_KEY (embeddings) in env / .env.
Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, Tuple

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"

GEMINI_MODEL = "gemini-2.5-flash-lite"
# flash-lite + response_schema can hang server-side (504 DEADLINE_EXCEEDED) on
# some articles; fall back to the sturdier non-lite model for those.
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"
# google-genai has NO default request timeout — without this a single stuck call
# blocks the whole run indefinitely. Milliseconds.
GEMINI_TIMEOUT_MS = 120_000
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Approximate published $/token rates for the running cost readout (verify
# against your billing console; rates change).
GEMINI_INPUT_COST = 0.10 / 1e6   # gemini-2.5-flash-lite input
GEMINI_OUTPUT_COST = 0.40 / 1e6  # gemini-2.5-flash-lite output
EMBED_COST = 0.02 / 1e6          # text-embedding-3-small

# Cap the article text sent to the model (chars; a token is >= 1 char so this is
# a safe upper bound). ~48k chars is well within Gemini's context window.
MAX_ARTICLE_CHARS = 48_000

PROMPT_TEMPLATE = """You are an equity analyst extracting decision-useful insights from a news article.

Your goal: capture only the chunks that would actually help analyze a company,
ticker, or the market — the kind of context worth recalling when reading a NEW
article later. Each chunk = one coherent, market-relevant idea.

EXTRACT chunks that convey real news or sentiment, such as:
- Financial results and metrics (revenue, EPS, margins, growth, cash flow).
- Guidance, outlook, forecasts, and estimate revisions.
- Demand/supply signals, orders, backlog, capacity, capex, pricing.
- Products, technology, launches, roadmaps, competitive positioning.
- Deals: M&A, partnerships, contracts, customers, investments.
- Management/strategy changes; regulatory, legal, or geopolitical events
  that materially affect the business.
- Analyst actions (ratings, price targets) and stock-price moves/reactions.
- Clear sentiment or risk signals — bullish/bearish drivers, catalysts, concerns.

IGNORE and DO NOT create boxes for non-substantive boilerplate, such as:
- Law-firm / "shareholder investigation" / class-action solicitations, how to
  join a lawsuit, eligibility for compensation, attorney advertising, contact
  details for legal counsel.
- Forward-looking-statement / safe-harbor disclaimers, legal notices, copyright.
- Subscription/paywall/advertising copy, newsletter promos, author bios.
- Generic "about the company" descriptions and navigation/footer text.
- Vague filler with no concrete, market-relevant information.

If the article is entirely boilerplate or has no decision-useful content,
return {{"boxes": []}} — an empty list is correct and expected. Quality over
coverage: do NOT pad with weak boxes.

Return ONE JSON object with a single key "boxes" whose value is an array of
strings. Each string is ONE text box, formatted EXACTLY like this (use \\n line
breaks inside the string):

TOPIC: <short label, 3-6 words>
INSIGHT: <1-2 sentences in your own words: the takeaway and, when relevant, its
bullish/bearish implication for the company or ticker>
QUOTES:
<verbatim quote 1, copied exactly from the article>
<verbatim quote 2>

Rules:
- One text box per market-relevant chunk; list them in reading order.
- QUOTES must be VERBATIM substrings of the article — copy character for
  character. Never paraphrase, fix, shorten, or stitch non-adjacent text.
  One quote per line. Include 1-3 quotes per box.
- Never invent facts or quotes.
- Output ONLY the JSON object. No markdown, no preamble.
- Valid JSON only: escape newlines inside strings as \\n and escape any
  double quotes inside the text as \\".

Example of one element in "boxes":
"TOPIC: Q3 revenue beat\\nINSIGHT: The company topped estimates, driven mainly by cloud growth — a bullish demand signal.\\nQUOTES:\\nRevenue rose 18% year over year to $4.2 billion.\\nCloud segment sales jumped 31%."

ARTICLE:
\"\"\"
{article}
\"\"\""""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(DB_DSN)
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the article_insights table (and embedding column) if missing."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone() is None:
            raise SystemExit(
                "pgvector extension is not installed in this database.\n"
                "Create it once as a superuser:\n"
                "    sudo -u postgres psql -d news -c 'CREATE EXTENSION vector;'"
            )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS public.article_insights (
                id            bigserial PRIMARY KEY,
                article_id    bigint NOT NULL
                              REFERENCES public.articles(id) ON DELETE CASCADE,
                source_url       text,       -- denormalized: which article this is from
                article_headline text,       -- denormalized: the source article's headline
                box_index     int  NOT NULL, -- order of the box within the article
                topic         text,
                insight       text,
                quotes        text[],
                box_text      text NOT NULL, -- box, prefixed with an ARTICLE_HEADLINE: line
                model         text,          -- generating model (e.g. gemini-2.5-flash-lite)
                embedding     vector({EMBED_DIM}),
                created_at    timestamptz NOT NULL DEFAULT now(),
                UNIQUE (article_id, box_index)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS article_insights_article_id_idx "
            "ON public.article_insights (article_id);"
        )
    conn.commit()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def articles_to_process(
    conn: psycopg.Connection, reprocess: bool, limit: Optional[int],
    ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, str, str, str]]:
    """Return (id, url, title, content) for articles needing insight extraction.

    By default skips articles that already have rows in article_insights;
    ``reprocess`` includes them (their old rows are deleted before re-inserting).
    ``ids`` restricts to specific article ids (implies reprocess for those rows).
    """
    clauses = ["content IS NOT NULL", "char_length(content) > 0"]
    params: List[object] = []
    if ids:
        clauses.append("a.id = ANY(%s)")
        params.append(list(ids))
    elif not reprocess:
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM public.article_insights ai "
            "WHERE ai.article_id = a.id)"
        )
    where = " AND ".join(clauses)
    lim = f"LIMIT {int(limit)}" if limit else ""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT a.id, a.url, a.title, a.content FROM public.articles a "
            f"WHERE {where} ORDER BY a.id {lim}",
            params,
        )
        return cur.fetchall()


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
_gemini_client = None
_gemini_config = None


def load_gemini():
    """Construct a singleton google-genai client + generation config for Gemini."""
    global _gemini_client, _gemini_config
    if _gemini_client is None:
        from google import genai
        from google.genai import types

        if not os.getenv("GOOGLE_API_KEY"):
            raise SystemExit("GOOGLE_API_KEY is not set (put it in .env).")
        print(f"Using {GEMINI_MODEL} via google-genai ...")
        _gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        # Force clean JSON and disable "thinking" for speed/cost determinism.
        # A response_schema constrains generation to {"boxes": [<string>, ...]}
        # so the SDK returns guaranteed-valid JSON — without it, flash-lite emits
        # degenerate {"<box text>"} objects with unescaped quotes that won't parse.
        _gemini_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "boxes": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["boxes"],
            },
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            # bound every request so a hung call can't freeze the run
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
    return _gemini_client, _gemini_config


def _is_retryable_server_error(exc: Exception) -> bool:
    """True for server deadline / unavailable / timeout errors worth a model swap."""
    s = str(exc).lower()
    return any(k in s for k in (
        "deadline_exceeded", "504", "503", "unavailable", "timeout", "timed out",
    ))


def generate_boxes(
    client, config, article_text: str, retries: int = 5
) -> Tuple[List[str], int, int, str]:
    """Run the analyst prompt over one article.

    Returns (boxes, input_tokens, output_tokens, model_used). Retries with
    exponential backoff both on transient API errors (rate limits, 5xx) AND when
    the model fails to answer usably — an empty/blocked response or output that
    doesn't contain a valid ``{"boxes": [...]}`` object. A *valid* empty list (the
    correct result for boilerplate-only articles) is returned without retry.

    On a server deadline/timeout (flash-lite + response_schema can 504 on certain
    inputs), or after a repeated parse failure, it switches to GEMINI_FALLBACK_MODEL
    for the remaining attempts. ``model_used`` reflects which model actually
    answered, so callers can record accurate provenance.
    """
    import time

    prompt = PROMPT_TEMPLATE.format(article=article_text[:MAX_ARTICLE_CHARS])
    last_reason = "no response"
    cur_model = GEMINI_MODEL
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=cur_model, contents=prompt, config=config
            )
            um = getattr(resp, "usage_metadata", None)
            in_tok = getattr(um, "prompt_token_count", 0) or 0
            out_tok = getattr(um, "candidates_token_count", 0) or 0
            boxes = _extract_boxes(resp.text or "")
            if boxes is not None:  # valid answer (possibly an intentional [])
                return boxes, in_tok, out_tok, cur_model
            last_reason = "no parseable boxes in response"
            # an unparseable response on the primary model: try the fallback next
            if cur_model != GEMINI_FALLBACK_MODEL and attempt >= 1:
                cur_model = GEMINI_FALLBACK_MODEL
        except Exception as exc:  # network / rate-limit / 5xx / timeout
            last_reason = repr(exc)
            if cur_model != GEMINI_FALLBACK_MODEL and _is_retryable_server_error(exc):
                cur_model = GEMINI_FALLBACK_MODEL  # swap models for remaining tries
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s, ...
    raise RuntimeError(f"model did not answer after {retries} tries: {last_reason}")


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


HEADLINE_PREFIX = "ARTICLE_HEADLINE:"


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


# --------------------------------------------------------------------------- #
# Verbatim quote recovery
# --------------------------------------------------------------------------- #
# Surrounding quote marks the model often adds; stripped before matching.
_WRAP_CHARS = " \t\"'“”‘’"
DEFAULT_QUOTE_THRESHOLD = 0.75


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


# --------------------------------------------------------------------------- #
# Extraction pass
# --------------------------------------------------------------------------- #
def _load_box_annotator(conn) -> Optional[Callable[[str], str]]:
    """Build the ticker<->company annotator from ticker_data (via tag_segments)."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tag_segments

    return tag_segments.build_annotator(tag_segments.load_ticker_data(conn))


def annotate_box(stored_box: str, annotate: Optional[Callable[[str], str]]) -> str:
    """Cross-annotate ticker/company mentions in a box, leaving QUOTES verbatim.

    Only the headline/TOPIC/INSIGHT portion is annotated; everything from the
    ``QUOTES:`` line onward is left byte-for-byte so quotes stay verbatim.
    """
    if not annotate:
        return stored_box
    lines = stored_box.split("\n")
    qi = next((i for i, ln in enumerate(lines) if ln.strip() == "QUOTES:"), len(lines))
    head = annotate("\n".join(lines[:qi]))
    tail = lines[qi:]
    return head + "\n" + "\n".join(tail) if tail else head


def _store_article_boxes(
    conn, aid, url, title, content, boxes, reprocess, quote_threshold, annotate=None,
    model=GEMINI_MODEL,
) -> Tuple[int, int]:
    """Verbatimize quotes and write one article's boxes. Returns (n_boxes, dropped)."""
    payload = []
    dropped_total = 0
    for idx, box_text in enumerate(boxes):
        topic, insight, quotes = split_box(box_text)
        quotes, dropped = verbatimize_quotes(quotes, content or "", quote_threshold)
        dropped_total += dropped
        # headline before box content, then cross-annotate tickers/company names
        stored_box = annotate_box(with_headline(box_text, title), annotate)
        payload.append(
            (aid, url, title, idx, topic, insight, quotes or None,
             stored_box, model)
        )

    with conn.cursor() as cur:
        if reprocess:
            cur.execute(
                "DELETE FROM public.article_insights WHERE article_id = %s", (aid,)
            )
        cur.executemany(
            "INSERT INTO public.article_insights "
            "(article_id, source_url, article_headline, box_index, topic, "
            " insight, quotes, box_text, model) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (article_id, box_index) DO UPDATE SET "
            "  topic = EXCLUDED.topic, insight = EXCLUDED.insight, "
            "  quotes = EXCLUDED.quotes, box_text = EXCLUDED.box_text, "
            "  model = EXCLUDED.model, embedding = NULL",
            payload,
        )
    conn.commit()
    return len(payload), dropped_total


def extract_all(
    reprocess: bool, limit: Optional[int],
    quote_threshold: float = DEFAULT_QUOTE_THRESHOLD,
    ids: Optional[Sequence[int]] = None,
    workers: int = 8,
) -> int:
    """Generate and store insight boxes. Returns the number of boxes written.

    Gemini calls run concurrently across ``workers`` threads (the work is
    network-bound); DB writes are serialized on the main thread for safety.
    """
    conn = get_conn()
    try:
        ensure_schema(conn)
        rows = articles_to_process(conn, reprocess, limit, ids=ids)
        if not rows:
            print("No articles to process.")
            return 0
        # An explicit id list always rebuilds those rows.
        do_reprocess = reprocess or bool(ids)
        print(f"Extracting insights from {len(rows)} article(s) with {workers} worker(s) ...")

        client, config = load_gemini()
        annotate = _load_box_annotator(conn)  # ticker<->company cross-annotation
        total_boxes = 0
        total_dropped = 0
        n_empty = 0       # articles the model judged to have no insight (boilerplate)
        n_failed = 0      # articles the model never answered for, even after retries
        in_tok = out_tok = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(generate_boxes, client, config, content or ""):
                    (aid, url, title, content)
                for aid, url, title, content in rows
            }
            completed = as_completed(futures)
            pbar = tqdm(completed, total=len(futures), unit="article") if tqdm else completed
            for fut in pbar:
                aid, url, title, content = futures[fut]
                try:
                    boxes, a_in, a_out, model_used = fut.result()
                except Exception as exc:  # exhausted retries on a single article
                    n_failed += 1
                    print(f"  article {aid}: {exc}")
                    continue
                in_tok += a_in
                out_tok += a_out
                if boxes:
                    n_boxes, dropped = _store_article_boxes(
                        conn, aid, url, title, content, boxes, do_reprocess,
                        quote_threshold, annotate=annotate, model=model_used,
                    )
                    total_boxes += n_boxes
                    total_dropped += dropped
                else:
                    n_empty += 1  # valid empty result — nothing worth storing
                    if do_reprocess:
                        # the article is now judged boilerplate; clear any stale rows
                        with conn.cursor() as cur:
                            cur.execute(
                                "DELETE FROM public.article_insights WHERE article_id = %s",
                                (aid,),
                            )
                        conn.commit()

                if tqdm:
                    cost = in_tok * GEMINI_INPUT_COST + out_tok * GEMINI_OUTPUT_COST
                    pbar.set_postfix_str(
                        f"${cost:.2f} | {(in_tok + out_tok) / 1e6:.2f}M tok | "
                        f"{total_boxes} boxes | {n_empty} empty | {n_failed} failed"
                    )

        cost = in_tok * GEMINI_INPUT_COST + out_tok * GEMINI_OUTPUT_COST
        print(
            f"Wrote {total_boxes} insight box(es) from {len(rows)} article(s); "
            f"{n_empty} had no insight, {n_failed} failed. "
            f"Dropped {total_dropped} non-verbatim quote(s)."
        )
        print(
            f"Gemini usage: {in_tok/1e6:.2f}M input + {out_tok/1e6:.2f}M output "
            f"tokens ~ ${cost:.2f}"
        )
        return total_boxes
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Re-verbatimize quotes on already-stored rows (no LLM)
# --------------------------------------------------------------------------- #
def fix_quotes(quote_threshold: float = DEFAULT_QUOTE_THRESHOLD) -> int:
    """Re-run verbatim quote recovery over existing insights using their source.

    Re-parses each stored ``box_text`` (the raw model output) so corrections are
    applied to the original quotes, then writes the verbatim quotes back to the
    ``quotes`` column. Idempotent. Returns the number of rows updated.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ai.id, ai.box_text, a.content "
                "FROM public.article_insights ai "
                "JOIN public.articles a ON a.id = ai.article_id "
                "ORDER BY ai.id"
            )
            rows = cur.fetchall()
        if not rows:
            print("No insights to fix.")
            return 0

        updated = 0
        dropped = 0
        iterator = tqdm(rows, unit="insight") if tqdm else rows
        for iid, box_text, content in iterator:
            _, _, quotes = split_box(box_text or "")
            verbatim, n_dropped = verbatimize_quotes(quotes, content or "", quote_threshold)
            dropped += n_dropped
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.article_insights SET quotes = %s WHERE id = %s",
                    (verbatim or None, iid),
                )
            updated += 1
        conn.commit()
        print(f"Fixed quotes on {updated} insight(s); dropped {dropped} non-verbatim quote(s).")
        return updated
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Embedding pass
# --------------------------------------------------------------------------- #
# text-embedding-3-small hard-caps inputs at 8192 tokens; trim just under it.
MAX_EMBED_TOKENS = 8000
_openai_client = None
_embed_encoder = None


def _truncate_embed(text: str) -> str:
    """Trim to <= MAX_EMBED_TOKENS tokens (cl100k_base). Tokens <= chars, so
    short text skips the tokenizer."""
    global _embed_encoder
    text = (text or "").strip()
    if len(text) <= MAX_EMBED_TOKENS:
        return text or " "
    try:
        if _embed_encoder is None:
            import tiktoken
            _embed_encoder = tiktoken.get_encoding("cl100k_base")
        ids = _embed_encoder.encode(text)
        if len(ids) > MAX_EMBED_TOKENS:
            text = _embed_encoder.decode(ids[:MAX_EMBED_TOKENS])
    except ImportError:
        text = text[: MAX_EMBED_TOKENS * 3]
    return text or " "


def _embed_texts(texts: List[str], batch_size: int = 256) -> Tuple[List, int]:
    """Embed texts with OpenAI text-embedding-3-small.

    Returns (np.float32 arrays, total_tokens_used).
    """
    global _openai_client
    import numpy as np

    if _openai_client is None:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (put it in .env).")
        _openai_client = OpenAI()

    cleaned = [_truncate_embed(t) for t in texts]
    out: List = []
    tokens = 0
    for start in range(0, len(cleaned), batch_size):
        chunk = cleaned[start : start + batch_size]
        resp = _openai_client.embeddings.create(model=EMBED_MODEL, input=chunk)
        tokens += getattr(resp.usage, "total_tokens", 0) or 0
        for d in sorted(resp.data, key=lambda d: d.index):
            out.append(np.asarray(d.embedding, dtype=np.float32))
    return out, tokens


def embed_missing(batch_size: int = 256, build_index: bool = True) -> int:
    """Embed every insight whose embedding is still NULL with text-embedding-3-small."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, box_text FROM public.article_insights "
                "WHERE embedding IS NULL ORDER BY id"
            )
            rows = cur.fetchall()
        if not rows:
            print("No insights need embedding.")
            return 0

        print(f"Embedding {len(rows)} insight(s) via {EMBED_MODEL} ...")

        done = 0
        tokens = 0
        batches = range(0, len(rows), batch_size)
        pbar = tqdm(batches, unit="batch") if tqdm else batches
        for start in pbar:
            chunk = rows[start : start + batch_size]
            texts = [t for _, t in chunk]
            embs, used = _embed_texts(texts, batch_size=batch_size)
            tokens += used
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE public.article_insights SET embedding = %s WHERE id = %s",
                    list(zip(embs, [rid for rid, _ in chunk])),
                )
            conn.commit()
            done += len(chunk)
            if tqdm:
                pbar.set_postfix_str(
                    f"${tokens * EMBED_COST:.2f} | {tokens / 1e6:.2f}M tok"
                )

        print(f"Embedded {done} insight(s) "
              f"({tokens/1e6:.2f}M tokens ~ ${tokens * EMBED_COST:.2f}).")
        if build_index:
            print("Building HNSW cosine index (article_insights_embedding_idx) ...")
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS article_insights_embedding_idx "
                    "ON public.article_insights USING hnsw (embedding vector_cosine_ops);"
                )
            conn.commit()
        return done
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Chunk articles into insight boxes with Gemini and store them."
    )
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N pending articles (smoke test)")
    p.add_argument("--reprocess", action="store_true",
                   help="re-extract articles that already have insights")
    p.add_argument("--ids", type=str, default=None,
                   help="comma-separated article ids to (re)process, e.g. 78,79,80")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent Gemini requests (default 8)")
    p.add_argument("--no-embed", dest="embed", action="store_false",
                   help="extract boxes only; skip embedding (fill later)")
    p.add_argument("--embed-only", action="store_true",
                   help="skip extraction; only embed insights missing a vector")
    p.add_argument("--fix-quotes", action="store_true",
                   help="re-verbatimize quotes on existing rows (no LLM, no embedding)")
    p.add_argument("--quote-threshold", type=float, default=DEFAULT_QUOTE_THRESHOLD,
                   help="min similarity to accept a fuzzy quote match "
                        f"(default {DEFAULT_QUOTE_THRESHOLD})")
    p.add_argument("--no-index", dest="build_index", action="store_false",
                   help="skip building the HNSW index after embedding")
    args = p.parse_args()

    if args.fix_quotes:
        fix_quotes(quote_threshold=args.quote_threshold)
        return

    ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None

    if not args.embed_only:
        extract_all(
            reprocess=args.reprocess, limit=args.limit,
            quote_threshold=args.quote_threshold, ids=ids, workers=args.workers,
        )
    if args.embed or args.embed_only:
        embed_missing(build_index=args.build_index)


if __name__ == "__main__":
    main()
