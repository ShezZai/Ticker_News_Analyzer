"""Insight-box extraction: chunk each article into decision-useful boxes
(Gemini, structured output), verbatimize quotes against the source, store in
public.article_insights, embed with the shared embedding stage."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import psycopg
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from ticker_news.shared.llm import EMBED_DIM, GEMINI_FLASH, GEMINI_FLASH_LITE, gemini_chat

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

logger = logging.getLogger(__name__)

GEMINI_TIMEOUT_S = 120.0
MAX_ARTICLE_CHARS = 48_000
RETRIES = 5  # lite-chain attempt count before handing off to flash fallback

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


class InsightBoxes(BaseModel):
    boxes: list[str]


@lru_cache(maxsize=1)
def _box_chain():
    def _tagged(structured, model_name):
        return structured | RunnableLambda(
            lambda r, m=model_name: {"boxes": r.boxes, "model": m}
        )

    lite = gemini_chat(GEMINI_FLASH_LITE, timeout_s=GEMINI_TIMEOUT_S)
    flash = gemini_chat(GEMINI_FLASH, timeout_s=GEMINI_TIMEOUT_S)
    lite_chain = _tagged(
        lite.with_structured_output(InsightBoxes).with_retry(
            stop_after_attempt=RETRIES, wait_exponential_jitter=True
        ),
        GEMINI_FLASH_LITE,
    )
    flash_chain = _tagged(
        flash.with_structured_output(InsightBoxes).with_retry(
            stop_after_attempt=2, wait_exponential_jitter=True
        ),
        GEMINI_FLASH,
    )
    # NOTE: falls back on ANY lite-chain failure (legacy only swapped on 5xx/parse
    # errors) — auth/quota errors will burn 5+2 attempts before surfacing.
    return lite_chain.with_fallbacks([flash_chain])


def generate_boxes(article_text: str, *, chain=None) -> tuple[list[str], str]:
    """Run the analyst prompt over one article. Returns (boxes, model_used)."""
    chain = chain if chain is not None else _box_chain()
    result = chain.invoke(PROMPT_TEMPLATE.format(article=article_text[:MAX_ARTICLE_CHARS]))
    return result["boxes"], result["model"]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
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
        # Marker stamped once an article has been through extraction, INCLUDING
        # articles that yielded zero boxes (pure boilerplate). Absence of insight
        # rows alone can't tell "processed, empty" from "never processed", so
        # without this such articles would be re-sent to the LLM on every run.
        cur.execute(
            "ALTER TABLE public.articles "
            "ADD COLUMN IF NOT EXISTS insights_extracted_at timestamptz"
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

    By default skips articles already processed -- those with rows in
    article_insights OR stamped with ``insights_extracted_at`` (the boilerplate
    "zero boxes" case). ``reprocess`` includes them (their old rows are deleted
    before re-inserting). ``ids`` restricts to specific article ids (implies
    reprocess for those rows).
    """
    clauses = ["content IS NOT NULL", "char_length(content) > 0"]
    params: List[object] = []
    if ids:
        clauses.append("a.id = ANY(%s)")
        params.append(list(ids))
    elif not reprocess:
        clauses.append(
            "a.insights_extracted_at IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM public.article_insights ai "
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
# Storage helpers
# --------------------------------------------------------------------------- #
def _load_box_annotator(conn) -> Optional[Callable[[str], str]]:
    """Build the ticker<->company annotator from ticker_data (via tagging)."""
    from ticker_news.enrichment.tagging import build_annotator, load_ticker_data

    try:
        data = load_ticker_data(conn)
        return build_annotator(data)
    except SystemExit:
        logger.warning("ticker_data is empty; skipping annotation")
        return None


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
    model=GEMINI_FLASH_LITE,  # NOTE: callers must pass the actual producing model
) -> Tuple[int, int]:
    """Verbatimize quotes and write one article's boxes. Returns (n_boxes, dropped)."""
    from ticker_news.enrichment.insights_text import (
        split_box,
        verbatimize_quotes,
        with_headline,
    )

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
        cur.execute(
            "UPDATE public.articles SET insights_extracted_at = now() WHERE id = %s",
            (aid,),
        )
    conn.commit()
    return len(payload), dropped_total


# --------------------------------------------------------------------------- #
# Extraction pass
# --------------------------------------------------------------------------- #
def extract_all(
    reprocess: bool, limit: Optional[int],
    quote_threshold: float = 0.75,
    ids: Optional[Sequence[int]] = None,
    workers: int = 8,
) -> int:
    """Generate and store insight boxes. Returns the number of boxes written.

    Gemini calls run concurrently across ``workers`` threads (the work is
    network-bound); DB writes are serialized on the main thread for safety.
    """
    from ticker_news.shared.db import connect

    conn = connect(vector=True)
    try:
        ensure_schema(conn)
        rows = articles_to_process(conn, reprocess, limit, ids=ids)
        if not rows:
            print("No articles to process.")
            return 0
        # An explicit id list always rebuilds those rows.
        do_reprocess = reprocess or bool(ids)
        print(f"Extracting insights from {len(rows)} article(s) with {workers} worker(s) ...")

        annotate = _load_box_annotator(conn)  # ticker<->company cross-annotation
        total_boxes = 0
        total_dropped = 0
        n_empty = 0       # articles the model judged to have no insight (boilerplate)
        n_failed = 0      # articles the model never answered for, even after retries

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(generate_boxes, content or ""):
                    (aid, url, title, content)
                for aid, url, title, content in rows
            }
            completed = as_completed(futures)
            pbar = tqdm(completed, total=len(futures), unit="article") if tqdm else completed
            for fut in pbar:
                aid, url, title, content = futures[fut]
                try:
                    boxes, model_used = fut.result()
                except Exception as exc:  # exhausted retries on a single article
                    n_failed += 1
                    logger.exception("article %s: %s", aid, exc)
                    print(f"  article {aid}: {exc}")
                    continue
                if boxes:
                    n_boxes, dropped = _store_article_boxes(
                        conn, aid, url, title, content, boxes, do_reprocess,
                        quote_threshold, annotate=annotate, model=model_used,
                    )
                    total_boxes += n_boxes
                    total_dropped += dropped
                else:
                    n_empty += 1  # valid empty result — nothing worth storing
                    with conn.cursor() as cur:
                        if do_reprocess:
                            # now judged boilerplate; clear any stale rows
                            cur.execute(
                                "DELETE FROM public.article_insights WHERE article_id = %s",
                                (aid,),
                            )
                        # stamp it processed so it isn't re-sent to the LLM next run
                        cur.execute(
                            "UPDATE public.articles SET insights_extracted_at = now() "
                            "WHERE id = %s",
                            (aid,),
                        )
                    conn.commit()

        print(
            f"Wrote {total_boxes} insight box(es) from {len(rows)} article(s); "
            f"{n_empty} had no insight, {n_failed} failed. "
            f"Dropped {total_dropped} non-verbatim quote(s)."
        )
        return total_boxes
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Re-verbatimize quotes on already-stored rows (no LLM)
# --------------------------------------------------------------------------- #
def fix_quotes(quote_threshold: float = 0.75) -> int:
    """Re-run verbatim quote recovery over existing insights using their source.

    Re-parses each stored ``box_text`` (the raw model output) so corrections are
    applied to the original quotes, then writes the verbatim quotes back to the
    ``quotes`` column. Idempotent. Returns the number of rows updated.
    """
    from ticker_news.enrichment.insights_text import split_box, verbatimize_quotes
    from ticker_news.shared.db import connect

    conn = connect(vector=True)
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
def embed_missing(batch_size: int = 256, build_index: bool = True) -> int:
    """Embed every insight whose embedding is still NULL with text-embedding-3-small."""
    from ticker_news.embedding.embedder import embed_texts
    from ticker_news.shared.db import connect

    conn = connect(vector=True)
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

        from ticker_news.shared.llm import EMBED_MODEL
        print(f"Embedding {len(rows)} insight(s) via {EMBED_MODEL} ...")

        done = 0
        batches = range(0, len(rows), batch_size)
        pbar = tqdm(batches, unit="batch") if tqdm else batches
        for start in pbar:
            chunk = rows[start : start + batch_size]
            texts = [t for _, t in chunk]
            embs = embed_texts(texts, batch_size=batch_size)
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE public.article_insights SET embedding = %s WHERE id = %s",
                    list(zip(embs, [rid for rid, _ in chunk])),
                )
            conn.commit()
            done += len(chunk)

        print(f"Embedded {done} insight(s).")
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
