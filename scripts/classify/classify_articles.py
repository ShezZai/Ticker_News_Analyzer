"""Classify each article into a content category with a two-pass Gemini pipeline.

Every article is first sent to gemini-2.5-flash-lite (fast, cheap). If the result
is "real news", the same prompt is re-run on gemini-2.5-flash to confirm: the
stronger model either keeps the label or reclassifies it. All other categories are
accepted as-is from the lite model without a second call.

Categories:

    conference-PR        announcement of a company sponsoring/speaking/exhibiting
                         at a conference, trade show, or awards event; event promo
    marketing fluff      promotional / advertorial / vendor self-promotion, product
                         marketing, newsletter or subscription promos -- no
                         decision-useful financial substance
    real news            first-hand reporting of a concrete company/market event:
                         earnings results, guidance, deals (M&A/partnerships/
                         contracts), product launches, management/regulatory/legal
                         events, analyst actions -- the article IS the primary
                         source breaking the event, not a commentary written after
                         the fact explaining why a stock moved
    recap/review         opinion, educational, or retrospective pieces -- including
                         "why X stock rose/fell/plummeted/skyrocketed today/this
                         week", performance explainers, "should you buy X?",
                         "3 stocks to watch", market-wrap summaries -- any article
                         whose main purpose is to explain or analyse a move that
                         already happened, rather than to report a new event
    market speculation   forward-looking conjecture: predictions, "will X hit $Y",
                         speculative outlooks driven by guesswork
    legal solicitation   law-firm / class-action / securities-fraud solicitations,
                         "investor alert" / "deadline alert" notices to join a suit
    regulatory filing    routine corporate/regulatory disclosures: Form 8.3, annual-
                         meeting / record-date notices, proxy / resolution notices
    book PR              book-launch / author-promotion press releases
    politics/macro       general political, policy, geopolitical, or macroeconomic
                         news not specific to a single company
    other                anything that genuinely fits none of the above (obituaries,
                         human-interest, navigation/boilerplate, etc.)

The label is written to ``public.articles.category`` (plus a one-line
``category_reason``). A ``response_schema`` with an ``enum`` constrains the model
to a valid label, so the output is always one of the categories above.

Re-runnable: by default only rows whose ``category`` is still NULL are processed,
so the job is fully resumable. ``--reprocess`` re-classifies every (matching) row;
``--ids`` restricts to specific article ids.

Usage:
    python classify_articles.py                 # classify all unclassified articles
    python classify_articles.py --limit 20      # smoke test on a few
    python classify_articles.py --workers 16    # more concurrent requests
    python classify_articles.py --ids 78,79,80  # (re)classify specific ids
    python classify_articles.py --reprocess     # re-classify everything

Requires GOOGLE_API_KEY (Gemini) in the environment / .env.
Connection comes from NEWS_DB_DSN / DATABASE_URL, defaulting to ``dbname=news``.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Tuple

import psycopg
from dotenv import load_dotenv

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"

GEMINI_MODEL = "gemini-2.5-flash-lite"   # initial pass (all articles)
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash-lite"
CONFIRM_MODEL = "gemini-2.5-flash"       # confirmation pass (real news only)
# google-genai has NO default request timeout -- without this a single stuck call
# blocks the whole run. Milliseconds.
GEMINI_TIMEOUT_MS = 60_000

# Approximate published $/token rates for the running cost readout (verify against
# your billing console; rates change).
GEMINI_INPUT_COST   = 0.10 / 1e6   # flash-lite input
GEMINI_OUTPUT_COST  = 0.40 / 1e6   # flash-lite output
CONFIRM_INPUT_COST  = 0.30 / 1e6   # flash input  (confirmation pass)
CONFIRM_OUTPUT_COST = 2.50 / 1e6   # flash output (confirmation pass)

# Classification needs only the lede, not the whole article; cap to keep it cheap.
MAX_ARTICLE_CHARS = 6_000

CATEGORIES = [
    "conference-PR",
    "marketing fluff",
    "real news",
    "recap/review",
    "market speculation",
    "legal solicitation",
    "regulatory filing",
    "book PR",
    "politics/macro",
    "other",
]

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


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def get_conn() -> psycopg.Connection:
    return psycopg.connect(DB_DSN)


def ensure_schema(conn: psycopg.Connection) -> None:
    """Add the category columns + a filter index if they don't already exist."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE public.articles "
            "ADD COLUMN IF NOT EXISTS category        text, "
            "ADD COLUMN IF NOT EXISTS category_reason text"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_category_idx "
            "ON public.articles (category)"
        )
    conn.commit()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def articles_to_process(
    conn: psycopg.Connection, reprocess: bool, limit: Optional[int],
    ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, str, str]]:
    """Return (id, title, content) for articles needing classification.

    Only content-bearing rows are considered. By default rows that already have a
    ``category`` are skipped; ``reprocess`` includes them; ``ids`` restricts to
    specific article ids (and re-classifies them regardless of current category).
    """
    clauses = ["content IS NOT NULL", "char_length(content) > 0"]
    params: List[object] = []
    if ids:
        clauses.append("id = ANY(%s)")
        params.append(list(ids))
    elif not reprocess:
        clauses.append("category IS NULL")
    where = " AND ".join(clauses)
    lim = f"LIMIT {int(limit)}" if limit else ""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, title, content FROM public.articles "
            f"WHERE {where} ORDER BY id {lim}",
            params,
        )
        return cur.fetchall()


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
_client = None
_config = None


def load_gemini():
    """Construct a singleton google-genai client + generation config.

    A response_schema with an ``enum`` constrains the answer to a valid category;
    an http timeout bounds every request so a hung call can't freeze the run.
    """
    global _client, _config
    if _client is None:
        from google import genai
        from google.genai import types

        if not os.getenv("GOOGLE_API_KEY"):
            raise SystemExit("GOOGLE_API_KEY is not set (put it in .env).")
        print(f"Using {GEMINI_MODEL} via google-genai …")
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        _config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "category": {"type": "STRING", "enum": CATEGORIES},
                    "reason": {"type": "STRING"},
                },
                "required": ["category"],
            },
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
    return _client, _config


def _is_retryable_server_error(exc: Exception) -> bool:
    """True for server deadline / unavailable / timeout errors worth a model swap."""
    s = str(exc).lower()
    return any(k in s for k in (
        "deadline_exceeded", "504", "503", "unavailable", "timeout", "timed out",
    ))


def _parse(text: str) -> Optional[Tuple[str, str]]:
    """Pull (category, reason) out of the model's JSON, or None if unusable."""
    if not text or not text.strip():
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    cat = (obj.get("category") or "").strip()
    if cat not in CATEGORIES:
        return None
    return cat, (obj.get("reason") or "").strip()


def classify_one(
    client, config, title: Optional[str], content: str, retries: int = 4
) -> Tuple[str, str, int, int, int, int]:
    """Classify one article with a two-pass strategy.

    First pass: gemini-2.5-flash-lite (all articles).
    Second pass: gemini-2.5-flash only when the first pass returns "real news",
    to confirm or correct the label.

    Returns (category, reason, lite_in_tok, lite_out_tok, confirm_in_tok,
    confirm_out_tok). Confirm tokens are 0 when no confirmation was needed.
    """
    import time

    prompt = PROMPT_TEMPLATE.format(
        title=(title or "").strip()[:300], body=(content or "")[:MAX_ARTICLE_CHARS]
    )

    # ── first pass: flash-lite ─────────────────────────────────────────────────
    last_reason = "no response"
    cur_model = GEMINI_MODEL
    lite_in = lite_out = 0
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=cur_model, contents=prompt, config=config
            )
            um = getattr(resp, "usage_metadata", None)
            lite_in  = getattr(um, "prompt_token_count",     0) or 0
            lite_out = getattr(um, "candidates_token_count", 0) or 0
            parsed = _parse(resp.text or "")
            if parsed is not None:
                category, reason = parsed
                break
            last_reason = "no valid category in response"
            if cur_model != GEMINI_FALLBACK_MODEL and attempt >= 1:
                cur_model = GEMINI_FALLBACK_MODEL
        except Exception as exc:  # network / rate-limit / 5xx / timeout
            last_reason = repr(exc)
            if cur_model != GEMINI_FALLBACK_MODEL and _is_retryable_server_error(exc):
                cur_model = GEMINI_FALLBACK_MODEL
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"model did not answer after {retries} tries: {last_reason}")

    # ── second pass: flash confirmation (real news only) ──────────────────────
    if category != "real news":
        return category, reason, lite_in, lite_out, 0, 0

    conf_in = conf_out = 0
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=CONFIRM_MODEL, contents=prompt, config=config
            )
            um = getattr(resp, "usage_metadata", None)
            conf_in  = getattr(um, "prompt_token_count",     0) or 0
            conf_out = getattr(um, "candidates_token_count", 0) or 0
            parsed = _parse(resp.text or "")
            if parsed is not None:
                return parsed[0], parsed[1], lite_in, lite_out, conf_in, conf_out
            last_reason = "no valid category in confirmation response"
        except Exception as exc:
            last_reason = repr(exc)
            if _is_retryable_server_error(exc) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    # confirmation failed: keep the flash-lite "real news" result
    return category, reason, lite_in, lite_out, conf_in, conf_out


# --------------------------------------------------------------------------- #
# Classification pass
# --------------------------------------------------------------------------- #
def classify_all(
    reprocess: bool = False,
    limit: Optional[int] = None,
    ids: Optional[Sequence[int]] = None,
    workers: int = 8,
    batch_size: int = 200,
) -> int:
    """Classify articles and write the labels back. Returns the number classified.

    Gemini calls run concurrently across ``workers`` threads (network-bound); DB
    writes are batched and committed on the main thread.
    """
    conn = get_conn()
    try:
        ensure_schema(conn)
        rows = articles_to_process(conn, reprocess, limit, ids=ids)
        if not rows:
            print("Nothing to classify.")
            return 0

        print(f"Classifying {len(rows)} article(s) with {workers} worker(s) "
              f"(lite first-pass; flash confirmation for real news) …")
        client, config = load_gemini()

        done = 0
        n_failed = 0
        lite_in = lite_out = conf_in = conf_out = 0
        counts = {c: 0 for c in CATEGORIES}
        pending: List[tuple] = []  # (category, reason, id) awaiting UPDATE

        def flush() -> None:
            if not pending:
                return
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE public.articles SET category = %s, category_reason = %s "
                    "WHERE id = %s",
                    pending,
                )
            conn.commit()
            pending.clear()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(classify_one, client, config, title, content or ""):
                    (aid, title)
                for aid, title, content in rows
            }
            completed = as_completed(futures)
            pbar = tqdm(completed, total=len(futures), unit="article") if tqdm else completed
            for fut in pbar:
                aid, _title = futures[fut]
                try:
                    category, reason, a_lite_in, a_lite_out, a_conf_in, a_conf_out = fut.result()
                except Exception as exc:  # exhausted retries on a single article
                    n_failed += 1
                    print(f"  article {aid}: {exc}")
                    continue
                lite_in  += a_lite_in
                lite_out += a_lite_out
                conf_in  += a_conf_in
                conf_out += a_conf_out
                counts[category] += 1
                done += 1
                pending.append((category, reason or None, aid))
                if len(pending) >= batch_size:
                    flush()
                if tqdm:
                    cost = (lite_in  * GEMINI_INPUT_COST  + lite_out  * GEMINI_OUTPUT_COST
                          + conf_in  * CONFIRM_INPUT_COST + conf_out  * CONFIRM_OUTPUT_COST)
                    pbar.set_postfix_str(
                        f"${cost:.2f} | {(lite_in+lite_out+conf_in+conf_out)/1e6:.2f}M tok "
                        f"| {n_failed} failed"
                    )
            flush()

        lite_cost = lite_in * GEMINI_INPUT_COST + lite_out * GEMINI_OUTPUT_COST
        conf_cost = conf_in * CONFIRM_INPUT_COST + conf_out * CONFIRM_OUTPUT_COST
        print(f"Classified {done} article(s); {n_failed} failed.")
        for c in CATEGORIES:
            print(f"  {c:<20} {counts[c]}")
        print(f"flash-lite: {lite_in/1e6:.2f}M in + {lite_out/1e6:.2f}M out ~ ${lite_cost:.2f}")
        if conf_in or conf_out:
            print(f"flash conf: {conf_in/1e6:.2f}M in + {conf_out/1e6:.2f}M out ~ ${conf_cost:.2f}")
        print(f"total cost: ~${lite_cost + conf_cost:.2f}")
        return done
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Classify articles into content categories with Gemini."
    )
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N pending articles (smoke test)")
    p.add_argument("--reprocess", action="store_true",
                   help="re-classify articles that already have a category")
    p.add_argument("--ids", type=str, default=None,
                   help="comma-separated article ids to (re)classify, e.g. 78,79,80")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent Gemini requests (default 8)")
    args = p.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
    classify_all(reprocess=args.reprocess, limit=args.limit, ids=ids, workers=args.workers)


if __name__ == "__main__":
    main()
