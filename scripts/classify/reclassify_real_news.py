#!/usr/bin/env python3
"""Temp script: re-classify all current 'real news' articles with gemini-2.5-flash.

Reads every article whose category is already 'real news' and re-runs the same
classification prompt on gemini-2.5-flash, overwriting category + category_reason
with whatever the stronger model returns.

Usage:
    python reclassify_real_news.py
    python reclassify_real_news.py --limit 20   # smoke test
    python reclassify_real_news.py --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys

# reuse prompt / helpers from the main classifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_articles import (  # noqa: E402
    CATEGORIES, DB_DSN, GEMINI_TIMEOUT_MS, MAX_ARTICLE_CHARS,
    PROMPT_TEMPLATE, _is_retryable_server_error, _parse, get_conn,
)

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import psycopg
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

load_dotenv()

MODEL = "gemini-2.5-flash"
INPUT_COST  = 0.30 / 1e6
OUTPUT_COST = 2.50 / 1e6


def load_articles(limit: Optional[int]) -> List[tuple]:
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        lim = f"LIMIT {int(limit)}" if limit else ""
        cur.execute(
            f"SELECT id, title, content FROM public.articles "
            f"WHERE category = 'real news' AND content IS NOT NULL "
            f"AND char_length(content) > 0 ORDER BY id {lim}"
        )
        return cur.fetchall()


def _rate_limit_wait(exc: Exception) -> float:
    """Extract the retry-after seconds from a 429 message, default 60."""
    import re
    m = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", str(exc), re.I)
    return float(m.group(1)) if m else 60.0


def classify_one(client, config, title: Optional[str], content: str, retries: int = 6):
    prompt = PROMPT_TEMPLATE.format(
        title=(title or "").strip()[:300],
        body=(content or "")[:MAX_ARTICLE_CHARS],
    )
    last = "no response"
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            um = getattr(resp, "usage_metadata", None)
            in_tok  = getattr(um, "prompt_token_count",     0) or 0
            out_tok = getattr(um, "candidates_token_count", 0) or 0
            parsed = _parse(resp.text or "")
            if parsed:
                return parsed[0], parsed[1], in_tok, out_tok
            last = "no valid category"
        except Exception as exc:
            last = repr(exc)
            is_429 = "429" in str(exc) or "resource_exhausted" in str(exc).lower()
            if is_429:
                wait = _rate_limit_wait(exc)
                time.sleep(wait)
                continue
            if not _is_retryable_server_error(exc):
                raise
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} tries: {last}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit",   type=int, default=None, help="process only first N rows")
    p.add_argument("--workers", type=int, default=8,    help="concurrent requests (default 8)")
    args = p.parse_args()

    rows = load_articles(args.limit)
    if not rows:
        print("No 'real news' articles found.")
        return

    print(f"Re-classifying {len(rows)} 'real news' article(s) with {MODEL} …")

    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "enum": CATEGORIES},
                "reason":   {"type": "STRING"},
            },
            "required": ["category"],
        },
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )

    done = n_failed = in_tok = out_tok = 0
    counts = {c: 0 for c in CATEGORIES}
    pending: list[tuple] = []
    BATCH = 200

    def flush(conn):
        if not pending:
            return
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE public.articles SET category = %s, category_reason = %s WHERE id = %s",
                pending,
            )
        conn.commit()
        pending.clear()

    with psycopg.connect(DB_DSN) as conn:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(classify_one, client, config, title, content or ""): aid
                for aid, title, content in rows
            }
            it = as_completed(futures)
            pbar = tqdm(it, total=len(futures), unit="article") if tqdm else it
            for fut in pbar:
                aid = futures[fut]
                try:
                    cat, reason, a_in, a_out = fut.result()
                except Exception as exc:
                    n_failed += 1
                    print(f"  article {aid}: {exc}")
                    continue
                in_tok  += a_in
                out_tok += a_out
                counts[cat] += 1
                done += 1
                pending.append((cat, reason or None, aid))
                if len(pending) >= BATCH:
                    flush(conn)
                if tqdm:
                    cost = in_tok * INPUT_COST + out_tok * OUTPUT_COST
                    pbar.set_postfix_str(f"${cost:.2f} | {n_failed} failed")
            flush(conn)

    cost = in_tok * INPUT_COST + out_tok * OUTPUT_COST
    print(f"\nDone. {done} updated, {n_failed} failed.")
    kept = counts.get("real news", 0)
    moved = done - kept
    print(f"  kept 'real news': {kept}  |  reclassified to other: {moved}")
    for c, n in sorted(counts.items()):
        if n:
            print(f"    {c:<20} {n}")
    print(f"  {MODEL}: {in_tok/1e6:.2f}M in + {out_tok/1e6:.2f}M out ~ ${cost:.2f}")


if __name__ == "__main__":
    main()
