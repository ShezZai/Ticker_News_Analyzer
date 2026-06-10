#!/usr/bin/env python3
"""Second-pass refinement of a --top-2 sentiment run.

Takes the consolidated JSON emitted by ``insight_sentiment.py --top-2`` and, for
each of the top-2 separate verdicts whose confidence is below a threshold
(default 0.8), re-asks the model -- but with MORE context:

  * the original context (the full current article, its insight boxes, and the
    related prior insights), exactly as before, PLUS
  * up to ``--just-k`` (default 5) PAST insights most similar (cosine > 0.75) to
    the model's OWN earlier justification text,

and the prompt now reminds the model: "you stated about all this that <...>" and
presents those retrieved rows as "the past insights related to it". The model
then re-evaluates the immediate-reaction buy/sell verdict for that ticker.

High-confidence verdicts (>= threshold) are passed through unchanged. The result
is one consolidated JSON with each ticker's original verdict, the extra insights
used, and the refined verdict.

Usage:
    insight_sentiment.py 4125 --top-2 | followup_sentiment.py
    followup_sentiment.py top2.json
    followup_sentiment.py top2.json --threshold 0.8 --just-k 5

Requires GOOGLE_API_KEY (Gemini) + OPENAI_API_KEY (embeddings) and
NEWS_DB_DSN / DATABASE_URL (point at the content-rich DB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insight_sentiment import (  # noqa: E402
    DEFAULT_K, DEFAULT_MIN_SIMILARITY, DEFAULT_MONTHS_BEFORE, GEMINI_MODEL,
    _fmt_et, ask_gemini, build_prompt, gather_related, load_article,
)
from search_articles_by_insights import (  # noqa: E402
    InsightHit, _seed_window, get_conn, insights_of, search_insights,
)

CONF_THRESHOLD = 0.8
JUST_K = 5
JUST_MIN_SIM = 0.75


def read_top2(path: Optional[str]) -> dict:
    """Load a --top-2 output from a file (or stdin when path is None/'-')."""
    text = sys.stdin.read() if path in (None, "-") else open(path).read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {exc}")
    if "article_id" not in data or "top2_separate" not in data:
        raise SystemExit("input is not a --top-2 output (need article_id + top2_separate).")
    return data


def justification_insights(
    conn, justification: str, seed_id: int, since, until, exclusive: bool,
    k: int, min_sim: float,
) -> List[InsightHit]:
    """Past insights most similar to the justification, excluding the seed article."""
    # over-fetch so dropping the seed's own (likely very similar) insights still
    # leaves up to k rows; restrict to the pre-article window (no lookahead).
    hits = search_insights(
        justification, k=k + 10, since=since, until=until,
        until_exclusive=exclusive, min_similarity=min_sim, conn=conn,
    )
    hits = [h for h in hits if h.article_id != seed_id][:k]
    hits.sort(key=lambda h: (h.published_utc is None, h.published_utc))  # earliest -> latest
    return hits


def build_followup_prompt(ticker, article, seeds, related, prior, extra) -> str:
    """Former context + a reflexive 'you said X; here is related past' addendum."""
    parts = [build_prompt([ticker], article, seeds, related)]
    parts.append(f"\n===== YOUR EARLIER ASSESSMENT OF {ticker} =====")
    parts.append("You stated about all this that:")
    parts.append(f"\"{prior['justification']}\"")
    parts.append(f"(your earlier call: {str(prior['action']).upper()} at confidence "
                 f"{float(prior['confidence']):.2f})")

    parts.append("\n===== PAST INSIGHTS RELATED TO WHAT YOU SAID -- already known "
                 "(earliest -> latest) =====")
    if extra:
        for h in extra:
            when = _fmt_et(h.published_utc)
            tick = ",".join(h.tickers) if h.tickers else "-"
            parts.append(f"- {when} | [{tick}] | sim={h.similarity:.3f} | "
                         f"TOPIC: {h.topic or '(no topic)'}")
            if h.insight:
                parts.append(f"    {h.insight}")
            parts.append(f"    from a#{h.article_id}: {h.article_headline or '(no headline)'}")
    else:
        parts.append("(none found above the similarity threshold)")

    parts.append(
        f"\nNow RE-EVALUATE the immediate-reaction verdict for {ticker} for the SAME "
        f"current article, taking this additional PAST context into account (it is "
        f"already-known background, not new news). Keep or change your action and "
        f"confidence as warranted. Return ONLY a JSON array with one object: "
        f'[{{"ticker": "{ticker}", "action": "...", "confidence": 0.0, "justification": "..."}}]'
    )
    return "\n".join(parts)


def _hit_json(h: InsightHit) -> dict:
    return {
        "article_id": h.article_id, "published_et": _fmt_et(h.published_utc),
        "tickers": h.tickers, "similarity": round(h.similarity, 3),
        "topic": h.topic, "insight": h.insight, "headline": h.article_headline,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", nargs="?", default=None,
                   help="--top-2 JSON file (default: read stdin)")
    p.add_argument("--threshold", type=float, default=CONF_THRESHOLD,
                   help=f"re-run a verdict when its confidence is below this "
                        f"(default {CONF_THRESHOLD})")
    p.add_argument("--months-before", type=int, default=DEFAULT_MONTHS_BEFORE,
                   help="months-before window for the FORMER context (match the top-2 run)")
    p.add_argument("-k", "--k", type=int, default=DEFAULT_K,
                   help="top-k related insights per seed insight for the former context")
    p.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True,
                   help="strict pre-article window (match the top-2 run); on by default")
    p.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY,
                   help="min cosine for the former related insights")
    p.add_argument("--just-k", type=int, default=JUST_K,
                   help=f"how many justification-similar past insights to add (default {JUST_K})")
    p.add_argument("--just-min-similarity", type=float, default=JUST_MIN_SIM,
                   help=f"min cosine for justification-similar insights (default {JUST_MIN_SIM})")
    p.add_argument("--model", default=GEMINI_MODEL, help=f"Gemini model (default {GEMINI_MODEL})")
    p.add_argument("--show-context", action="store_true",
                   help="print each re-run prompt to stderr")
    args = p.parse_args()

    data = read_top2(args.input)
    article_id = int(data["article_id"])

    conn = get_conn()
    try:
        article = load_article(conn, article_id)
        if article["published_utc"] is None:
            raise SystemExit("seed article has no published_utc to anchor the window.")
        seeds = insights_of(conn, article_id)
        related = gather_related(
            conn, article_id, article["published_utc"],
            args.months_before, args.k, args.exclusive, args.min_similarity,
        )
        since, until = _seed_window(article["published_utc"], args.months_before, args.exclusive)

        refined = []
        for v in data["top2_separate"]:
            ticker = str(v["ticker"]).upper()
            original = {key: v[key] for key in ("action", "confidence", "justification")}
            entry = {"ticker": ticker, "original": original}

            if float(v["confidence"]) >= args.threshold:
                entry["reran"] = False
                entry["verdict"] = original
                refined.append(entry)
                continue

            extra = justification_insights(
                conn, v["justification"], article_id, since, until,
                args.exclusive, args.just_k, args.just_min_similarity,
            )
            prompt = build_followup_prompt(ticker, article, seeds, related, v, extra)
            if args.show_context:
                print(prompt, file=sys.stderr)
                print("\n" + "=" * 70 + "\n", file=sys.stderr)
            new = ask_gemini(prompt, args.model)
            nv = new[0] if new else None
            entry["reran"] = True
            entry["extra_insights"] = [_hit_json(h) for h in extra]
            entry["verdict"] = (
                {key: nv[key] for key in ("action", "confidence", "justification")}
                if nv else None
            )
            refined.append(entry)
    finally:
        conn.close()

    out = {
        "article_id": article_id,
        "published_et": _fmt_et(article["published_utc"]),
        "title": article["title"],
        "threshold": args.threshold,
        "refined": refined,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
