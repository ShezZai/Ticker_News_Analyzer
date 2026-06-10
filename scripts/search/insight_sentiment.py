#!/usr/bin/env python3
"""Buy/sell/hold sentiment for a stock at the moment a given article published.

Given an article id and a ticker, this reproduces the
``search_articles_by_insights.py --like <id> --months-before N --k K --exclusive``
retrieval (no lookahead past the seed's publish time), assembles a context of:

  1. the full seed article (title + content),
  2. the insight boxes extracted from that article, and
  3. every retrieved related insight from PRIOR coverage, ordered earliest->latest,

then asks Gemini for an immediate trading sentiment on the ticker -- BUY, SELL,
or HOLD -- with a confidence in [0, 1] and a short justification grounded only in
the supplied material (it must not use knowledge of what happened afterwards).

If ``--ticker`` is given, only that ticker is judged; otherwise every ticker the
article names (primary + more_tickers) gets its own verdict in one model call --
the same article can be bullish for one and bearish for another.

Usage:
    python insight_sentiment.py <article_id>              # all of the article's tickers
    python insight_sentiment.py 4235 --ticker INTC        # just one
    python insight_sentiment.py 4235 --months-before 3 --k 5
    python insight_sentiment.py 4235 --no-exclusive --show-context
    python insight_sentiment.py 4235 --retrieval super              # hybrid retriever
    python insight_sentiment.py 4235 --retrieval super --budget 30
    python insight_sentiment.py 4235 --model gemini-2.5-flash-lite --json

Requires GOOGLE_API_KEY (Gemini) in env / .env. The article + insight data come
from NEWS_DB_DSN / DATABASE_URL (default dbname=news) -- point this at the
content-rich database, or the seed article will have no title/insights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# search_articles_by_insights lives next to this file and also wires up the
# embedding import path; reuse its retrieval API verbatim.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_articles_by_insights import (  # noqa: E402
    InsightHit, SeedInsight, _seed_window, get_conn, insights_of, search_by_insights,
)
from hybrid_retrieval import (  # noqa: E402
    DEF_BUDGET, DEF_NET_K, DEF_NET_MIN_SIM, DEF_TAU_INS, retrieve,
)

load_dotenv()

MARKET_TZ = ZoneInfo("America/New_York")
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_TIMEOUT_MS = 120_000
MAX_ARTICLE_CHARS = 16_000
MAX_RELATED = 80  # safety cap on retrieved insights folded into the prompt

DEFAULT_MONTHS_BEFORE = 3
DEFAULT_K = 5
DEFAULT_MIN_SIMILARITY = 0.7
DEFAULT_RETRIEVAL = "insight"
RETRIEVALS = ("insight", "super")
ACTIONS = ["buy", "sell", "hold"]

PROMPT_INSTRUCTIONS = """You are an equity analyst reacting to breaking news in real time.

TIMELINE -- this is strict:
- "THE ARTICLE" below is breaking RIGHT NOW, at {pub} (ET). It is the ONLY new
  information and it is the event you must react to.
- "INSIGHTS FROM THIS ARTICLE" are points pulled from that same article -- also
  now.
- "RELATED PRIOR INSIGHTS" are entirely in the PAST: every one is dated before
  now, was already public, and is already reflected in the current price. They
  are background only -- NOT new news, and NOT something to trade on by itself.

Your task: for EACH ticker listed below, give a sentiment on the IMMEDIATE market
reaction to THIS article -- how the stock should move right after this article
hits (next minutes to hours) if you enter the position now, at {pub}:
  {tickers}

Rules:
- React to the CURRENT article only. Use the prior insights solely to judge how
  new or surprising the article is versus what was already known/expected (more
  surprise = bigger immediate move; already-anticipated news barely moves a
  stock). Never treat the past insights as a fresh catalyst.
- You have NO knowledge of anything at or after {pub}. No hindsight whatsoever.
- Judge each ticker on its OWN merits: one article can be bullish for one ticker
  and bearish for another (e.g. a rival's good news). If the article is barely
  relevant to a ticker, "hold" with low confidence is appropriate.
- For each ticker return one action: "buy" (expect an immediate rise), "sell"
  (expect an immediate fall, i.e. short / avoid), or "hold" (no clear edge); a
  confidence in [0, 1]; and a 2-4 sentence justification grounded in the article.

Return ONLY a JSON array with one object per ticker, in the same order:
[{{"ticker": "...", "action": "...", "confidence": 0.0, "justification": "..."}}]
"""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_article(conn, article_id: int) -> dict:
    """Fetch the seed article's title/content/date/tickers, or raise."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, content, published_utc, primary_ticker, more_tickers, "
            "       category "
            "FROM public.articles WHERE id = %s",
            (article_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"article {article_id} not found in the connected database.")
    return {
        "id": row[0], "title": row[1], "content": row[2], "published_utc": row[3],
        "primary_ticker": row[4], "more_tickers": list(row[5] or []), "category": row[6],
    }


def gather_related(
    conn, article_id: int, seed_pub, months_before: Optional[int],
    k: int, exclusive: bool, min_similarity: float,
) -> List[InsightHit]:
    """Retrieve related prior insights, deduped and sorted earliest->latest.

    Mirrors the CLI's window logic: search_by_insights over the months-before
    window anchored on the seed's publish time (exclusive => strict, no lookahead).
    """
    since, until = _seed_window(seed_pub, months_before, exclusive)
    groups = search_by_insights(
        article_id=article_id, k=k, since=since, until=until,
        min_similarity=min_similarity, until_exclusive=exclusive, conn=conn,
    )
    # an insight may surface under several seed insights; keep its best score
    best: dict[int, InsightHit] = {}
    for g in groups:
        for h in g.hits:
            cur = best.get(h.insight_id)
            if cur is None or h.similarity > cur.similarity:
                best[h.insight_id] = h
    hits = sorted(best.values(), key=lambda h: (h.published_utc is None, h.published_utc))
    return hits[:MAX_RELATED]


def _load_insight_hits(conn, scored) -> List[InsightHit]:
    """Hydrate (insight_id, article_id, score) tuples into full InsightHit rows.

    The hybrid retriever returns only ids + a per-insight score; join back to
    article_insights/articles for the topic, text, tickers and date the prompt
    needs. ``similarity`` carries whatever score the method produced.
    """
    score_by_id = {iid: sc for iid, _aid, sc in scored}
    if not score_by_id:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ai.id, ai.article_id, ai.source_url, ai.article_headline, "
            "       ai.topic, ai.insight, ai.box_text, a.tickers, a.published_utc "
            "FROM public.article_insights ai "
            "JOIN public.articles a ON a.id = ai.article_id "
            "WHERE ai.id = ANY(%s)",
            (list(score_by_id),),
        )
        rows = cur.fetchall()
    hits = [
        InsightHit(
            insight_id=r[0], article_id=r[1], source_url=r[2], article_headline=r[3],
            topic=r[4], insight=r[5], box_text=r[6], tickers=list(r[7] or []),
            published_utc=r[8], similarity=float(score_by_id.get(r[0], 0.0)),
        )
        for r in rows
    ]
    hits.sort(key=lambda h: (h.published_utc is None, h.published_utc))
    return hits[:MAX_RELATED]


def gather_related_super(
    conn, article_id: int, months_before: Optional[int], k: int,
    exclusive: bool, min_similarity: float, net_k: int, net_min_sim: float,
    tau_ins: float, budget: int,
) -> List[InsightHit]:
    """Retrieve related prior insights via the two-stage 'super' hybrid retriever.

    Uses ``hybrid_retrieval.retrieve(method="super")`` -- a WIDE whole-article
    net gated by an insight-cosine threshold, then RRF reranked -- and hydrates
    the kept insight ids into InsightHit rows, sorted earliest->latest.
    """
    res = retrieve(
        article_id, method="super", months_before=months_before,
        exclusive=exclusive, k=k, min_sim=min_similarity, net_k=net_k,
        net_min_sim=net_min_sim, tau_ins=tau_ins, budget=budget, conn=conn,
    )
    return _load_insight_hits(conn, res.scored)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def _fmt_et(dt) -> str:
    return dt.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M") if dt else "(unknown)"


def build_prompt(targets: List[str], article: dict, seeds: List[SeedInsight],
                 related: List[InsightHit]) -> str:
    pub = _fmt_et(article["published_utc"])
    parts = [PROMPT_INSTRUCTIONS.format(tickers=", ".join(targets), pub=pub)]

    parts.append(f"\n===== THE ARTICLE -- breaking NOW at {pub} ET (react to this) =====")
    parts.append(f"TICKERS TO JUDGE: {', '.join(targets)}")
    parts.append(f"TITLE: {article['title'] or '(no title)'}")
    parts.append("CONTENT:")
    parts.append((article["content"] or "(no content)")[:MAX_ARTICLE_CHARS])

    parts.append("\n===== INSIGHTS FROM THIS ARTICLE (also now) =====")
    if seeds:
        for i, s in enumerate(seeds, 1):
            parts.append(f"[{i}] {s.box_text or ('TOPIC: ' + (s.topic or ''))}")
    else:
        parts.append("(none)")

    parts.append("\n===== RELATED PRIOR INSIGHTS -- PAST, already known/priced "
                 "(earliest -> latest) =====")
    if related:
        for h in related:
            when = _fmt_et(h.published_utc)
            tick = ",".join(h.tickers) if h.tickers else "-"
            parts.append(
                f"- {when} | [{tick}] | sim={h.similarity:.3f} | "
                f"TOPIC: {h.topic or '(no topic)'}"
            )
            if h.insight:
                parts.append(f"    {h.insight}")
            parts.append(f"    from a#{h.article_id}: {h.article_headline or '(no headline)'}")
    else:
        parts.append("(none retrieved in the window)")

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def ask_gemini(prompt: str, model: str, retries: int = 3) -> List[dict]:
    """Send the prompt and parse the per-ticker [{ticker, action, confidence, ...}]."""
    from google import genai
    from google.genai import types

    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set (put it in .env).")
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema={
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "ticker": {"type": "STRING"},
                    "action": {"type": "STRING", "enum": ACTIONS},
                    "confidence": {"type": "NUMBER"},
                    "justification": {"type": "STRING"},
                },
                "required": ["ticker", "action", "confidence", "justification"],
            },
        },
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )
    last = "no response"
    for attempt in range(retries):
        try:
            data = json.loads(resp.text) if (resp := client.models.generate_content(
                model=model, contents=prompt, config=config)).text else None
            if isinstance(data, list) and data and all(
                isinstance(d, dict) and d.get("action") in ACTIONS for d in data
            ):
                return data
            last = f"unexpected payload: {str(data)[:120]}"
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise SystemExit(f"Gemini did not return a usable answer: {last}")


def _role(article: dict, ticker: str) -> str:
    return "primary" if ticker.upper() == (article["primary_ticker"] or "").upper() else "mentioned"


def _annotate(verdicts: List[dict], article: dict) -> List[dict]:
    """Tag each verdict with its primary/mentioned role (in place)."""
    for v in verdicts:
        v["role"] = _role(article, str(v.get("ticker", "")))
    return verdicts


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("article_id", type=int, help="seed article id")
    p.add_argument("--ticker", default=None,
                   help="ticker to judge (e.g. INTC); if omitted, judge ALL of the "
                        "article's tickers (primary + more_tickers)")
    p.add_argument("--months-before", type=int, default=DEFAULT_MONTHS_BEFORE,
                   help=f"only use insights from N months before the article "
                        f"(default {DEFAULT_MONTHS_BEFORE}; 0/None for open-ended)")
    p.add_argument("-k", "--k", type=int, default=DEFAULT_K,
                   help=f"top-k related insights per seed insight (default {DEFAULT_K})")
    p.add_argument("--exclusive", action=argparse.BooleanOptionalAction, default=True,
                   help="anchor the window strictly on the seed's exact publish time "
                        "(no same-day lookahead); on by default")
    p.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY,
                   help=f"drop related insights below this cosine sim "
                        f"(default {DEFAULT_MIN_SIMILARITY})")
    p.add_argument("--retrieval", choices=RETRIEVALS, default=DEFAULT_RETRIEVAL,
                   help="related-insight retriever: 'insight' = per-seed-insight "
                        "ANN over article_insights (the original); 'super' = the "
                        "two-stage whole-article-net + insight-rerank hybrid "
                        f"(default {DEFAULT_RETRIEVAL})")
    p.add_argument("--net-k", type=int, default=DEF_NET_K,
                   help=f"[--retrieval super] whole-article net size "
                        f"(default {DEF_NET_K})")
    p.add_argument("--net-min-similarity", type=float, default=DEF_NET_MIN_SIM,
                   help=f"[--retrieval super] cosine floor for the wide article "
                        f"net (default {DEF_NET_MIN_SIM})")
    p.add_argument("--tau-insight", type=float, default=DEF_TAU_INS,
                   help=f"[--retrieval super] keep net insights with max cosine-to-"
                        f"seed >= this (default {DEF_TAU_INS})")
    p.add_argument("--budget", type=int, default=DEF_BUDGET,
                   help=f"[--retrieval super] how many reranked insights to keep "
                        f"(default {DEF_BUDGET})")
    p.add_argument("--model", default=GEMINI_MODEL, help=f"Gemini model (default {GEMINI_MODEL})")
    p.add_argument("--show-context", action="store_true",
                   help="print the full prompt context sent to the model")
    p.add_argument("--json", action="store_true", help="print only the JSON result")
    p.add_argument("--top-2", "--top_2", dest="top_2", action="store_true",
                   help="judge ALL the article's tickers, then re-run the 2 highest-"
                        "confidence non-hold tickers each on its own; print one "
                        "consolidated JSON from the 3 calls")
    args = p.parse_args()

    conn = get_conn()
    try:
        article = load_article(conn, args.article_id)
        if not article["title"]:
            print("warning: seed article has no title -- NEWS_DB_DSN may point at the "
                  "content-poor DB; results will be meaningless.", file=sys.stderr)
        if article["published_utc"] is None:
            raise SystemExit("seed article has no published_utc to anchor the window.")

        # every ticker the article names (primary first, then more_tickers), deduped
        seen: set[str] = set()
        all_targets = [
            t.upper() for t in ([article["primary_ticker"]] + article["more_tickers"])
            if t and not (t.upper() in seen or seen.add(t.upper()))
        ]
        # --top-2 always judges everything first; otherwise honor an explicit --ticker
        if args.ticker and not args.top_2:
            targets = [args.ticker.strip().upper()]
        else:
            targets = all_targets
            if not targets:
                raise SystemExit("article names no tickers; pass --ticker explicitly.")
            if args.top_2 and len(targets) < 3:
                print(
                    f"warning: --top-2 requires ≥ 3 tickers; article has {len(targets)} "
                    f"-- running separate calls per ticker instead.",
                    file=sys.stderr,
                )
                args.top_2 = False

        seeds = insights_of(conn, args.article_id)
        if args.retrieval == "super":
            related = gather_related_super(
                conn, args.article_id, args.months_before, args.k,
                args.exclusive, args.min_similarity, args.net_k,
                args.net_min_similarity, args.tau_insight, args.budget,
            )
        else:
            related = gather_related(
                conn, args.article_id, article["published_utc"],
                args.months_before, args.k, args.exclusive, args.min_similarity,
            )
    finally:
        conn.close()

    def judge(tks: List[str]) -> List[dict]:
        return _annotate(
            ask_gemini(build_prompt(tks, article, seeds, related), args.model), article
        )

    if args.show_context:
        print(build_prompt(targets, article, seeds, related))
        print("\n" + "=" * 70 + "\n")

    # --top-2: one joint call, then re-run the 2 best non-hold tickers separately.
    if args.top_2:
        joint = judge(targets)
        ranked = sorted(
            (v for v in joint if v["action"] != "hold"),
            key=lambda v: float(v["confidence"]), reverse=True,
        )
        picks = ranked[:2]
        separate = [s for v in picks for s in judge([v["ticker"].upper()])]
        out = {
            "article_id": article["id"],
            "published_et": _fmt_et(article["published_utc"]),
            "title": article["title"],
            "context": {"article_insights": len(seeds), "related_insights": len(related)},
            "tickers_judged": targets,
            "top2_tickers": [v["ticker"].upper() for v in picks],
            "joint": joint,
            "top2_separate": separate,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # fewer than 3 tickers with no explicit --ticker: one call per ticker
    if not args.ticker and len(targets) < 3:
        verdicts = []
        for tk in targets:
            verdicts.extend(judge([tk]))
    else:
        verdicts = judge(targets)

    if args.json:
        print(json.dumps(verdicts, ensure_ascii=False, indent=2))
        return

    by_ticker = {str(v.get("ticker", "")).upper(): v for v in verdicts}
    print(f"\nSeed: a#{article['id']} {_fmt_et(article['published_utc'])} ET")
    print(f"  {article['title'] or '(no title)'}")
    print(f"  context: {len(seeds)} article insight(s) + {len(related)} related prior "
          f"insight(s) [retrieval={args.retrieval}]")
    print(f"  judging {len(targets)} ticker(s): {', '.join(targets)}")
    for tk in targets:
        v = by_ticker.get(tk)
        if v is None:
            print(f"\n  {tk}: (no verdict returned)")
            continue
        print(f"\n  {tk} ({v.get('role', '?')})")
        print(f"    ACTION:     {v['action'].upper()}")
        print(f"    CONFIDENCE: {float(v['confidence']):.2f}")
        print(f"    WHY:        {v['justification']}")


if __name__ == "__main__":
    main()
