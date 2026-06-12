#!/usr/bin/env python3
"""Distill insight boxes with the NEW two-label policy — to CSV, never the DB.

This is the validation-time twin of ``extract_insights.py``. It runs the tighter
``distill_insights_prompt.txt`` (emit only ``evidance-event`` / ``informative``;
drop promotion, analyst opinion, market-research and fluff) over a set of
articles and writes the result as a ground-truthable table that mirrors
``insights_GT.csv`` — so the distilled boxes can be judged side-by-side with the
hand-labeled set.

It reads articles from NEWS_DB_DSN read-only and DOES NOT touch the database:
no rows are written to ``article_insights``, no schema change, no embeddings.

Output columns (insights_GT.csv-equivalent, plus the model's ``label``):
    id, header, new_category GT, box_index, label, topic, insight, quotes,
    insight_GT, url_link
Articles that yield zero boxes get one placeholder row (``box_index`` blank,
topic "(no insights extracted)") so coverage is visible. ``insight_GT`` is left
blank for you to fill.

Usage:
    python extract_insights_new.py                       # all ids in classification_GT.csv
    python extract_insights_new.py --limit 15            # smoke test
    python extract_insights_new.py --ids 20074,9491,7943
    python extract_insights_new.py --gate                # + strict per-box second pass
    python extract_insights_new.py --workers 12 --out /tmp/distilled.csv

Requires GOOGLE_API_KEY in env/.env. Connection from NEWS_DB_DSN / DATABASE_URL.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"

GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"   # sturdier on 504s
# The second-pass gate defaults to the stronger model: per-box it hits flash-level
# precision (~85%) without flash's whole-article over-gating, since each box is
# judged alone. flash-lite extract + flash gate measured best (84.6% / 92.6% recall).
GATE_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_MS = 120_000
MAX_ARTICLE_CHARS = 48_000
DEFAULT_QUOTE_THRESHOLD = 0.75

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_PROMPT_FILE = os.path.join(REPO, "docs", "validations", "distill_insights_prompt.txt")
DEFAULT_GT_CSV = os.path.join(REPO, "docs", "validations", "classification_GT.csv")
DEFAULT_OUT = os.path.join(REPO, "docs", "validations", "insights_GT_distilled.csv")

OUT_FIELDS = ["id", "header", "new_category GT", "Act GT", "box_index", "label",
              "topic", "insight", "quotes", "insight_GT", "url_link"]


# --------------------------------------------------------------------------- #
# Prompt (single source of truth = the .txt the validation was run against)
# --------------------------------------------------------------------------- #
def load_prompt(path: str) -> Tuple[str, List[str], Optional[str]]:
    """Exec the prompt file and pull PROMPT_TEMPLATE + LABELS (+ optional gate)."""
    ns: Dict[str, object] = {}
    with open(path) as fh:
        exec(fh.read(), ns)  # noqa: S102 - trusted local prompt definition
    template = ns.get("PROMPT_TEMPLATE")
    labels = ns.get("LABELS")
    gate = ns.get("GATE_PROMPT_TEMPLATE")
    if not isinstance(template, str) or not isinstance(labels, list):
        raise SystemExit(f"{path} must define PROMPT_TEMPLATE (str) and LABELS (list)")
    return template, list(labels), gate if isinstance(gate, str) else None


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
_client = None
_config = None


def load_gemini(labels: Sequence[str]):
    global _client, _config
    if _client is None:
        from google import genai
        from google.genai import types

        if not os.getenv("GOOGLE_API_KEY"):
            raise SystemExit("GOOGLE_API_KEY is not set (put it in .env).")
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        _config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "boxes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "label": {"type": "STRING", "enum": list(labels)},
                                "topic": {"type": "STRING"},
                                "insight": {"type": "STRING"},
                                "quotes": {"type": "ARRAY", "items": {"type": "STRING"}},
                            },
                            "required": ["label", "topic", "insight", "quotes"],
                        },
                    }
                },
                "required": ["boxes"],
            },
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
    return _client, _config


_gate_config = None


def load_gate_config(labels: Sequence[str]):
    """Config for the per-box second-pass gate (decision enum)."""
    global _gate_config
    if _gate_config is None:
        from google.genai import types
        _gate_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {"decision": {"type": "STRING",
                                            "enum": list(labels) + ["DROP"]}},
                "required": ["decision"],
            },
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
    return _gate_config


def gate_decision(client, gate_config, gate_template: str, box: dict,
                  retries: int = 4) -> str:
    """Re-judge one box; return 'evidance-event' | 'informative' | 'DROP'.

    Fail-OPEN: if the gate can't answer, keep the box (return its current label)
    rather than silently dropping real evidence on a transient API error.
    """
    prompt = gate_template.format(topic=box.get("topic", ""),
                                  insight=(box.get("insight") or "")[:600])
    model = GATE_MODEL
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt,
                                                   config=gate_config)
            return json.loads(resp.text or "{}").get("decision") or box.get("label")
        except Exception as exc:  # noqa: BLE001
            if model != GEMINI_FALLBACK_MODEL and _is_retryable(exc):
                model = GEMINI_FALLBACK_MODEL
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return box.get("label")  # fail-open: keep on persistent error


def _is_retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("deadline_exceeded", "504", "503", "unavailable",
                                "timeout", "timed out", "429", "resource_exhausted"))


def generate_boxes(client, config, template: str, article: str,
                   retries: int = 5) -> List[dict]:
    """Run the distill prompt over one article; return list of box dicts."""
    prompt = template.format(article=article[:MAX_ARTICLE_CHARS])
    model = GEMINI_MODEL
    last = "no response"
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            data = json.loads(resp.text or "{}")
            boxes = data.get("boxes")
            if isinstance(boxes, list):
                return [b for b in boxes if isinstance(b, dict)]
            last = "no boxes array"
            if model != GEMINI_FALLBACK_MODEL and attempt >= 1:
                model = GEMINI_FALLBACK_MODEL
        except Exception as exc:  # network / 5xx / parse
            last = repr(exc)
            if model != GEMINI_FALLBACK_MODEL and _is_retryable(exc):
                model = GEMINI_FALLBACK_MODEL
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"model did not answer after {retries} tries: {last}")


# --------------------------------------------------------------------------- #
# Verbatim quote recovery (ported from extract_insights.py)
# --------------------------------------------------------------------------- #
_WRAP_CHARS = " \t\"'“”‘’"


def _clean_quote(q: str) -> str:
    return q.strip().strip(_WRAP_CHARS).strip()


def fuzzy_find_in_source(quote: str, article: str,
                         threshold: float = DEFAULT_QUOTE_THRESHOLD) -> Optional[str]:
    q = _clean_quote(quote)
    if not q:
        return None
    if q in article:
        return q
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


def verbatimize_quotes(quotes, article, threshold=DEFAULT_QUOTE_THRESHOLD):
    out: List[str] = []
    seen = set()
    for q in quotes or []:
        match = fuzzy_find_in_source(str(q), article, threshold)
        if match and match not in seen:
            seen.add(match)
            out.append(match)
    return out


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_gt_rows(gt_csv: str) -> List[dict]:
    """Per-article metadata from classification_GT.csv (id, header, category, url)."""
    rows = []
    with open(gt_csv) as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "id": int(r["id"]),
                "header": r.get("header", ""),
                "new_category GT": r.get("new_category GT", ""),
                "Act GT": r.get("Act_GT", ""),
                "url_link": r.get("url_link", ""),
            })
    return rows


def fetch_content(conn, ids: Sequence[int]) -> Dict[int, Tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, content FROM public.articles WHERE id = ANY(%s)",
                    (list(ids),))
        return {r[0]: (r[1] or "", r[2] or "") for r in cur.fetchall()}


def main(argv: Optional[List[str]] = None) -> int:
    global GEMINI_MODEL, GATE_MODEL
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-csv", default=DEFAULT_GT_CSV,
                   help="source of ids + header/category/url (default classification_GT.csv)")
    p.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--ids", help="comma-separated article ids to restrict to")
    p.add_argument("--limit", type=int, help="process only the first N articles")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quote-threshold", type=float, default=DEFAULT_QUOTE_THRESHOLD)
    p.add_argument("--model", default=GEMINI_MODEL,
                   help=f"primary Gemini model (default {GEMINI_MODEL})")
    p.add_argument("--gate", action="store_true",
                   help="run a strict per-box second pass; drop boxes it rejects "
                        "(higher precision, ~same recall). Needs GATE_PROMPT_TEMPLATE.")
    p.add_argument("--gate-model", default=GATE_MODEL,
                   help=f"model for the --gate second pass (default {GATE_MODEL})")
    args = p.parse_args(argv)

    GEMINI_MODEL = args.model
    GATE_MODEL = args.gate_model

    template, labels, gate_template = load_prompt(args.prompt_file)
    if args.gate and not gate_template:
        raise SystemExit(f"--gate set but {args.prompt_file} has no GATE_PROMPT_TEMPLATE")
    gt = load_gt_rows(args.gt_csv)
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        gt = [r for r in gt if r["id"] in want]
    if args.limit:
        gt = gt[:args.limit]
    ids = [r["id"] for r in gt]

    print(f"distilling {len(ids)} articles via {GEMINI_MODEL} "
          f"(prompt: {os.path.relpath(args.prompt_file, REPO)}"
          f"{', +gate' if args.gate else ''}) — DB read-only, no writes")

    with psycopg.connect(DB_DSN) as conn:
        content = fetch_content(conn, ids)

    client, config = load_gemini(labels)
    gate_config = load_gate_config(labels) if args.gate else None
    n_gate_dropped = [0]  # mutable counter shared across threads (GIL-safe increments)

    def work(meta: dict):
        title, body = content.get(meta["id"], ("", ""))
        article = f"ARTICLE_HEADLINE: {title}\n\n{body}" if (title or body) else ""
        if not article.strip():
            return meta, [], "no-content"
        try:
            boxes = generate_boxes(client, config, template, article)
        except Exception as exc:
            return meta, [], f"ERROR: {exc}"
        # force quotes to verbatim source spans; keep label/topic/insight
        clean = []
        for b in boxes:
            clean.append({
                "label": (b.get("label") or "").strip(),
                "topic": (b.get("topic") or "").strip(),
                "insight": (b.get("insight") or "").strip(),
                "quotes": verbatimize_quotes(b.get("quotes"), body, args.quote_threshold),
            })
        # second-pass gate: re-judge each box; drop rejects, adopt the gate's tier
        if args.gate:
            kept = []
            for b in clean:
                decision = gate_decision(client, gate_config, gate_template, b)
                if decision == "DROP":
                    n_gate_dropped[0] += 1
                    continue
                b["label"] = decision  # gate also corrects evid<->info mislabels
                kept.append(b)
            clean = kept
        return meta, clean, "ok"

    results: Dict[int, Tuple[dict, List[dict], str]] = {}
    n_box = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, r) for r in gt]
        for i, fut in enumerate(as_completed(futs), 1):
            meta, boxes, status = fut.result()
            results[meta["id"]] = (meta, boxes, status)
            n_box += len(boxes)
            if i % 25 == 0 or i == len(gt):
                print(f"  {i}/{len(gt)} done ({n_box} boxes so far)")

    n_empty = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        for meta in gt:  # preserve classification_GT order
            _m, boxes, status = results[meta["id"]]
            base = {"id": meta["id"], "header": meta["header"],
                    "new_category GT": meta["new_category GT"],
                    "Act GT": meta["Act GT"],
                    "insight_GT": "", "url_link": meta["url_link"]}
            if not boxes:
                n_empty += 1
                note = "(no insights extracted)" if status in ("ok", "no-content") else status
                w.writerow({**base, "box_index": "", "label": "",
                            "topic": note, "insight": "", "quotes": ""})
                continue
            for bi, b in enumerate(boxes):
                w.writerow({**base, "box_index": bi, "label": b["label"],
                            "topic": b["topic"], "insight": b["insight"],
                            "quotes": "\n".join(b["quotes"])})

    print(f"wrote {args.out}")
    print(f"articles: {len(gt)} | insight boxes: {n_box} | "
          f"zero-box articles: {n_empty}")
    if args.gate:
        print(f"gate: dropped {n_gate_dropped[0]} boxes in the second pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
