"""Pre-warm the upstream pipeline stages for the eval article set so verdict
evals run only the sentiment call.

For each article it runs embed -> classify -> tag -> insights (the SAME service
stage adapters the eval uses, so the warmed outputs match exactly) and commits
to Postgres. Each stage is idempotent: it no-ops when its output already exists,
so re-running is cheap. Once warmed, `ticker-news eval pipeline --skip-stages all`
finds every upstream output present and fires ONLY the sentiment verdict per
article (no insights extraction or embedding mid-eval).

Work is parallelised across a thread pool; each worker owns its own psycopg
connection (sync connections are not shareable across threads). The TagContext
is built once and shared (read-only). Gemini calls are bounded by the shared
in-process rate limiter, so raising --workers past the limiter buys little.

Run (smoke set):   .venv\\Scripts\\python.exe prewarm_eval_upstream.py --ids 613,623,645
Run (whole 400):   .venv\\Scripts\\python.exe prewarm_eval_upstream.py
Force finegrained reclassify: add --reclassify
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg

from ticker_news.evals.pipeline_eval import connect_eval
from ticker_news.service import stages
from ticker_news.shared.config import get_settings

DEFAULT_CSV = r"C:\Agents\final-project\langfuse_datasets\400-e2e.filled.csv"


def load_ids_from_csv(path: str) -> list[int]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [int(r["article_id"]) for r in csv.DictReader(fh, delimiter="\t")]


def load_urls(ids: list[int]) -> dict[int, str]:
    with psycopg.connect(get_settings().database_url) as conn:
        rows = conn.execute(
            "SELECT id, url FROM public.articles WHERE id = ANY(%s)", (ids,)
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def prewarm_one(url: str, tag_ctx, reclassify: bool) -> tuple[str, str | None]:
    """Run the four upstream stages for one article on a private connection."""
    conn = connect_eval()  # registers pgvector; own connection per worker
    try:
        if reclassify:
            conn.execute(
                "UPDATE public.articles SET category = NULL, category_reason = NULL, "
                "is_act = NULL WHERE url = %s",
                (url,),
            )
            conn.commit()
        stages.embed_stage(conn, url)            # OpenAI embed (skips if present)
        stages.classify_stage(conn, url)         # Gemini finegrained (skips if set)
        stages.tag_stage(conn, url, tag_ctx)     # no LLM (skips if primary set)
        stages.insights_stage(conn, url, tag_ctx)  # Gemini boxes + OpenAI embed
        return url, None
    except Exception as exc:  # noqa: BLE001 - keep going; report per article
        return url, repr(exc)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="CSV with an article_id column (tab-separated).")
    ap.add_argument("--ids", help="Comma-separated article ids (overrides --csv).")
    ap.add_argument("--workers", type=int, default=8, help="Thread-pool size (Gemini limiter is the real ceiling).")
    ap.add_argument("--reclassify", action="store_true",
                    help="Clear category/is_act first so classify re-runs with the finegrained classifier.")
    args = ap.parse_args()

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = load_ids_from_csv(args.csv)
    urls = load_urls(ids)
    missing = sorted(set(ids) - set(urls))
    if missing:
        print(f"WARNING: {len(missing)} id(s) not in DB, skipping: {missing[:10]}", file=sys.stderr)

    setup = connect_eval()
    try:
        tag_ctx = stages.TagContext.load(setup)
    finally:
        setup.close()

    total = len(urls)
    print(f"Pre-warming {total} article(s) with {args.workers} worker(s)"
          f"{' [reclassify]' if args.reclassify else ''} ...")
    start = time.monotonic()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(prewarm_one, url, tag_ctx, args.reclassify): aid
                for aid, url in urls.items()}
        for fut in as_completed(futs):
            aid = futs[fut]
            url, err = fut.result()
            if err:
                failed += 1
                print(f"  art {aid}: {err}", file=sys.stderr)
            else:
                done += 1
            if (done + failed) % 25 == 0 or (done + failed) == total:
                el = time.monotonic() - start
                print(f"  {done + failed}/{total}  (ok={done} failed={failed}, {el:.0f}s)")

    el = time.monotonic() - start
    print(f"done: {done} warmed, {failed} failed, {total} total in {el:.0f}s "
          f"({el / total:.1f}s/article). Verdict evals can now run --skip-stages all.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
