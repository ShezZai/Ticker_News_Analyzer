"""Seed a Langfuse dataset for sentiment-verdict evaluation from the filled
400-article E2E CSV (langfuse_datasets/400-e2e.filled.csv).

Each dataset item is keyed on the article id (bodies are read from the DB at
eval time, never stored in the dataset -- same contract as the existing
`140-articles-*` datasets). The expected output is the set of ACCEPTABLE
verdicts under the +-0.3% deadband rule, so a grader marks a run's
buy/sell/hold verdict correct iff it is in that set:

    expected_output = {
        "acceptable_verdicts": ["buy"] | ["sell"] | ["buy","hold"] | ["sell","hold"],
        "gain_pct": <realized published->close move, %>,
    }

Idempotent: re-running upserts the same items (no duplicates) via the project's
upsert_dataset_items helper.

Run:  .venv\\Scripts\\python.exe seed_sentiment_dataset.py
"""

from __future__ import annotations

import csv
import json
from datetime import timezone

import psycopg

from ticker_news.evals.pipeline_eval import upsert_dataset_items
from ticker_news.shared import observability as obs
from ticker_news.shared.config import get_settings

CSV_PATH = r"C:\Agents\final-project\langfuse_datasets\400-e2e.filled.csv"
DATASET_NAME = "400-articles-sentiment-verdict"
DESCRIPTION = (
    "400 AI-compute articles for sentiment-panel evaluation. Input = {article_id} "
    "(body read from the DB at eval time). expected_output.acceptable_verdicts is "
    "the set of correct buy/sell/hold calls under a +-0.3% deadband on the realized "
    "published->market-close move (price_at_published = last trade as of publication; "
    "price_at_market_close = the regular-session close after publication, rolled to "
    "the next session for after-hours/weekend news). Inside the deadband both the "
    "gain-direction call AND hold are acceptable; only the opposite-direction call is "
    "wrong. Prices from Massive; see fill_400_e2e_dataset.py."
)


def build_items() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    ids = [int(r["article_id"]) for r in rows]

    # The pipeline-eval task drives stages by URL and the evaluators read
    # published_utc, so the item input must carry url + published_utc + title
    # (same shape as pipeline_eval.build_items), not just article_id.
    with psycopg.connect(get_settings().database_url) as conn:
        db = {
            r[0]: (r[1], r[2], r[3])
            for r in conn.execute(
                "SELECT id, url, published_utc, title FROM public.articles "
                "WHERE id = ANY(%s)",
                (ids,),
            ).fetchall()
        }
    missing = sorted(set(ids) - set(db))
    if missing:
        raise SystemExit(f"article ids not found in DB: {missing}")

    items: list[dict] = []
    for r in rows:
        aid = int(r["article_id"])
        url, published, title = db[aid]
        pub = float(r["price_at_published"])
        close = float(r["price_at_market_close"])
        gain = round((close - pub) / pub * 100.0, 4)
        items.append({
            "input": {
                "article_id": aid,
                "url": url,
                "published_utc": published.astimezone(timezone.utc).isoformat(),
                "title": title or "",
            },
            "expected_output": {
                "acceptable_verdicts": json.loads(r["expected_output"]),
                "gain_pct": gain,
            },
            "metadata": {
                "main_ticker": r["main ticker"],
                "published_utc": r["publishe_datetime"],
                "price_at_published": pub,
                "price_at_market_close": close,
                "deadband_pct": 0.3,
                "source": "400-e2e.filled.csv",
            },
        })
    return items


def main() -> int:
    client = obs.client()
    if client is None:
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are required to seed a dataset."
        )
    items = build_items()
    try:
        client.create_dataset(name=DATASET_NAME, description=DESCRIPTION)
    except Exception:  # noqa: BLE001 - already exists is fine
        pass
    upsert_dataset_items(client, DATASET_NAME, items)
    obs.flush()

    dataset = client.get_dataset(DATASET_NAME)
    print(f"dataset '{DATASET_NAME}': {len(dataset.items)} items "
          f"(seeded {len(items)} from {CSV_PATH}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
