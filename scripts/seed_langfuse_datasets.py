"""Seed two Langfuse datasets from CSV ground-truth files."""
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv(Path(__file__).parent.parent / ".env")

lf = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
)

DATASETS = [
    {
        "name": "140-articles-act-no-act",
        "description": "140 articles labelled YES/NO for actionability (ACT / NO-ACT classification).",
        "csv": Path(
            r"C:\Agents\final-project\langfuse_datasets"
            r"\140-articles-dataset-articles-act-no-act.csv"
        ),
        "input_col": "article_id",
        "output_col": "descision",
    },
    {
        "name": "140-articles-categories",
        "description": "140 articles with ground-truth fine-grained category labels.",
        "csv": Path(
            r"C:\Agents\final-project\langfuse_datasets"
            r"\140-articles-dataset-articles-categories.csv"
        ),
        "input_col": "article_id",
        "output_col": "category_gt",
    },
]


def seed(ds_cfg: dict) -> None:
    name = ds_cfg["name"]
    print(f"\n--- {name} ---")

    lf.create_dataset(name=name, description=ds_cfg["description"])
    print(f"Dataset created (or already exists): {name}")

    created = 0
    with open(ds_cfg["csv"], newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            article_id = row[ds_cfg["input_col"]].strip()
            expected = row[ds_cfg["output_col"]].strip()
            lf.create_dataset_item(
                dataset_name=name,
                id=f"{name}--{article_id}",
                input={"article_id": int(article_id)},
                expected_output={"label": expected},
            )
            created += 1

    lf.flush()
    print(f"Upserted {created} items into '{name}'.")


if __name__ == "__main__":
    for cfg in DATASETS:
        seed(cfg)
    print("\nDone.")
