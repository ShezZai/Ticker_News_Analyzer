import argparse
import asyncio
from dataclasses import replace

from .config import Settings
from .pipeline import run


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV-driven news article scraper.")
    parser.add_argument("--csv", required=True, help="Path to the articles CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N rows.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-process URLs even if already stored ok.")
    parser.add_argument("--ignore-robots", action="store_true", help="Skip robots.txt checks.")
    parser.add_argument("--concurrency", type=int, default=None, help="Worker count override.")
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if args.ignore_robots:
        settings = replace(settings, respect_robots=False)
    if args.concurrency:
        settings = replace(settings, concurrency=args.concurrency)
    return settings


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = build_settings(args)
    asyncio.run(run(args.csv, settings, limit=args.limit, retry_errors=args.retry_errors))
