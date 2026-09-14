"""Command-line entry point.

Each build stage adds its own subcommand here, so there is always one place to exercise whatever
has been built so far.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from self_rag.config import Settings, get_settings


def _print_config(settings: Settings) -> None:
    fields = settings.model_dump()
    width = max(len(name) for name in fields)
    print("configuration")
    for name in sorted(fields):
        print(f"  {name:<{width}}  {fields[name]!r}")

    derived = {
        "markdown_dir": settings.markdown_dir,
        "parent_store_dir": settings.parent_store_dir,
        "qdrant_path": settings.qdrant_path,
        "checkpoint_db_path": settings.checkpoint_db_path,
    }
    print("\nderived paths")
    for name, value in derived.items():
        print(f"  {name:<{width}}  {value}")

    key = settings.api_key_for_provider()
    status = "present" if key and key.get_secret_value().strip() else "MISSING"
    print(f"\ncredentials\n  api key for {settings.llm_provider}: {status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self-rag",
        description="An agentic, self-correcting RAG system.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("config", help="Print the resolved configuration and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config":
        _print_config(get_settings())
    return 0
