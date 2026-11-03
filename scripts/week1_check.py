"""Week 1 capstone check: run every helper on N real records.

    uv run python scripts/week1_check.py                    # data/sample_20.jsonl, expects 20
    uv run python scripts/week1_check.py path.jsonl -n 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from capstone_project.labels import count_labels, count_tags
from capstone_project.records import load_records
from capstone_project.text import ticket_text, tokenize


def print_counts(title: str, counts: dict[str, int], top: int | None = None) -> None:
    print(f"\n{title}")
    for label, count in list(counts.items())[:top]:
        print(f"  {count:>3}  {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", type=Path, default=Path("data/sample_20.jsonl"))
    parser.add_argument("-n", type=int, default=20, help="expected number of records")
    args = parser.parse_args()

    # 1. load a record
    tickets = load_records(args.path, language="en")
    print(f"Loaded {len(tickets)} English tickets from {args.path}")

    # 2. count labels
    for field in ("queue", "priority", "type"):
        print_counts(f"{field} counts", count_labels(tickets, field))
    print_counts("top 10 tags", count_tags(tickets), top=10)

    # 3. clean text
    texts = [ticket_text(t) for t in tickets]
    raw_chars = sum(len(f"{t.subject}\n\n{t.body}") for t in tickets)
    clean_chars = sum(len(text) for text in texts)
    print(f"\ncleaned text: {raw_chars} -> {clean_chars} chars, "
          f"avg {sum(len(tokenize(text)) for text in texts) / max(len(texts), 1):.0f} words/ticket")
    for ticket, text in list(zip(tickets, texts))[:3]:
        preview = text.replace("\n\n", " | ")[:100]
        print(f"  [{ticket.queue} / {ticket.priority}] {preview}...")

    ok = len(tickets) == args.n and all(texts)
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(tickets)}/{args.n} records loaded, "
          f"{sum(bool(text) for text in texts)}/{len(texts)} with non-empty cleaned text")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
