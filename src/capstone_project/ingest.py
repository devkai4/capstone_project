"""Ingest CLI: load N English tickets from the public dataset's CSV into Ticket dataclasses,
print a count per queue, save them as JSON, and check the result.

    capstone-ingest                                   # download 100 English tickets
    capstone-ingest --csv tickets.csv -n 50           # read a local CSV instead
    capstone-ingest --out /out/tickets_100.json       # choose the output file

Exit code 0 = PASS (exactly N tickets, counts sum to N, JSON reloads identically), 1 = FAIL.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

from capstone_project.labels import count_labels
from capstone_project.records import (
    Ticket,
    iter_csv_records,
    load_csv_records,
    load_json,
    save_json,
    select_tickets,
)

CSV_URL = (
    "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/"
    "resolve/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)


def load_from_url(url: str, language: str, limit: int) -> list[Ticket]:
    """Stream the CSV over HTTP and stop reading as soon as `limit` tickets are found."""
    request = urllib.request.Request(url, headers={"User-Agent": "capstone-project/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = io.TextIOWrapper(response, encoding="utf-8-sig", newline="")
        return select_tickets(iter_csv_records(text, url), language, limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="capstone-ingest", description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, help="read a local CSV instead of downloading")
    parser.add_argument("-n", type=int, default=100, help="number of tickets (default 100)")
    parser.add_argument("--language", default="en")
    parser.add_argument("-o", "--out", type=Path, default=Path("data/tickets_100.json"))
    args = parser.parse_args(argv)

    source = str(args.csv) if args.csv else CSV_URL
    if args.csv:
        tickets = load_csv_records(args.csv, language=args.language, limit=args.n)
    else:
        tickets = load_from_url(CSV_URL, args.language, args.n)
    print(f"Loaded {len(tickets)} {args.language} tickets from {source}")

    counts = count_labels(tickets, "queue")
    print("\nqueue counts")
    for queue, count in counts.items():
        print(f"  {count:>3}  {queue}")
    total = sum(counts.values())
    print(f"  ---\n  {total:>3}  total")

    save_json(tickets, args.out)
    reloaded = load_json(args.out)
    print(f"\nSaved {len(tickets)} tickets to {args.out} "
          f"(reloaded {len(reloaded)}, identical: {reloaded == tickets})")

    ok = total == args.n == len(tickets) and reloaded == tickets
    print(f"\n{'PASS' if ok else 'FAIL'}: queue counts sum to {total} (expected exactly {args.n})")
    return 0 if ok else 1
