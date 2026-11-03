"""Download the first N tickets in one language from the Hugging Face dataset.

Uses the public datasets-server API with the standard library only.

    uv run python scripts/fetch_sample.py              # 20 English tickets -> data/sample_20.jsonl
    uv run python scripts/fetch_sample.py -n 50 -o data/sample_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "Tobi-Bueck/customer-support-tickets"
PAGE_SIZE = 100  # API maximum per request
FIELDS = (
    "subject", "body", "answer", "type", "queue", "priority", "language",
    *(f"tag_{i}" for i in range(1, 9)),
)


def fetch_page(offset: int, length: int = PAGE_SIZE) -> list[dict]:
    params = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    })
    request = urllib.request.Request(
        f"{API_URL}?{params}", headers={"User-Agent": "capstone-project/0.1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["rows"]


def fetch_tickets(n: int, language: str) -> list[dict]:
    tickets: list[dict] = []
    offset = 0
    while len(tickets) < n:
        rows = fetch_page(offset)
        if not rows:
            break  # end of dataset
        for item in rows:
            row = item["row"]
            if row.get("language") == language:
                tickets.append({"row_idx": item["row_idx"], **{k: row.get(k) for k in FIELDS}})
                if len(tickets) == n:
                    break
        offset += len(rows)
    return tickets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", type=int, default=20, help="number of tickets (default 20)")
    parser.add_argument("--language", default="en", help="language code (default en)")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output JSONL (default data/sample_<n>.jsonl)")
    args = parser.parse_args()

    out = args.out or Path("data") / f"sample_{args.n}.jsonl"
    tickets = fetch_tickets(args.n, args.language)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ticket in tickets:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    print(f"Wrote {len(tickets)} {args.language} tickets to {out}")


if __name__ == "__main__":
    main()
