"""Load support-ticket records (JSONL or CSV) into Ticket dataclasses, and save them as JSON.

A raw record is one flat dict with the dataset's columns, e.g.
{"subject": ..., "body": ..., "queue": ..., "priority": ..., "type": ...,
 "language": "en", "tag_1": "Network", ..., "tag_8": null}
JSONL files hold one such object per line; CSV files hold the same columns as a header row.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

REQUIRED_FIELDS = ("body", "queue", "priority", "type", "language")
_TAG_FIELD = re.compile(r"tag_(\d+)")


@dataclass(frozen=True)
class Ticket:
    subject: str
    body: str
    queue: str
    priority: str
    type: str
    language: str
    tags: tuple[str, ...] = ()
    answer: str = ""


def parse_record(raw: dict) -> Ticket:
    """Turn one raw dataset row (a dict) into a Ticket.

    - Required fields must be present and non-empty (CSV gives "" for empty cells, JSON gives null).
    - subject / answer may be null or "" -> stored as "".
    - tag_1, tag_2, ... (any number) are collapsed into one tuple in order, skipping empty tags.
    """
    missing = [name for name in REQUIRED_FIELDS if not raw.get(name)]
    if missing:
        raise ValueError(f"record is missing required field(s): {', '.join(missing)}")

    tag_names = sorted(
        (name for name in raw if _TAG_FIELD.fullmatch(name)),
        key=lambda name: int(name.split("_")[1]),
    )
    return Ticket(
        subject=raw.get("subject") or "",
        body=raw["body"],
        queue=raw["queue"],
        priority=raw["priority"],
        type=raw["type"],
        language=raw["language"],
        tags=tuple(raw[name] for name in tag_names if raw[name]),
        answer=raw.get("answer") or "",
    )


def select_tickets(
    tickets: Iterable[Ticket], language: str | None, limit: int | None
) -> list[Ticket]:
    """Keep one language (if given) and stop after `limit` tickets (if given)."""
    selected = []
    for ticket in tickets:
        if language is not None and ticket.language != language:
            continue
        selected.append(ticket)
        if limit is not None and len(selected) >= limit:
            break
    return selected


# ---------- JSON Lines ----------

def iter_records(path: str | Path) -> Iterator[Ticket]:
    """Yield Tickets from a JSONL file one line at a time. Blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield parse_record(json.loads(line))
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e


def load_records(
    path: str | Path, language: str | None = None, limit: int | None = None
) -> list[Ticket]:
    """Load Tickets from a JSONL file."""
    return select_tickets(iter_records(path), language, limit)


# ---------- CSV ----------

def iter_csv_records(f: TextIO, source: str = "<csv>") -> Iterator[Ticket]:
    """Yield Tickets from an open CSV text stream with a header row.

    Open files with newline="" so line breaks inside quoted fields are kept (csv module docs).
    """
    reader = csv.DictReader(f)
    header = reader.fieldnames or []
    missing = [name for name in REQUIRED_FIELDS if name not in header]
    if missing:
        raise ValueError(f"{source}: CSV header has no column(s) {missing}; found {header}")
    for row in reader:
        try:
            yield parse_record(row)
        except ValueError as e:
            raise ValueError(f"{source}:{reader.line_num}: {e}") from e


def load_csv_records(
    path: str | Path, language: str | None = None, limit: int | None = None
) -> list[Ticket]:
    """Load Tickets from a CSV file."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return select_tickets(iter_csv_records(f, str(path)), language, limit)


# ---------- JSON ----------

def save_json(tickets: Iterable[Ticket], path: str | Path) -> None:
    """Save Tickets as one JSON array (tags become a list)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(t) for t in tickets]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: str | Path) -> list[Ticket]:
    """Load Tickets saved by save_json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Ticket(**{**item, "tags": tuple(item["tags"])}) for item in data]
