"""Load support-ticket records from a JSON Lines file.

Each line is one flat JSON object with the dataset's columns, e.g.
{"subject": ..., "body": ..., "queue": ..., "priority": ..., "type": ...,
 "language": "en", "tag_1": "Network", ..., "tag_8": null}
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("body", "queue", "priority", "type", "language")
TAG_FIELDS = tuple(f"tag_{i}" for i in range(1, 9))


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

    - subject / answer may be null in the dataset -> stored as "".
    - tag_1..tag_8 are collapsed into one tuple, skipping null/empty tags.
    """
    missing = [name for name in REQUIRED_FIELDS if raw.get(name) is None]
    if missing:
        raise ValueError(f"record is missing required field(s): {', '.join(missing)}")

    tags = tuple(raw[name] for name in TAG_FIELDS if raw.get(name))
    return Ticket(
        subject=raw.get("subject") or "",
        body=raw["body"],
        queue=raw["queue"],
        priority=raw["priority"],
        type=raw["type"],
        language=raw["language"],
        tags=tags,
        answer=raw.get("answer") or "",
    )


def iter_records(path: str | Path) -> Iterator[Ticket]:
    """Yield Tickets from a JSONL file one line at a time. Blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                yield parse_record(raw)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e


def load_records(
    path: str | Path,
    language: str | None = None,
    limit: int | None = None,
) -> list[Ticket]:
    """Load Tickets into a list, optionally keeping one language and at most `limit` rows."""
    tickets = []
    for ticket in iter_records(path):
        if language is not None and ticket.language != language:
            continue
        tickets.append(ticket)
        if limit is not None and len(tickets) >= limit:
            break
    return tickets
