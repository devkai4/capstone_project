"""Count label values (queue, priority, type, tags) across tickets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from capstone_project.records import Ticket

LABEL_FIELDS = ("queue", "priority", "type", "language")


def _sorted_counts(counter: Counter) -> dict[str, int]:
    """Most common first; ties broken alphabetically so output is stable."""
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def count_labels(tickets: Iterable[Ticket], field: str) -> dict[str, int]:
    """Return {label: count} for one label field, most common first.

    >>> count_labels(tickets, "queue")
    {'Technical Support': 9, 'Billing and Payments': 4, ...}
    """
    if field not in LABEL_FIELDS:
        raise ValueError(f"unknown label field {field!r}; expected one of {LABEL_FIELDS}")
    return _sorted_counts(Counter(getattr(t, field) for t in tickets))


def count_tags(tickets: Iterable[Ticket]) -> dict[str, int]:
    """Return {tag: count} over all tags of all tickets, most common first."""
    return _sorted_counts(Counter(tag for t in tickets for tag in t.tags))
