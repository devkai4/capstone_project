from pathlib import Path

import pytest

from capstone_project.labels import count_labels, count_tags
from capstone_project.records import load_records

FIXTURE = Path(__file__).parent / "fixtures" / "tickets.jsonl"


@pytest.fixture
def tickets():
    return load_records(FIXTURE)


def test_count_queue(tickets):
    assert count_labels(tickets, "queue") == {"Billing and Payments": 2, "Technical Support": 2}


def test_count_priority_most_common_first(tickets):
    counts = count_labels(tickets, "priority")
    assert counts == {"high": 2, "low": 1, "medium": 1}
    assert list(counts) == ["high", "low", "medium"]  # ties are alphabetical


def test_counts_add_up_to_number_of_tickets(tickets):
    for field in ("queue", "priority", "type", "language"):
        assert sum(count_labels(tickets, field).values()) == len(tickets)


def test_count_tags(tickets):
    assert count_tags(tickets) == {"Billing": 2, "Network": 2, "Outage": 1, "VPN": 1}


def test_unknown_field_raises(tickets):
    with pytest.raises(ValueError, match="unknown label field"):
        count_labels(tickets, "subject")


def test_empty_input():
    assert count_labels([], "queue") == {}
    assert count_tags([]) == {}
