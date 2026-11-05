"""Day 4: parametrized tests for the ingest code (CSV text -> dict -> Ticket).

Exactly 10 test cases come from 3 test functions:
    test_csv_quoting                     2 cases  (one parametrize, pytest.param ids)
    test_tags                            2 cases  (one parametrize, pytest.param ids)
    test_blank_required_value_is_rejected  2 x 3 = 6 cases  (two stacked parametrize decorators)

    uv run pytest -q tests/test_ingest.py      -> 10 passed
    uv run pytest tests/test_ingest.py -v      -> shows every case id
"""

import io

import pytest

from capstone_project.records import iter_csv_records, parse_record

HEADER = "subject,body,queue,priority,type,language\n"


def valid_row(**overrides) -> dict:
    """A raw record that parses fine; each test changes only what it is about."""
    row = {
        "subject": "VPN down",
        "body": "We cannot connect.",
        "queue": "Technical Support",
        "priority": "high",
        "type": "Incident",
        "language": "en",
    }
    row.update(overrides)
    return row


# 1. CSV quoting: the value must arrive in one piece and the columns must not shift.
@pytest.mark.parametrize(
    ("csv_body", "expected_body"),
    [
        pytest.param('"Hi, the VPN is down"', "Hi, the VPN is down", id="comma-inside-quotes"),
        pytest.param('"Send the ""March"" invoice"', 'Send the "March" invoice', id="doubled-quotes"),
    ],
)
def test_csv_quoting(csv_body, expected_body):
    text = io.StringIO(HEADER + f"VPN down,{csv_body},Technical Support,high,Incident,en\n")
    (ticket,) = list(iter_csv_records(text))
    assert ticket.body == expected_body
    assert ticket.queue == "Technical Support"  # the column after body did not shift


# 2. Tags: tag_1..tag_N collapse into one tuple, empty ones dropped, numeric order kept.
@pytest.mark.parametrize(
    ("tag_columns", "expected_tags"),
    [
        pytest.param(
            {"tag_1": "Network", "tag_2": None, "tag_3": "", "tag_4": "VPN"},
            ("Network", "VPN"),
            id="null-and-empty-skipped",
        ),
        pytest.param(
            {"tag_10": "Urgent", "tag_2": "VPN", "tag_1": "Network"},
            ("Network", "VPN", "Urgent"),
            id="tag_10-sorts-after-tag_2",  # string sort would give tag_1, tag_10, tag_2
        ),
    ],
)
def test_tags(tag_columns, expected_tags):
    assert parse_record(valid_row(**tag_columns)).tags == expected_tags


# 3. Required fields: stacked decorators run every combination (2 fields x 3 values = 6).
@pytest.mark.parametrize("bad_value", ["", None, "   "], ids=["empty", "null", "whitespace"])
@pytest.mark.parametrize("field", ["queue", "language"])
def test_blank_required_value_is_rejected(field, bad_value):
    with pytest.raises(ValueError, match=field):
        parse_record(valid_row(**{field: bad_value}))
