import json
from pathlib import Path

import pytest

from capstone_project.records import Ticket, load_records, parse_record

FIXTURE = Path(__file__).parent / "fixtures" / "tickets.jsonl"


def test_load_all_records_skips_blank_lines():
    tickets = load_records(FIXTURE)
    assert len(tickets) == 4
    assert all(isinstance(t, Ticket) for t in tickets)


def test_language_filter():
    tickets = load_records(FIXTURE, language="en")
    assert [t.language for t in tickets] == ["en", "en", "en"]


def test_limit():
    assert len(load_records(FIXTURE, language="en", limit=2)) == 2


def test_first_record_fields():
    first = load_records(FIXTURE)[0]
    assert first.subject == "VPN Access Issue"
    assert first.queue == "Technical Support"
    assert first.priority == "high"
    assert first.type == "Incident"
    assert first.answer == "Please restart the router."


def test_tags_skip_null_and_empty():
    tickets = load_records(FIXTURE)
    assert tickets[0].tags == ("Network", "VPN")
    assert tickets[3].tags == ("Outage",)  # tag_2 is "" in the fixture


def test_null_subject_and_answer_become_empty_string():
    second = load_records(FIXTURE)[1]
    assert second.subject == ""
    assert second.answer == ""


def test_missing_required_field_raises():
    with pytest.raises(ValueError, match="queue"):
        parse_record({"body": "x", "priority": "low", "type": "Request", "language": "en"})


def test_bad_json_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    good = json.dumps({"body": "x", "queue": "q", "priority": "low", "type": "Request", "language": "en"})
    path.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_records(path)
