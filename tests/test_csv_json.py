import io
from pathlib import Path

import pytest

from capstone_project.labels import count_labels
from capstone_project.records import (
    iter_csv_records,
    load_csv_records,
    load_json,
    save_json,
    select_tickets,
)

CSV_FIXTURE = Path(__file__).parent / "fixtures" / "tickets.csv"


def test_load_csv_all_rows():
    assert len(load_csv_records(CSV_FIXTURE)) == 4


def test_load_csv_language_and_limit():
    english = load_csv_records(CSV_FIXTURE, language="en")
    assert [t.language for t in english] == ["en", "en", "en"]
    assert len(load_csv_records(CSV_FIXTURE, language="en", limit=2)) == 2


def test_quoted_comma_and_line_break_stay_in_one_field():
    first = load_csv_records(CSV_FIXTURE)[0]
    assert first.body == "We cannot connect, the VPN\nkeeps dropping."
    assert first.queue == "Technical Support"  # columns did not shift


def test_escaped_quotes_and_empty_cells():
    second = load_csv_records(CSV_FIXTURE)[1]
    assert second.body == 'Please send the "March" invoice.'
    assert second.subject == ""
    assert second.answer == ""


def test_tags_from_any_number_of_tag_columns():
    tickets = load_csv_records(CSV_FIXTURE)
    assert tickets[0].tags == ("Network", "VPN")
    assert tickets[3].tags == ("Outage", "Urgent")  # tag_10 is kept, empty cells skipped


def test_queue_counts_sum_to_number_of_tickets():
    tickets = load_csv_records(CSV_FIXTURE, language="en")
    counts = count_labels(tickets, "queue")
    assert counts == {"Technical Support": 2, "Billing and Payments": 1}
    assert sum(counts.values()) == len(tickets)


def test_missing_header_column_lists_found_columns():
    f = io.StringIO("subject,body,queue\nHi,Hello,Technical Support\n")
    with pytest.raises(ValueError, match=r"no column\(s\) \['priority', 'type', 'language'\]"):
        list(iter_csv_records(f))


def test_empty_required_cell_reports_line():
    f = io.StringIO(
        "body,queue,priority,type,language\n"
        "Hello,Technical Support,high,Incident,en\n"
        "Hello,,high,Incident,en\n"
    )
    with pytest.raises(ValueError, match=r"<csv>:3: .*queue"):
        list(iter_csv_records(f))


def test_select_stops_reading_after_limit():
    seen = []

    def stream():
        for ticket in load_csv_records(CSV_FIXTURE):
            seen.append(ticket)
            yield ticket

    assert len(select_tickets(stream(), "en", 1)) == 1
    assert len(seen) == 1  # the rest of the (possibly huge) file is never read


def test_json_round_trip(tmp_path):
    tickets = load_csv_records(CSV_FIXTURE)
    path = tmp_path / "out" / "tickets.json"
    save_json(tickets, path)
    assert load_json(path) == tickets
    assert path.read_text(encoding="utf-8").startswith("[\n  {\n")
