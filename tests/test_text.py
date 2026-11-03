from pathlib import Path

import pytest

from capstone_project.records import load_records
from capstone_project.text import clean_text, ticket_text, tokenize

FIXTURE = Path(__file__).parent / "fixtures" / "tickets.jsonl"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        ("", ""),
        ("  hello   world \n", "hello world"),
        ("Dear Team,\n\nThe VPN\tis down.", "Dear Team, The VPN is down."),
        ("ＡＢＣ　１２３", "ABC 123"),                  # full-width -> half-width (NFKC)
        ("pass​word", "password"),                # zero-width space removed
        ("non breaking", "non breaking"),         # NBSP -> space
        ("bell\x07char", "bell char"),                 # control char -> space
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_lower():
    assert clean_text("VPN Is DOWN", lower=True) == "vpn is down"


def test_ticket_text_joins_subject_and_body():
    first = load_records(FIXTURE)[0]
    assert ticket_text(first) == "VPN Access Issue\n\nWe cannot connect to the VPN."


def test_ticket_text_without_subject():
    second = load_records(FIXTURE)[1]
    assert ticket_text(second) == "Please send the invoice for March."


def test_tokenize():
    assert tokenize("Can't log-in to the VPN (error 401)!") == [
        "can't", "log-in", "to", "the", "vpn", "error", "401",
    ]
    assert tokenize(None) == []
