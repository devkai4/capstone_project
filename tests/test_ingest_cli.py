"""The ingest CLI end to end, on the local CSV fixture (no network)."""

import json
from pathlib import Path

from capstone_project.ingest import main

CSV_FIXTURE = Path(__file__).parent / "fixtures" / "tickets.csv"


def test_cli_pass(tmp_path, capsys):
    out = tmp_path / "out" / "tickets.json"
    assert main(["--csv", str(CSV_FIXTURE), "-n", "3", "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "PASS: queue counts sum to 3" in printed
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 3


def test_cli_fails_when_fewer_tickets_than_requested(tmp_path, capsys):
    out = tmp_path / "tickets.json"
    assert main(["--csv", str(CSV_FIXTURE), "-n", "4", "--out", str(out)]) == 1  # only 3 are English
    assert "FAIL" in capsys.readouterr().out
