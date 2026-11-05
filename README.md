# Capstone: Ticket Triage Agent (Option A)

Reads a support ticket, classifies queue/priority/type, retrieves similar
resolved tickets, uses MCP tools, and answers or escalates.

## Dataset
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
(filter: language = en, license: CC BY-NC 4.0)
## Helpers (Week 1)

Plain-Python helpers in `src/capstone_project/` (standard library only):

| Module | Function | What it does |
|---|---|---|
| `records.py` | `load_records(path, ...)`, `load_csv_records(path, ...)` | Load JSONL / CSV rows into `Ticket` objects (tags collected, nulls → `""`) |
| `records.py` | `save_json(tickets, path)`, `load_json(path)` | Save / reload Tickets as one JSON array |
| `labels.py` | `count_labels(tickets, field)`, `count_tags(tickets)` | `{label: count}` for queue / priority / type / language / tags, most common first |
| `text.py` | `clean_text(text)`, `ticket_text(ticket)`, `tokenize(text)` | NFKC, drop invisible chars, collapse whitespace; subject + body; lowercase words |

```bash
uv run pytest                                # unit tests
uv run python scripts/fetch_sample.py        # 20 real English tickets -> data/sample_20.jsonl
uv run python scripts/week1_check.py         # run every helper on them (prints PASS/FAIL)
uv run python scripts/day2_load_100.py      # Day 2: 100 English tickets from the CSV -> data/tickets_100.json
```

## Docker (Week 2 Day 5)

```bash
docker build -t capstone-ingest .                                  # works from a clean clone
docker run --rm -v "$PWD/out:/out" capstone-ingest                 # ingest 100 tickets -> out/tickets_100.json
docker run --rm capstone-ingest pytest -q tests/test_ingest.py     # 10 tests inside the container
```

Without Docker the same command is `uv run capstone-ingest`. The `docker` GitHub Actions
workflow repeats all three steps on every push to main and every pull request.
