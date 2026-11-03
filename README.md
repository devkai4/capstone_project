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
| `records.py` | `load_records(path, language=None, limit=None)` | Load JSONL rows into `Ticket` objects (tags collected, nulls → `""`) |
| `labels.py` | `count_labels(tickets, field)`, `count_tags(tickets)` | `{label: count}` for queue / priority / type / language / tags, most common first |
| `text.py` | `clean_text(text)`, `ticket_text(ticket)`, `tokenize(text)` | NFKC, drop invisible chars, collapse whitespace; subject + body; lowercase words |

```bash
uv run pytest                                # unit tests
uv run python scripts/fetch_sample.py        # 20 real English tickets -> data/sample_20.jsonl
uv run python scripts/week1_check.py         # run every helper on them (prints PASS/FAIL)
```
