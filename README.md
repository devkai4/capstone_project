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

## Metrics (Week 3: classifier on 50 labeled tickets)

Eval set: tickets 0-49 of `data/tickets_100.json`, exact match against the dataset labels.
Gate: 50/50 valid JSON, label accuracy >= 70%, cost per ticket and p50/p95 latency recorded.
Cost is estimated from Anthropic list prices; latency is per request through the staging LLM gateway.

| date | model | prompt | examples | valid JSON | queue | priority | type | mean | cost / ticket | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-05 | claude-haiku-4-5-20251001 | classifier_v1 | 5 static | 50/50 | 32% | 54% | 76% | 54% | $0.00173 | 1.20s | 3.17s |

`scripts/week3_classify.py` prints the next row ready to paste.

### Label-noise check (non-LLM baselines, `scripts/week3_baseline.py`)

How predictable the labels are from the text alone, on the same 50 tickets:

| method | queue | priority | type | mean |
|---|---|---|---|---|
| majority class | | | | |
| BM25 k-NN (k=5) | | | | |
| TF-IDF + logistic regression | | | | |
| TF-IDF + logistic regression, 5-fold CV on the pool | | | | |

Observed in the eval set: near-identical connectivity tickets are labeled Technical Support (#20),
IT Support (#19, #45), Product Support (#7, #49) and Customer Service (#26); some labels do not
match the text (#1 smart-home question -> Returns and Exchanges, #17 Office apps -> Human Resources).
