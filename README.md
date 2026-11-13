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

Capstone Option A: **accuracy = the queue label matches the dataset label.** priority and type
are recorded for reference only.
Week 3 gate: 50/50 valid JSON, queue accuracy >= 70%, cost per ticket and p50/p95 latency in this README.

Tuning set: tickets 0-49 of `data/tickets_100.json` (all prompt / k / pool choices were made here).
Held-out check: tickets 50-99, used once for the final configuration (v1 took its 5 static
examples from there, so v1 is not run on it). Cost is estimated from Anthropic list prices;
latency is per request through the staging LLM gateway.

| date | eval | model | prompt | examples | valid JSON | **queue** | priority | type | cost / ticket | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-05 | tickets 0-49 | claude-haiku-4-5-20251001 | classifier_v1 | 5 static | 50/50 | **32%** | 54% | 76% | $0.00173 | 1.20s | 3.17s |
| 2026-11-13 | tickets 0-49 | claude-haiku-4-5-20251001 | classifier_v2 | BM25 top-8 of 2000 | 50/50 | **52%** | 64% | 78% | $0.00225 | 0.83s | 1.92s |
| 2026-11-13 | tickets 0-49 | claude-haiku-4-5-20251001 | classifier_v3 | BM25 top-3 of 2000, ranked+overlap | 50/50 | **62%** | 64% | 82% | $0.00158 | 0.80s | 1.67s |
| 2026-11-13 | tickets 0-49 | claude-haiku-4-5-20251001 | classifier_v3 | BM25 top-8 of 2000, ranked+overlap | 50/50 | **66%** | 70% | 82% | $0.00244 | 0.83s | 1.52s |
| 2026-11-13 | tickets 0-49 | claude-haiku-4-5-20251001 | classifier_v3 | BM25 top-8 of 10000, ranked+overlap | 50/50 | **66%** | 74% | 84% | $0.00242 | 0.77s | 1.66s |
| 2026-11-13 | tickets 0-49 | claude-haiku-4-5-20251001 | classifier_v3 + near-duplicate rule | BM25 top-8 of 10000, ranked+overlap, copy rank-1 queue if overlap >= 0.5 | 50/50 | **72%** | 74% | 84% | $0.00242 | 0.74s | 1.71s |
| 2026-11-13 | **tickets 50-99 (held-out)** | claude-haiku-4-5-20251001 | classifier_v3 + near-duplicate rule | same | 50/50 | **58%** | 42% | 86% | $0.00245 | 0.76s | 1.19s |

`scripts/week3_classify.py` prints the next row ready to paste.

**Week 3 gate: passed on the eval set (tickets 0-49): queue accuracy 72% (36/50), 50/50 valid
JSON, $0.00242 per ticket, p50 0.74s, p95 1.71s** (SPEC limits: <= $0.01, p95 <= 8s).

**Held-out check (tickets 50-99, never used for tuning): queue accuracy 58% (29/50).**
The 72% is optimistic: prompt, k, pool size and threshold were all chosen on tickets 0-49.
Realistic accuracy on new tickets is closer to 58-65% (65/100 over both sets), and 50 tickets
is a small sample (one ticket = 2 points).

### What decides queue accuracy: is there a near-duplicate in the history?

The dataset holds several versions of the same ticket. When the closest past ticket shares
at least half of its words with the new one (rank-1 word overlap >= 0.5), its queue label was
right **every time (36/36 across both sets; 40/40 for overlap >= 0.3)**. Without such a match
the model is right only about half the time (labels are inconsistent, see below).

| | tickets 0-49 | tickets 50-99 |
|---|---|---|
| near-duplicate in the pool (overlap >= 0.5) | 22 | 14 |
| queue accuracy on those | 22/22 | 14/14 |
| queue accuracy on the rest | 14/28 (50%) | 15/36 (42%) |
| overall | 72% | 58% |

So overall accuracy mostly measures how many tickets have a near-duplicate in the pool.
Using every English ticket in the CSV (16,338) as the pool found no additional near-duplicates.

Steps that moved queue accuracy on tickets 0-49:
- v1 -> v2 (32% -> 52%): retrieve similar labeled tickets instead of 5 fixed examples.
- v2 -> v3 (52% -> 66%): tell the model each example's rank and word overlap, and to copy a
  clear rank-1 match.
- near-duplicate rule (66% -> 72%): the model still ignored two near-duplicates (#9 overlap 0.65,
  #45 overlap 0.54), so code now copies the rank-1 queue when overlap >= 0.5. The threshold sits
  in the gap between near-duplicates (>= 0.53) and the rest; 0.3 would score 72%+ on 0-49 too but
  only because of one ticket at 0.31, so it was not chosen. The rule never broke a correct answer.

Next (Week 4+): treat "no near-duplicate" as low confidence. That is where the agent should
escalate to a human (capstone Option A's escalation) instead of guessing a queue.

Reproduce:

```bash
uv run capstone-ingest -n 10100 --out data/pool_10100.json
uv run python scripts/week3_classify.py --prompt prompts/classifier_v3.txt \
    --pool data/pool_10100.json --k 8 --show-similarity --copy-threshold 0.5           # 0-49
uv run python scripts/week3_classify.py --start 50 --prompt prompts/classifier_v3.txt \
    --pool data/pool_10100.json --k 8 --show-similarity --copy-threshold 0.5           # 50-99
uv run python scripts/week3_errors.py --log logs/classify_<run>.jsonl --pool data/pool_10100.json
```

### Label-noise check (non-LLM baselines, `scripts/week3_baseline.py`)

How predictable the labels are from the text alone, on the same 50 tickets:

| method | queue | priority | type | mean |
|---|---|---|---|---|
| majority class | 34% | 50% | 38% | 41% |
| BM25 k-NN, k=1 (copy the closest ticket) | 62% | 70% | 82% | 71% |
| BM25 k-NN, k=5 (vote) | 44% | 50% | 78% | 57% |
| BM25 k-NN, k=10 (vote) | 46% | 64% | 74% | 61% |
| TF-IDF + logistic regression | 58% | 72% | 84% | 71% |
| TF-IDF + logistic regression, 5-fold CV on the 2000-ticket pool | 50% | 54% | 80% | 61% |

Pool: first 2100 English tickets of the CSV minus the 100 in `tickets_100.json` (2000 left).
k=1 beating k=5/k=10 suggests the pool holds near-duplicate versions of the eval tickets
(the CSV is a "version" dataset): the single closest ticket carries the label, extra
neighbours mostly add noise. This makes retrieval-based scores optimistic for truly new tickets.

Observed in the eval set: near-identical connectivity tickets are labeled Technical Support (#20),
IT Support (#19, #45), Product Support (#7, #49) and Customer Service (#26); some labels do not
match the text (#1 smart-home question -> Returns and Exchanges, #17 Office apps -> Human Resources).
