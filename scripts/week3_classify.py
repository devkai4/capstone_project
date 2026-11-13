"""Week 3: classify the 50 eval tickets into JSON {queue, priority, type} validated with Pydantic.

    # v1: static few-shot (5 examples inside the system prompt)
    uv run python scripts/week3_classify.py

    # v2: dynamic few-shot, the 8 most similar labeled tickets from a pool (BM25)
    uv run capstone-ingest -n 2100 --out data/pool_2100.json      # once
    uv run python scripts/week3_classify.py --prompt prompts/classifier_v2.txt \
        --pool data/pool_2100.json --k 8

    # v3: same, but each example shows its rank and word overlap with the ticket
    uv run python scripts/week3_classify.py --prompt prompts/classifier_v3.txt \
        --pool data/pool_2100.json --k 3 --show-similarity

Eval set = tickets 0-49 of data/tickets_100.json. Tickets 50-99 hold the v1 examples.
Every pool ticket whose body appears in tickets_100.json is removed (no leakage).
Prints a markdown row for the README metrics table; per-ticket results go to logs/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from capstone_project.classifier import classify_many, load_prompt
from capstone_project.evaluation import accuracy, percentile
from capstone_project.llm import DEFAULT_MODEL, PRICE_PER_MTOK, make_async_client
from capstone_project.records import load_json
from capstone_project.retrieval import (
    BM25Index,
    near_duplicate_queue,
    ranked_examples,
    remove_overlap,
)
from capstone_project.text import clean_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/tickets_100.json"))
    parser.add_argument("-n", type=int, default=50, help="number of tickets to classify (default 50)")
    parser.add_argument("--start", type=int, default=0,
                        help="first ticket index (default 0; --start 50 = held-out check on 50-99)")
    parser.add_argument("--prompt", type=Path, default=Path("prompts/classifier_v1.txt"))
    parser.add_argument("--pool", type=Path, help="labeled tickets to retrieve examples from")
    parser.add_argument("--k", type=int, default=8, help="examples per ticket with --pool")
    parser.add_argument("--copy-threshold", type=float,
                        help="if the rank-1 example's word overlap is >= this, use its queue "
                             "instead of the model's (needs --pool)")
    parser.add_argument("--show-similarity", action="store_true",
                        help="tag each example with its rank and word overlap (use with prompt v3)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-schema", action="store_true", help="do not send response_format")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  log %(levelname)s %(message)s")

    all_100 = load_json(args.data)
    tickets = all_100[args.start: args.start + args.n]
    eval_range = f"tickets {args.start}-{args.start + len(tickets) - 1}"
    system = load_prompt(args.prompt)

    leaked = [i for i, t in enumerate(tickets) if clean_text(t.body) in system]
    if leaked:
        print(f"ABORT: eval tickets {leaked} also appear in {args.prompt}")
        return 1

    examples_for = None
    retrieval = "none"
    if args.pool:
        pool, dropped = remove_overlap(load_json(args.pool), all_100)
        index = BM25Index(pool)
        if args.show_similarity:
            examples_for = lambda t: ranked_examples(index, t, args.k)  # noqa: E731
            retrieval = f"BM25 top-{args.k} of {len(pool)}, ranked+overlap"
        else:
            examples_for = lambda t: index.top_k(t, args.k)  # noqa: E731
            retrieval = f"BM25 top-{args.k} of {len(pool)}"
        print(f"pool: {len(pool)} tickets ({dropped} removed because they are in {args.data.name})")

    use_schema = not args.no_schema
    print(f"Classifying {eval_range} with {args.model} (prompt {args.prompt.name}, "
          f"examples: {retrieval}, structured outputs {'on' if use_schema else 'off'})")
    client = make_async_client()
    results = asyncio.run(classify_many(client, tickets, system, args.model, use_schema,
                                        args.concurrency, examples_for=examples_for))

    valid = sum(r.labels is not None for r in results)  # counted on the model's own output

    if args.copy_threshold is not None:
        if not args.pool:
            print("ABORT: --copy-threshold needs --pool")
            return 1
        copied = 0
        for i, (t, r) in enumerate(zip(tickets, results)):
            queue = near_duplicate_queue(ranked_examples(index, t, 1), args.copy_threshold)
            if queue and r.labels is not None and queue != r.labels.queue:
                results[i] = replace(r, labels=r.labels.model_copy(update={"queue": queue}))
                copied += 1
        retrieval += f", copy rank-1 queue if overlap >= {args.copy_threshold}"
        print(f"near-duplicate rule changed the queue of {copied} tickets")
    print(f"\nvalid JSON parsed into TicketLabels: {valid}/{len(results)}")
    for i, r in enumerate(results):
        if r.labels is None:
            print(f"  #{i}: {r.error}  raw={r.raw[:60]!r}")

    acc = accuracy(tickets, [r.labels for r in results])  # invalid output counts as wrong
    print("\naccuracy (exact match vs dataset labels, invalid = wrong):")
    for label in ("queue", "priority", "type", "mean"):
        print(f"  {label:<8} {acc[label]:.0%}")

    n = len(results)
    tokens_in = sum(r.input_tokens for r in results)
    tokens_out = sum(r.output_tokens for r in results)
    latencies = [r.seconds for r in results]
    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    price = PRICE_PER_MTOK.get(args.model)
    cost = (tokens_in * price[0] + tokens_out * price[1]) / 1e6 / n if price else None
    print(f"\ntokens per ticket: {tokens_in / n:.0f} in / {tokens_out / n:.0f} out")
    print(f"latency: p50 {p50:.2f}s, p95 {p95:.2f}s")
    if cost is not None:
        print(f"cost per ticket: ${cost:.5f} (list price)")

    log = Path("logs") / f"classify_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl"
    log.parent.mkdir(exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        for i, (t, r) in enumerate(zip(tickets, results)):
            f.write(json.dumps({
                "index": args.start + i, "true": {"queue": t.queue, "priority": t.priority, "type": t.type},
                "pred": r.labels.model_dump() if r.labels else None, "raw": r.raw,
                "error": r.error, "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens, "seconds": round(r.seconds, 3),
                "finish_reason": r.finish_reason,
            }) + "\n")
    print(f"per-ticket results: {log}")

    model_short = args.model.split("/")[-1]
    cost_cell = f"${cost:.5f}" if cost is not None else "n/a"
    print("\nREADME metrics row:")
    print(f"| {datetime.now():%Y-%m-%d} | {eval_range} | {model_short} | {args.prompt.stem} "
          f"| {retrieval} | {valid}/{n} | **{acc['queue']:.0%}** | {acc['priority']:.0%} "
          f"| {acc['type']:.0%} | {cost_cell} | {p50:.2f}s | {p95:.2f}s |")
    print(f"\nqueue accuracy (capstone metric): {acc['queue']:.0%} "
          f"({'meets' if acc['queue'] >= 0.7 else 'below'} the 70% gate)")
    return 0 if valid == n else 1


if __name__ == "__main__":
    sys.exit(main())
