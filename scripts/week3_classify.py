"""Week 3 Day 3: classify 50 tickets into JSON {queue, priority, type} validated with Pydantic.

    uv run python scripts/week3_classify.py                 # tickets 0-49, structured outputs on
    uv run python scripts/week3_classify.py --no-schema     # same prompt, no response_format (comparison)

Tickets 50-99 are kept out of the evaluation because the few-shot examples in
prompts/classifier_v1.txt come from there. Per-ticket results go to logs/ (git-ignored).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from capstone_project.classifier import classify_many, load_prompt
from capstone_project.llm import DEFAULT_MODEL, PRICE_PER_MTOK, make_async_client
from capstone_project.records import load_json
from capstone_project.text import clean_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/tickets_100.json"))
    parser.add_argument("-n", type=int, default=50, help="classify tickets 0..n-1 (default 50)")
    parser.add_argument("--prompt", type=Path, default=Path("prompts/classifier_v1.txt"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-schema", action="store_true", help="do not send response_format")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  log %(levelname)s %(message)s")

    tickets = load_json(args.data)[: args.n]
    system = load_prompt(args.prompt)

    leaked = [i for i, t in enumerate(tickets) if clean_text(t.body) in system]
    if leaked:
        print(f"ABORT: tickets {leaked} are also few-shot examples in {args.prompt}")
        return 1

    use_schema = not args.no_schema
    print(f"Classifying {len(tickets)} tickets with {args.model} "
          f"(prompt {args.prompt.name}, structured outputs {'on' if use_schema else 'off'})")
    client = make_async_client()
    results = asyncio.run(classify_many(client, tickets, system, args.model, use_schema,
                                        args.concurrency))

    parsed = [r for r in results if r.labels is not None]
    print(f"\nparsed into TicketLabels: {len(parsed)}/{len(results)}")
    for i, r in enumerate(results):
        if r.labels is None:
            print(f"  #{i}: {r.error}  raw={r.raw[:60]!r}")

    pairs = [(t, r.labels) for t, r in zip(tickets, results) if r.labels is not None]
    if pairs:
        print("\naccuracy on parsed outputs (labels in this dataset are noisy):")
        for field in ("queue", "priority", "type"):
            hits = sum(getattr(t, field) == getattr(lab, field) for t, lab in pairs)
            print(f"  {field:<8} {hits}/{len(pairs)} = {hits / len(pairs):.0%}")

    tokens_in = sum(r.input_tokens for r in results)
    tokens_out = sum(r.output_tokens for r in results)
    latencies = sorted(r.seconds for r in results)
    p95 = latencies[max(0, round(0.95 * len(latencies)) - 1)]
    price = PRICE_PER_MTOK.get(args.model)
    cost = (tokens_in * price[0] + tokens_out * price[1]) / 1e6 if price else None
    print(f"\ntokens: {tokens_in} in / {tokens_out} out "
          f"(avg {tokens_in / len(results):.0f} / {tokens_out / len(results):.0f} per ticket)")
    print(f"latency: median {statistics.median(latencies):.2f}s, p95 {p95:.2f}s")
    if cost is not None:
        print(f"est. cost: ${cost:.4f} total, ${cost / len(results):.5f} per ticket (list price)")

    log = Path("logs") / f"classify_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl"
    log.parent.mkdir(exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        for i, (t, r) in enumerate(zip(tickets, results)):
            f.write(json.dumps({
                "index": i, "true": {"queue": t.queue, "priority": t.priority, "type": t.type},
                "pred": r.labels.model_dump() if r.labels else None, "raw": r.raw,
                "error": r.error, "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens, "seconds": round(r.seconds, 3),
                "finish_reason": r.finish_reason,
            }) + "\n")
    print(f"per-ticket results: {log}")

    ok = len(parsed) == len(results)
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(parsed)}/{len(results)} outputs parse into the Pydantic model")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
