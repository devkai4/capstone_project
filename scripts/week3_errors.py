"""Week 3: why are queue predictions wrong? (reads a classify log, no LLM calls, no cost)

    uv run python scripts/week3_errors.py --log logs/classify_XXXX.jsonl \
        --pool data/pool_10100.json --k 8

For every ticket whose predicted queue is wrong, it re-runs the same BM25 retrieval and
sorts the error into one of three cases:

    retrieval  the true queue is not among the k retrieved examples
               -> better retrieval could fix it
    choice     the true queue IS among the examples, the model picked another one
               -> prompt / model could fix it
    (both columns show the rank-1 example so label-noise cases are visible too)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from capstone_project.records import load_json
from capstone_project.retrieval import BM25Index, ranked_examples, remove_overlap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True, help="logs/classify_*.jsonl")
    parser.add_argument("--data", type=Path, default=Path("data/tickets_100.json"))
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    all_100 = load_json(args.data)
    pool, _ = remove_overlap(load_json(args.pool), all_100)
    index = BM25Index(pool)
    rows = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line]

    cases = Counter()
    rank1_right = 0
    sweep_input = []  # (rank-1 overlap, rank-1 queue, model queue, true queue)
    print(f"{'#':>3}  {'case':<9} {'true queue':<24} {'predicted':<24} "
          f"{'rank-1 example (overlap)':<34} true queue at ranks")
    for row in rows:
        ticket = all_100[row["index"]]
        examples = ranked_examples(index, ticket, args.k)
        rank1 = examples[0]
        rank1_right += rank1.ticket.queue == ticket.queue
        pred = (row["pred"] or {}).get("queue", "(invalid)")
        sweep_input.append((rank1.overlap, rank1.ticket.queue, pred, ticket.queue))
        if pred == ticket.queue:
            continue
        hit_ranks = [e.rank for e in examples if e.ticket.queue == ticket.queue]
        case = "choice" if hit_ranks else "retrieval"
        cases[case] += 1
        print(f"{row['index']:>3}  {case:<9} {ticket.queue[:24]:<24} {pred[:24]:<24} "
              f"{rank1.ticket.queue[:26]:<26} ({rank1.overlap:.2f})  {hit_ranks or '-'}")
        print(f"     subject: {ticket.subject[:90]}")

    wrong = sum(cases.values())
    print(f"\nwrong queue: {wrong}/{len(rows)}  "
          f"retrieval (true queue not in top-{args.k}): {cases['retrieval']}  "
          f"choice (in top-{args.k}, model chose another): {cases['choice']}")
    print(f"rank-1 example has the true queue: {rank1_right}/{len(rows)}")

    print("\nnear-duplicate rule: use the rank-1 queue when its overlap >= threshold, else the model's")
    print("threshold  copied  fixed  broken  queue accuracy")
    for threshold in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        copied = fixed = broken = hits = 0
        for overlap, rank1_queue, pred, true in sweep_input:
            final = rank1_queue if overlap >= threshold else pred
            if final != pred:
                copied += 1
                fixed += final == true
                broken += pred == true
            hits += final == true
        print(f"  {threshold:<8} {copied:>6} {fixed:>6} {broken:>7}  {hits}/{len(sweep_input)} "
              f"= {hits / len(sweep_input):.0%}")
    print("\nrank-1 overlap of every ticket (correct = rank-1 queue is the true queue):")
    for overlap, rank1_queue, _, true in sorted(sweep_input, reverse=True):
        if overlap >= 0.3:
            print(f"  {overlap:.2f}  {'correct' if rank1_queue == true else 'WRONG  '}  {rank1_queue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
