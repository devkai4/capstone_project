"""Week 3 Day 1: send 1 ticket to Claude and log input / output token counts.

    export DATAROBOT_API_TOKEN=...            # read -s DATAROBOT_API_TOKEN && export DATAROBOT_API_TOKEN
    uv run python scripts/week3_send_one.py   # ticket #0 of data/tickets_100.json
    uv run python scripts/week3_send_one.py --index 5

Every call is also appended to logs/llm_calls.jsonl (git-ignored) for later cost tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from capstone_project.labels import count_labels
from capstone_project.llm import DEFAULT_MODEL, complete, make_client
from capstone_project.records import load_json
from capstone_project.text import ticket_text

LOG_FILE = Path("logs/llm_calls.jsonl")

logger = logging.getLogger("week3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/tickets_100.json"))
    parser.add_argument("--index", type=int, default=0, help="which ticket to send (default 0)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickets = load_json(args.data)
    ticket = tickets[args.index]
    queues = list(count_labels(tickets, "queue"))

    # Day 1 is only about the request itself; the real classifier prompt comes on Day 2.
    system = (
        "You route customer support tickets. Reply with exactly one queue name "
        "from this list and nothing else:\n" + "\n".join(f"- {q}" for q in queues)
    )
    user = f"<ticket>\n{ticket_text(ticket)}\n</ticket>"

    result = complete(make_client(), system, user, model=args.model, max_tokens=20)
    cost = result.estimated_cost_usd(args.model)

    logger.info("ticket #%d  true queue: %s", args.index, ticket.queue)
    logger.info("Claude says: %s  (%s)", result.text.strip(),
                "match" if result.text.strip() == ticket.queue else "no match")
    logger.info("model=%s input_tokens=%d output_tokens=%d latency=%.2fs est_cost=%s",
                result.model, result.input_tokens, result.output_tokens, result.seconds,
                f"${cost:.6f}" if cost is not None else "n/a")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "week3_send_one",
            "ticket_index": args.index,
            "requested_model": args.model,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "seconds": round(result.seconds, 3),
            "est_cost_usd": cost,
            "answer": result.text.strip(),
            "true_queue": ticket.queue,
        }) + "\n")
    logger.info("appended to %s", LOG_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
