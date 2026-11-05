"""Ticket classifier: Claude returns JSON {queue, priority, type}, validated with Pydantic.

The same Pydantic model is used twice:
1. its JSON Schema is sent as `response_format` (structured outputs: constrained decoding
   makes the model produce JSON that fits the schema), and
2. the reply is parsed with `TicketLabels.model_validate_json()`, so anything that does not
   fit (bad JSON, unknown label, extra or missing key) is caught here, not later.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from capstone_project.llm import DEFAULT_MODEL
from capstone_project.records import Ticket
from capstone_project.text import clean_text

logger = logging.getLogger(__name__)

QUEUES = (
    "Technical Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Product Support",
    "Service Outages and Maintenance",
    "Returns and Exchanges",
    "General Inquiry",
    "Human Resources",
    "Sales and Pre-Sales",
)
PRIORITIES = ("high", "medium", "low")
TYPES = ("Incident", "Request", "Problem", "Change")

_CANONICAL = {value.lower(): value for value in QUEUES + PRIORITIES + TYPES}


class TicketLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")  # -> "additionalProperties": false

    queue: Literal[QUEUES]
    priority: Literal[PRIORITIES]
    type: Literal[TYPES]

    @field_validator("queue", "priority", "type", mode="before")
    @classmethod
    def _canonical_case(cls, value: Any) -> Any:
        # Structured-outputs docs: enum casing can occasionally differ ("It support").
        # Accept a case-only difference, still reject unknown labels.
        if isinstance(value, str):
            return _CANONICAL.get(value.strip().lower(), value)
        return value


def _strip_titles(schema: Any) -> Any:
    """Pydantic adds "title" to every field; the model does not need them."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def response_format() -> dict:
    """OpenAI-style structured-output request built from the Pydantic model."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ticket_labels",
            "strict": True,
            "schema": _strip_titles(TicketLabels.model_json_schema()),
        },
    }


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def user_message(ticket: Ticket) -> str:
    return (
        "<ticket>\n"
        f"<subject>{clean_text(ticket.subject)}</subject>\n"
        f"<body>{clean_text(ticket.body)}</body>\n"
        "</ticket>"
    )


@dataclass(frozen=True)
class ClassifyResult:
    labels: TicketLabels | None   # None when the reply did not validate
    raw: str
    error: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    finish_reason: str = ""


async def classify(
    client: AsyncOpenAI,
    ticket: Ticket,
    system: str,
    model: str = DEFAULT_MODEL,
    use_schema: bool = True,
    max_tokens: int = 100,
) -> ClassifyResult:
    """One ticket -> one ClassifyResult. Never raises for a bad reply; API errors still raise."""
    extra = {"response_format": response_format()} if use_schema else {}
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message(ticket)},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        **extra,
    )
    seconds = time.perf_counter() - start
    choice = response.choices[0]
    raw = choice.message.content or ""
    usage = dict(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        seconds=seconds,
        finish_reason=choice.finish_reason,
    )
    try:
        labels = TicketLabels.model_validate_json(raw)
    except ValidationError as e:
        first = e.errors()[0]
        error = f"{first['type']}: {first['msg']}"
        logger.warning("parse failed (%s, finish=%s): %r", error, choice.finish_reason, raw[:80])
        return ClassifyResult(None, raw, error, **usage)
    return ClassifyResult(labels, raw, None, **usage)


async def classify_many(
    client: AsyncOpenAI,
    tickets: list[Ticket],
    system: str,
    model: str = DEFAULT_MODEL,
    use_schema: bool = True,
    concurrency: int = 5,
) -> list[ClassifyResult]:
    """Classify tickets concurrently (Week 2 Day 3 pattern), results in input order."""
    limit = asyncio.Semaphore(concurrency)

    async def one(ticket: Ticket) -> ClassifyResult:
        async with limit:
            return await classify(client, ticket, system, model, use_schema)

    return await asyncio.gather(*(one(t) for t in tickets))
