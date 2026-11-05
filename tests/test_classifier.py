import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from capstone_project.classifier import (
    TicketLabels,
    classify_many,
    load_prompt,
    response_format,
    user_message,
)
from capstone_project.llm import make_async_client
from capstone_project.records import Ticket

TICKET = Ticket(subject="VPN down", body="We cannot\nconnect.", queue="Technical Support",
                priority="high", type="Incident", language="en")


# ---------- the Pydantic model ----------

@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"queue": "IT Support", "priority": "low", "type": "Change"}', id="exact"),
        pytest.param('{"queue": "it support", "priority": "LOW", "type": "change"}', id="case-only-diff"),
    ],
)
def test_valid_json_parses_to_canonical_labels(raw):
    labels = TicketLabels.model_validate_json(raw)
    assert labels.model_dump() == {"queue": "IT Support", "priority": "low", "type": "Change"}


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"queue": "Network Team", "priority": "low", "type": "Change"}', id="unknown-queue"),
        pytest.param('{"queue": "IT Support", "priority": "urgent", "type": "Change"}', id="unknown-priority"),
        pytest.param('{"queue": "IT Support", "priority": "low"}', id="missing-type"),
        pytest.param('{"queue": "IT Support", "priority": "low", "type": "Change", "why": "x"}',
                     id="extra-key"),
        pytest.param('```json\n{"queue": "IT Support", "priority": "low", "type": "Change"}\n```',
                     id="markdown-fence"),
        pytest.param('{"queue": "IT Support", "priority": "low", "ty', id="truncated"),
    ],
)
def test_invalid_output_is_rejected(raw):
    with pytest.raises(ValidationError):
        TicketLabels.model_validate_json(raw)


def test_response_format_schema():
    fmt = response_format()
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["queue", "priority", "type"]
    assert schema["properties"]["priority"]["enum"] == ["high", "medium", "low"]
    assert "title" not in json.dumps(schema)


# ---------- the prompt ----------

def test_prompt_v1_has_five_examples_and_every_label():
    prompt = load_prompt("prompts/classifier_v1.txt")
    assert prompt.count("<example>") == 5
    for value in ("Technical Support", "Sales and Pre-Sales", "medium", "Change"):
        assert value in prompt


def test_user_message_wraps_cleaned_ticket():
    assert user_message(TICKET) == (
        "<ticket>\n<subject>VPN down</subject>\n<body>We cannot connect.</body>\n</ticket>"
    )


# ---------- calling the gateway (fake) ----------

def fake_gateway(replies, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        content = replies[len(seen) - 1]
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0,
            "model": "claude-haiku-4-5-20251001",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920},
        })
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_classify_many_parses_and_reports_failures():
    seen = []
    replies = ['{"queue": "IT Support", "priority": "low", "type": "Change"}',
               'Sure! {"queue": "IT Support"}']
    client = make_async_client(api_key="k", base_url="https://gw.test/llmgw",
                               http_client=fake_gateway(replies, seen))
    results = asyncio.run(classify_many(client, [TICKET, TICKET], "SYSTEM", concurrency=1))

    assert results[0].labels.queue == "IT Support" and results[0].error is None
    assert results[1].labels is None and results[1].error.startswith("json_invalid")
    assert results[0].input_tokens == 900
    assert seen[0]["response_format"]["json_schema"]["name"] == "ticket_labels"
    assert seen[0]["messages"][0] == {"role": "system", "content": "SYSTEM"}


def test_no_schema_mode_sends_no_response_format():
    seen = []
    client = make_async_client(api_key="k", base_url="https://gw.test/llmgw",
                               http_client=fake_gateway(['{"queue": "IT Support", "priority": "low", "type": "Change"}'], seen))
    asyncio.run(classify_many(client, [TICKET], "SYSTEM", use_schema=False))
    assert "response_format" not in seen[0]
