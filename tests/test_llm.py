"""LLM gateway client tests. The fake server replays a real gateway response shape."""

import json

import httpx
import pytest

from capstone_project.llm import DEFAULT_MODEL, complete, make_client

# Trimmed copy of a real staging gateway response (2026-11-05)
GATEWAY_RESPONSE = {
    "id": "chatcmpl-9e2eeef8-867c-487a-b2ee-cbfcb2ae7717",
    "object": "chat.completion",
    "created": 1793857879,
    "model": "claude-haiku-4-5-20251001",
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "Technical Support"},
    }],
    "usage": {"prompt_tokens": 22, "completion_tokens": 4, "total_tokens": 26},
}


def fake_gateway(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GATEWAY_RESPONSE)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_complete_returns_text_and_token_counts():
    seen = []
    client = make_client(api_key="test-key", base_url="https://gw.test/llmgw",
                         http_client=fake_gateway(seen))
    result = complete(client, system="Route tickets.", user="<ticket>VPN down</ticket>")

    assert result.text == "Technical Support"
    assert (result.input_tokens, result.output_tokens) == (22, 4)
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.finish_reason == "stop"
    assert result.estimated_cost_usd(DEFAULT_MODEL) == pytest.approx((22 * 1 + 4 * 5) / 1e6)


def test_request_shape():
    seen = []
    client = make_client(api_key="test-key", base_url="https://gw.test/llmgw",
                         http_client=fake_gateway(seen))
    complete(client, system="S", user="U", max_tokens=20)

    request = seen[0]
    assert str(request.url) == "https://gw.test/llmgw/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == DEFAULT_MODEL
    assert body["max_tokens"] == 20
    assert body["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]


def test_missing_token_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("DATAROBOT_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="DATAROBOT_API_TOKEN is not set"):
        make_client()
