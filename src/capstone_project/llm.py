"""Call Claude through the DataRobot LLM gateway (OpenAI-compatible chat completions).

Settings come from environment variables, never from code:
    DATAROBOT_API_TOKEN      required  personal API key (DataRobot > API keys and tools)
    LLM_GATEWAY_BASE_URL     optional  default: staging gateway
    LLM_MODEL                optional  default: anthropic/claude-haiku-4-5-20251001
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI, OpenAI

DEFAULT_BASE_URL = "https://staging.datarobot.com/api/v2/genai/llmgw"
DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
# The openai library waits up to 10 minutes by default; a stuck request should fail fast
# instead (Week 2 Day 3: always set a timeout). Timeouts and 5xx are retried twice.
DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 2

# Anthropic list prices in USD per 1M tokens (input, output). Used only for an estimate;
# what the gateway meters internally may differ.
PRICE_PER_MTOK = {
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
}


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str            # model name the gateway reports back
    input_tokens: int     # usage.prompt_tokens
    output_tokens: int    # usage.completion_tokens
    seconds: float
    finish_reason: str

    def estimated_cost_usd(self, requested_model: str) -> float | None:
        prices = PRICE_PER_MTOK.get(requested_model)
        if prices is None:
            return None
        return (self.input_tokens * prices[0] + self.output_tokens * prices[1]) / 1_000_000


def make_client(
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.Client | None = None,
) -> OpenAI:
    api_key = api_key or os.environ.get("DATAROBOT_API_TOKEN")
    if not api_key:
        raise RuntimeError(
            "DATAROBOT_API_TOKEN is not set. Create a personal API key in DataRobot "
            "(API keys and tools) and export it; do not write it in code."
        )
    return OpenAI(
        api_key=api_key,
        base_url=base_url or os.environ.get("LLM_GATEWAY_BASE_URL", DEFAULT_BASE_URL),
        http_client=http_client,
        timeout=DEFAULT_TIMEOUT,
        max_retries=MAX_RETRIES,
    )


def make_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> AsyncOpenAI:
    """Same settings as make_client(), for asyncio code."""
    sync = make_client(api_key, base_url)  # reuses the env-var checks
    return AsyncOpenAI(api_key=sync.api_key, base_url=sync.base_url, http_client=http_client,
                       timeout=DEFAULT_TIMEOUT, max_retries=MAX_RETRIES)


def complete(
    client: OpenAI,
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 300,
    temperature: float = 0.0,
) -> LLMResult:
    """One request: system prompt + one user message -> text and token counts."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model or os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.choices[0]
    return LLMResult(
        text=choice.message.content or "",
        model=response.model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        seconds=time.perf_counter() - start,
        finish_reason=choice.finish_reason,
    )
