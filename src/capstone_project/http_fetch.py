"""Fetch JSON from many URLs with httpx, sync and async, without crashing on bad responses.

Every request ends as a FetchResult. Failures are logged as ONE line (no traceback)
and sorted into a small set of error kinds:

    timeout      - no response within the timeout
    http_404     - (or http_500 etc.) the server answered with an error status
    bad_json     - status 200 but the body is not valid JSON
    request_error- DNS failure, connection refused, ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    url: str
    ok: bool
    status: int | None = None
    data: Any = None
    error: str | None = None  # "timeout", "http_404", "bad_json", "request_error"
    detail: str = ""
    seconds: float = 0.0


def _from_response(url: str, response: httpx.Response, start: float) -> FetchResult:
    """Turn a received response into a result: check the status first, then parse JSON."""
    seconds = time.perf_counter() - start
    if response.is_error:  # 4xx / 5xx
        return _fail(url, f"http_{response.status_code}", response.reason_phrase,
                     seconds, response.status_code)
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        snippet = response.text[:40].replace("\n", " ")
        return _fail(url, "bad_json", f"{e.msg} (body starts {snippet!r})",
                     seconds, response.status_code)
    return FetchResult(url, True, response.status_code, data, seconds=seconds)


def _from_exception(url: str, exc: httpx.HTTPError, start: float) -> FetchResult:
    """Turn a transport-level exception (no response at all) into a result."""
    seconds = time.perf_counter() - start
    kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "request_error"
    return _fail(url, kind, type(exc).__name__, seconds)


def _fail(url: str, kind: str, detail: str, seconds: float, status: int | None = None) -> FetchResult:
    # logger.warning (not logger.exception) -> one readable line, no traceback
    logger.warning("%-8s %s (%s, %.2fs)", kind, url, detail, seconds)
    return FetchResult(url, False, status, error=kind, detail=detail, seconds=seconds)


# ---------- sync ----------

def fetch_json(client: httpx.Client, url: str) -> FetchResult:
    start = time.perf_counter()
    try:
        response = client.get(url)
    except httpx.HTTPError as e:
        return _from_exception(url, e, start)
    return _from_response(url, response, start)


def fetch_all_sync(
    urls: list[str], timeout: float = 5.0, transport: httpx.BaseTransport | None = None
) -> list[FetchResult]:
    """One request after another: total time ~= sum of all request times."""
    with httpx.Client(timeout=timeout, transport=transport) as client:
        return [fetch_json(client, url) for url in urls]


# ---------- async ----------

async def afetch_json(client: httpx.AsyncClient, url: str, limit: asyncio.Semaphore) -> FetchResult:
    async with limit:  # at most N requests in flight
        start = time.perf_counter()
        try:
            response = await client.get(url)
        except httpx.HTTPError as e:
            return _from_exception(url, e, start)
        return _from_response(url, response, start)


async def fetch_all_async(
    urls: list[str],
    timeout: float = 5.0,
    max_concurrency: int = 10,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[FetchResult]:
    """All requests in flight together: total time ~= the slowest request (per batch)."""
    limit = asyncio.Semaphore(max_concurrency)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        # gather keeps the input order in its result list
        return await asyncio.gather(*(afetch_json(client, url, limit) for url in urls))
