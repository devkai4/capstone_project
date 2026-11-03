"""Offline tests: httpx.MockTransport plays the server, so no network is needed."""

import asyncio
import logging
import time

import httpx

from capstone_project.http_fetch import fetch_all_async, fetch_all_sync

URLS = [
    "https://api.test/ok",
    "https://api.test/missing",
    "https://api.test/broken",
    "https://api.test/slow",
    "https://api.test/down",
]


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/ok":
        return httpx.Response(200, json={"queue": "Technical Support"})
    if path == "/missing":
        return httpx.Response(404, json={"error": "not found"})
    if path == "/broken":
        return httpx.Response(200, text='{"queue": ')
    if path == "/slow":
        raise httpx.ReadTimeout("timed out", request=request)
    raise httpx.ConnectError("connection refused", request=request)


def check(results, caplog):
    assert [r.url for r in results] == URLS  # order preserved
    assert [r.error for r in results] == [None, "http_404", "bad_json", "timeout", "request_error"]
    assert results[0].ok and results[0].data == {"queue": "Technical Support"}
    assert results[1].status == 404
    assert results[2].status == 200 and not results[2].ok

    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 4  # one line per failure
    assert all(rec.exc_info is None for rec in warnings)  # no traceback attached
    assert "Traceback" not in caplog.text


def test_sync_handles_every_error_kind(caplog):
    results = fetch_all_sync(URLS, transport=httpx.MockTransport(handler))
    check(results, caplog)


def test_async_handles_every_error_kind(caplog):
    results = asyncio.run(fetch_all_async(URLS, transport=httpx.MockTransport(handler)))
    check(results, caplog)


def test_async_requests_overlap():
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={})

    urls = [f"https://api.test/{i}" for i in range(10)]
    start = time.perf_counter()
    results = asyncio.run(fetch_all_async(urls, transport=httpx.MockTransport(slow_handler)))
    elapsed = time.perf_counter() - start

    assert all(r.ok for r in results)
    assert elapsed < 1.0  # 10 x 0.2s would be 2.0s if they ran one after another


def test_max_concurrency_limits_requests_in_flight():
    in_flight = peak = 0

    async def counting_handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, json={})

    urls = [f"https://api.test/{i}" for i in range(12)]
    asyncio.run(fetch_all_async(urls, max_concurrency=3,
                                transport=httpx.MockTransport(counting_handler)))
    assert peak == 3
