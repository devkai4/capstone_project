"""Day 3: fetch 20 URLs with httpx, sync and async, and compare.

    uv run python scripts/day3_fetch.py          # local test API: 17 OK + timeout + 404 + bad JSON
    uv run python scripts/day3_fetch.py --hf     # real Hugging Face API: 20 pages of tickets

The local test API runs inside this script (standard library http.server), so the
timeout / 404 / bad-JSON cases happen the same way every time and need no internet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from capstone_project.http_fetch import FetchResult, fetch_all_async, fetch_all_sync

LATENCY = 0.3        # every normal local request takes this long
SLOW = 3.0           # /slow answers after this long ...
TIMEOUT = 1.0        # ... but the client gives up after this long
HF_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=Tobi-Bueck/customer-support-tickets"
           "&config=default&split=train&offset={offset}&length=5")


class TestAPI(BaseHTTPRequestHandler):
    """GET /tickets/<n> -> JSON, /missing -> 404, /broken -> invalid JSON, /slow -> too slow."""

    def do_GET(self) -> None:
        try:
            if self.path.startswith("/tickets/"):
                time.sleep(LATENCY)
                n = self.path.rsplit("/", 1)[1]
                self._send(200, json.dumps({"id": n, "queue": "Technical Support"}))
            elif self.path == "/broken":
                time.sleep(LATENCY)
                self._send(200, '{"id": 1, "queue": ')  # cut off -> not valid JSON
            elif self.path == "/slow":
                time.sleep(SLOW)
                self._send(200, "{}")
            else:
                time.sleep(LATENCY)
                self._send(404, json.dumps({"error": "not found"}))
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client already gave up (timeout); nothing to do

    def _send(self, status: int, body: str) -> None:
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # keep the server quiet
        pass


class TestServer(ThreadingHTTPServer):
    # Pending-connection queue (listen backlog). The default of 5 is smaller than the
    # 10 concurrent async connections: the extra connects get dropped and retried by the
    # OS after ~1s, which shows up as ConnectTimeout on a 1s client timeout.
    request_queue_size = 64
    daemon_threads = True


def start_test_api() -> str:
    server = TestServer(("127.0.0.1", 0), TestAPI)  # port 0 = any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}"


def print_results(results: list[FetchResult]) -> None:
    for r in results:
        status = r.status if r.status is not None else "-"
        print(f"  {'OK ' if r.ok else 'ERR'}  {status!s:>3}  {r.seconds:5.2f}s  "
              f"{r.error or ''}  {r.url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf", action="store_true", help="use the real Hugging Face API")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="  log %(levelname)s %(message)s")

    if args.hf:
        urls = [HF_ROWS.format(offset=i * 5) for i in range(20)]
        timeout = 15.0
    else:
        base = start_test_api()
        urls = [f"{base}/tickets/{i}" for i in range(1, 18)]
        urls += [f"{base}/slow", f"{base}/missing", f"{base}/broken"]
        timeout = TIMEOUT

    print(f"== sync: {len(urls)} requests one after another ==")
    start = time.perf_counter()
    sync_results = fetch_all_sync(urls, timeout=timeout)
    sync_s = time.perf_counter() - start

    print(f"\n== async: {len(urls)} requests concurrently ==")
    start = time.perf_counter()
    async_results = asyncio.run(fetch_all_async(urls, timeout=timeout))
    async_s = time.perf_counter() - start

    print("\nasync results (same order as the URL list):")
    print_results(async_results)

    speedup = sync_s / async_s
    print(f"\nsync  {sync_s:5.2f}s   {dict(Counter(r.error or 'ok' for r in sync_results))}")
    print(f"async {async_s:5.2f}s   {dict(Counter(r.error or 'ok' for r in async_results))}")
    print(f"speedup {speedup:.1f}x")

    if args.hf:
        ok = speedup >= 3
    else:
        expected = {"ok": 17, "timeout": 1, "http_404": 1, "bad_json": 1}
        ok = (speedup >= 3
              and Counter(r.error or "ok" for r in sync_results) == expected
              and Counter(r.error or "ok" for r in async_results) == expected)
    print(f"\n{'PASS' if ok else 'FAIL'}: errors handled without traceback, "
          f"async is {speedup:.1f}x faster (need >= 3x)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
