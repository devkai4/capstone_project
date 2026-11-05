# Ingest CLI image.
#   docker build -t capstone-ingest .
#   docker run --rm -v "$PWD/out:/out" capstone-ingest                 # ingest 100 tickets -> out/tickets_100.json
#   docker run --rm capstone-ingest pytest -q tests/test_ingest.py     # 10 tests inside the container

ARG PYTHON_VERSION=3.12

# ---------- builder: install the locked dependencies + the project into /app/.venv ----------
FROM python:${PYTHON_VERSION}-slim AS builder

# uv as a single static binary from its official image (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) Dependencies only. This layer is rebuilt only when pyproject.toml or uv.lock change,
#    so editing source code does not reinstall packages. --locked fails if uv.lock is stale.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# 2) The project itself (source, tests, scripts; see .dockerignore for what is left out).
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# ---------- runtime: same Python, no uv, non-root user ----------
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTEST_ADDOPTS="-p no:cacheprovider"

ARG UID=10001
RUN useradd --uid "${UID}" --no-create-home --shell /usr/sbin/nologin appuser \
 && mkdir /out && chown appuser /out

WORKDIR /app
COPY --from=builder /app /app

USER appuser
CMD ["capstone-ingest", "--out", "/out/tickets_100.json"]
