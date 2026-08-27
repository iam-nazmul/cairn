# Python is pinned to the version the project is developed and tested against;
# uv.lock is resolved for it. See .python-version.
FROM python:3.14-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependencies first, in their own layer: they change far less often than src/,
# so editing a node does not re-resolve the whole environment. Deliberately no
# BuildKit cache mounts -- this must build on the classic builder too.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
RUN uv sync --locked --no-dev

# Run as a non-root user.
RUN useradd --create-home --uid 10001 cairn && chown -R cairn:cairn /app
USER cairn

EXPOSE 8000

# --no-sync: the environment is already built; do not touch it at boot.
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api.routes:app", \
     "--host", "0.0.0.0", "--port", "8000"]
