# sora in a container: the dashboard, chat, memory and evals.
#
#   docker compose up -d --build        then open http://127.0.0.1:8770
#
# Your data (memory, traces, lessons, workspace, the embedding model) lives in a volume
# at /data, never in the image. Keys come from .env when the container starts
# (compose.yaml), never baked in: .dockerignore keeps .env out of the build.
#
# Off in here, because they need the Mac or the host's Docker: Apple Notes and
# Calendar, the browser tool, and run_code (it starts containers, and mounting the
# host's Docker socket would hand this one root on the host). Code still runs in the
# provider's own sandbox when SORA_CODE_EXECUTION is on.

FROM ghcr.io/astral-sh/uv:0.12.0 AS uv

FROM python:3.13-slim-bookworm AS build
COPY --from=uv /uv /bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# providers: Hugging Face / OpenAI-compatible models; otel: Phoenix / Langfuse; memory-pg: pgvector.
ARG EXTRAS="--extra providers --extra otel --extra memory-pg"
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project $EXTRAS
COPY . .

FROM python:3.13-slim-bookworm
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin sora \
    && mkdir /data && chown sora:sora /data
COPY --from=build --chown=sora:sora /app /app
# SORA_BIND=0.0.0.0 so Docker can forward to it; compose.yaml publishes the port on the
# host's 127.0.0.1 only. There's no login yet: don't publish it to the network.
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    SORA_HOME=/data SORA_MODELS_DIR=/data/models SORA_BIND=0.0.0.0 SORA_PORT=8765 \
    SORA_SANDBOX=off SORA_NOTES=0 SORA_CALENDAR=0 SORA_BROWSER=0
USER sora
WORKDIR /app
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=4)"
CMD ["python", "main.py", "serve"]
