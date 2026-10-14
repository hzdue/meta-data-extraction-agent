# Build with the staged runtime directory as context, never the repository.
# This recipe is unvalidated until an operator completes a local Docker build.
FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.13.7-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY . /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_CACHE_DIR=/tmp/uv-cache
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13.7-slim
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1
ENV LANGCHAIN_TRACING_V2=false LANGSMITH_TRACING=false
WORKDIR /app
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin pilot \
    && mkdir -p /app/runs/agent \
    && chmod -R a-w /app \
    && chown -R 10001:10001 /app/runs \
    && chmod 700 /app/runs /app/runs/agent
USER 10001:10001
# META_AGENT_PILOT_TOKEN must be supplied at launch; no key or token is baked in.
# Loopback only. Linux --network host is required for host-browser access.
CMD ["python", "-m", "meta_data_extraction.agent.pilot_http", "serve", "--root", "/app", "--host", "127.0.0.1", "--port", "8765"]
