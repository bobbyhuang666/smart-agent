FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || pip install --no-cache-dir requests aiohttp

# Copy source
COPY scripts/ scripts/
COPY tests/ tests/
COPY config.example.yaml config.yaml

# Environment
ENV PYTHONUNBUFFERED=1
ENV TASK_ROUTER_CACHE=/data/cache
VOLUME ["/data/cache"]

EXPOSE 8930

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8930/api/health')" || exit 1

CMD ["python", "scripts/api_server.py", "--host", "0.0.0.0", "--port", "8930"]
