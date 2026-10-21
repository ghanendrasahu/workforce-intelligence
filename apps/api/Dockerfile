# API image: two-stage, non-root, health-checked.
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the lock and filter dev tooling out of the runtime set.
COPY requirements.lock ./requirements.lock
RUN grep -vE '^(pytest==|pytest-cov==|coverage==|ruff==|mypy==|mypy-extensions==)' \
        requirements.lock > runtime.lock \
    && pip install --no-cache-dir -r runtime.lock

# Install the application itself with no re-resolution.
COPY pyproject.toml README.md ./
COPY core ./core
COPY data ./data
COPY ml ./ml
COPY ai ./ai
COPY apps/api ./apps/api
RUN pip install --no-cache-dir --no-deps .

# Runtime: libraries + sources only.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app/core ./core
COPY --from=build /app/data ./data
COPY --from=build /app/ml ./ml
COPY --from=build /app/ai ./ai
COPY --from=build /app/apps/api ./apps/api

# Artifacts (data/models/metrics) are mounted read-only at run time.
RUN chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

EXPOSE 8000

CMD ["uvicorn", "apps.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
