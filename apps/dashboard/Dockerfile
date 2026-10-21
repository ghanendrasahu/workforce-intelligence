# Dashboard image: two-stage, non-root, health-checked Streamlit.
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Same runtime set as the API (filtered lock), plus the app itself.
COPY requirements.lock ./requirements.lock
RUN grep -vE '^(pytest==|pytest-cov==|coverage==|ruff==|mypy==|mypy-extensions==)' \
        requirements.lock > runtime.lock \
    && pip install --no-cache-dir -r runtime.lock

COPY pyproject.toml README.md ./
COPY core ./core
COPY data ./data
COPY ml ./ml
COPY ai ./ai
COPY apps/dashboard ./apps/dashboard
RUN pip install --no-cache-dir --no-deps .

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
COPY --from=build /app/apps/dashboard ./apps/dashboard

RUN chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)" || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "apps/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
