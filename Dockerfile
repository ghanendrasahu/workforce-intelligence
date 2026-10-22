# syntax=docker/dockerfile:1
# Hugging Face Space image for the workforce API.
#
# Unlike apps/api/Dockerfile (which expects artifacts to be mounted at run
# time), Spaces cannot mount volumes - so the synthetic dataset is generated
# and the models are trained during the image build. Every restart boots the
# pre-trained artifact set in seconds.
FROM python:3.11-slim AS wheels

ENV PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.lock ./requirements.lock
RUN grep -vE '^(pytest==|pytest-cov==|coverage==|ruff==|mypy==|mypy-extensions==)' \
        requirements.lock > runtime.lock \
    && pip wheel --wheel-dir /wheels -r runtime.lock

FROM python:3.11-slim AS artifacts

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY --from=wheels /wheels /wheels
COPY --from=wheels /build/runtime.lock ./runtime.lock
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r runtime.lock

COPY pyproject.toml README.md ./
COPY core ./core
COPY data ./data
COPY ml ./ml
COPY ai ./ai
COPY apps/api ./apps/api
RUN pip install --no-cache-dir --no-deps .

RUN python -m data.ingestion.synthetic_generator
RUN python -m ml.training.train

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=artifacts /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=artifacts /usr/local/bin /usr/local/bin
COPY --from=artifacts /app/core ./core
COPY --from=artifacts /app/data ./data
COPY --from=artifacts /app/ml ./ml
COPY --from=artifacts /app/ai ./ai
COPY --from=artifacts /app/apps/api ./apps/api
COPY --from=artifacts /app/artifacts ./artifacts

RUN chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)" || exit 1

EXPOSE 7860

CMD ["uvicorn", "apps.api.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
