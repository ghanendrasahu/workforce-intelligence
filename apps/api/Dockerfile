FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY core ./core
COPY data ./data
COPY ml ./ml
COPY ai ./ai
COPY apps/api ./apps/api

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "apps.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
