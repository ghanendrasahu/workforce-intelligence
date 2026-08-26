FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY core ./core
COPY data ./data
COPY ml ./ml
COPY ai ./ai
COPY apps/dashboard ./apps/dashboard

RUN pip install --no-cache-dir .

EXPOSE 8501

CMD ["streamlit", "run", "apps/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
