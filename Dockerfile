FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY backend ./backend
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY backend ./backend
COPY alembic.ini ./alembic.ini
COPY data/reports.yaml ./data/reports.yaml
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080
CMD ["sh", "-c", "uvicorn elderhelp.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
