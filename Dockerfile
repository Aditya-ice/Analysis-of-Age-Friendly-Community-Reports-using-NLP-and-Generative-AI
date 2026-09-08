FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY backend/elderhelp ./backend/elderhelp
RUN uv sync --frozen --no-dev --no-editable
COPY contracts/reranker-manifest.json ./contracts/reranker-manifest.json
COPY tools/download_reranker.py ./tools/download_reranker.py
RUN /app/.venv/bin/python tools/download_reranker.py
ENV TIKTOKEN_CACHE_DIR=/app/models/tokenizer
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

FROM python:3.12-slim
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/models /app/models
ENV PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/app/models/tokenizer \
    PYTHONUNBUFFERED=1 \
    PORT=8080
USER 65532:65532
CMD ["sh", "-c", "uvicorn elderhelp.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --limit-concurrency 32 --no-access-log"]
