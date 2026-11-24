# Imagem de produção: build do React + FastAPI servindo o dist.
FROM node:22-alpine AS frontend

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# O Vite escreve em ../backend/frontend/dist (ou seja, /backend/frontend/dist).
RUN npm run build


FROM python:3.13-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --extra backend --no-install-project

COPY . .
COPY --from=frontend /backend/frontend/dist ./backend/frontend/dist
RUN uv sync --frozen --extra backend

ENV PATH="/opt/venv/bin:${PATH}"

RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app /opt/venv
USER app

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
