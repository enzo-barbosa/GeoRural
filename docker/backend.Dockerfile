# Imagem de desenvolvimento/produção do backend.
# O ambiente Python fica em /opt/venv para não ser sombreado pelo bind mount
# do repositório no Docker Compose (importante no Windows).
FROM python:3.13-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# OpenCV (cv2) precisa destas bibliotecas no slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Camada de dependências: só é refeita quando pyproject/uv.lock mudam.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --all-extras --no-install-project

COPY . .
RUN uv sync --frozen --all-extras

ENV PATH="/opt/venv/bin:${PATH}"

RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app /opt/venv
USER app

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
