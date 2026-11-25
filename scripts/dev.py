#!/usr/bin/env python3
"""Atalhos cross-platform para o ambiente Docker do GeoRural.

Funciona em Linux, macOS e Windows (PowerShell/cmd) porque só usa a biblioteca
padrão e o executável ``docker``.

Exemplos:
    uv run python scripts/dev.py up
    uv run python scripts/dev.py logs backend
    uv run python scripts/dev.py models
    uv run python scripts/dev.py test
    uv run python scripts/dev.py rebuild
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIALS = Path.home() / ".config" / "earthengine"


def _env() -> dict[str, str]:
    """Prepara o ambiente do Compose, detectando as credenciais do GEE."""
    env = os.environ.copy()

    credentials = Path(
        env.get("GEE_CREDENTIALS_DIR", str(DEFAULT_CREDENTIALS))
    ).expanduser()
    if credentials.exists():
        env["GEE_CREDENTIALS_DIR"] = str(credentials)
    else:
        print(
            f"[aviso] Credenciais do GEE não encontradas em {credentials}.\n"
            "        Rode `earthengine authenticate` no host ou defina "
            "GEE_CREDENTIALS_DIR.",
            file=sys.stderr,
        )
    return env


def _run(args: list[str], *, check: bool = True) -> int:
    """Executa `docker compose ...` na raiz do projeto."""
    command = ["docker", "compose", *args]
    print("$", " ".join(command))
    try:
        result = subprocess.run(command, cwd=ROOT, env=_env(), check=False)
    except FileNotFoundError:
        print(
            "Docker não encontrado. Instale o Docker Desktop/Engine.",
            file=sys.stderr,
        )
        return 1
    if check and result.returncode != 0:
        return result.returncode
    return result.returncode


def _require_env() -> bool:
    """Confere se o .env existe antes de subir os serviços."""
    if (ROOT / ".env").exists():
        return True
    print(
        "Arquivo .env não encontrado.\n"
        "Copie .env.example para .env e preencha GEE_PROJECT_ID:\n"
        "  Linux/macOS: cp .env.example .env\n"
        "  Windows:     Copy-Item .env.example .env",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="Sobe banco, backend e frontend")
    up.add_argument("-d", "--detach", action="store_true", help="Em background")

    down = sub.add_parser("down", help="Derruba os serviços")
    down.add_argument(
        "-v", "--volumes", action="store_true", help="Remove também os volumes"
    )

    logs = sub.add_parser("logs", help="Acompanha os logs")
    logs.add_argument("service", nargs="?", help="db, backend ou frontend")

    sub.add_parser("rebuild", help="Reconstrói as imagens e recria os containers")
    sub.add_parser("config", help="Valida o compose.yml")
    sub.add_parser("test", help="Roda o pytest dentro do backend")
    sub.add_parser("lint", help="Roda o ruff dentro do backend")
    sub.add_parser("shell", help="Abre um shell no backend")
    sub.add_parser("psql", help="Abre o psql no banco")
    sub.add_parser("models", help="Baixa os modelos via GitHub Release")
    sub.add_parser("migrate", help="Aplica as migrações do banco (alembic)")

    args = parser.parse_args(argv)

    if args.command == "up":
        if not _require_env():
            return 1
        extra = ["-d"] if args.detach else []
        return _run(["up", "--build", *extra])

    if args.command == "down":
        extra = ["-v"] if args.volumes else []
        return _run(["down", *extra])

    if args.command == "logs":
        extra = [args.service] if args.service else []
        return _run(["logs", "-f", *extra])

    if args.command == "rebuild":
        if not _require_env():
            return 1
        code = _run(["build", "--pull"])
        if code != 0:
            return code
        return _run(["up", "-d", "--force-recreate"])

    if args.command == "config":
        return _run(["config"])

    if args.command == "models":
        return _run(["--profile", "tools", "run", "--rm", "download-models"])

    if args.command == "migrate":
        return _run(["exec", "backend", "alembic", "upgrade", "head"])

    if args.command == "test":
        return _run(["exec", "backend", "python", "-m", "pytest", "-q"])

    if args.command == "lint":
        return _run(
            ["exec", "backend", "ruff", "check", "src", "backend", "tests", "scripts"]
        )

    if args.command == "shell":
        return _run(["exec", "backend", "bash"])

    if args.command == "psql":
        user = os.environ.get("POSTGRES_USER", "georural")
        database = os.environ.get("POSTGRES_DB", "georural")
        return _run(["exec", "db", "psql", "-U", user, "-d", database])

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
