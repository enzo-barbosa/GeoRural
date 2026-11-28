#!/usr/bin/env python3
"""Remove diretórios de saída sem análise correspondente no banco.

Exemplos:
    # só mostra o que seria removido (padrão)
    uv run python scripts/cleanup_outputs.py --dry-run

    # remove de verdade
    uv run python scripts/cleanup_outputs.py --yes

    # remove apenas órfãos com mais de 30 dias
    uv run python scripts/cleanup_outputs.py --yes --older-than-days 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEE_PROJECT_ID", "maintenance")

from sqlalchemy import select

from backend.app.config import get_settings
from backend.app.db.models import Analysis
from backend.app.db.session import SessionLocal
from backend.app.services.output_cleanup import (
    delete_orphan_outputs,
    find_orphan_outputs,
)


def _fmt_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirma a remoção; sem isso apenas simula.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista os órfãos (comportamento padrão).",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        help="Considera apenas diretórios mais antigos que N dias.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Diretório de saída (padrão: OUTPUTS_DIR).",
    )
    args = parser.parse_args(argv)

    if args.older_than_days is not None and args.older_than_days < 0:
        parser.error("--older-than-days não pode ser negativo.")

    base_dir = Path(args.outputs_dir or get_settings().OUTPUTS_DIR)

    with SessionLocal() as db:
        known_ids = set(db.scalars(select(Analysis.id)).all())

    orphans = find_orphan_outputs(
        base_dir,
        known_ids,
        older_than_days=args.older_than_days,
    )
    if not orphans:
        print(f"Nenhum diretório órfão em {base_dir}.")
        return 0

    total_bytes = sum(item.size_bytes for item in orphans)
    print(
        f"{len(orphans)} diretório(s) órfão(s), "
        f"{_fmt_size(total_bytes)} em {base_dir}:"
    )
    for item in orphans:
        print(
            f"  - {item.path.name}  "
            f"({_fmt_size(item.size_bytes)}, "
            f"modificado em {item.modified_at:%d/%m/%Y %H:%M} UTC)"
        )

    should_delete = args.yes and not args.dry_run
    if not should_delete:
        print("Nada foi removido. Use --yes para excluir.")
        return 0

    deleted = delete_orphan_outputs(base_dir, orphans)
    print(f"{deleted} diretório(s) removido(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
