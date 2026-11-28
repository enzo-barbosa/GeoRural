"""Identificação e remoção de diretórios de saída sem análise no banco."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from backend.app.services.output_files import GENERATED_JOB_ID


@dataclass(frozen=True)
class OrphanOutput:
    """Diretório de saída sem análise correspondente."""

    path: Path
    size_bytes: int
    modified_at: datetime


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def find_orphan_outputs(
    base_dir: Path,
    known_job_ids: set[str],
    *,
    older_than_days: int | None = None,
) -> list[OrphanOutput]:
    """Lista pastas de jobs gerados que não têm linha em ``analyses``.

    Só considera diretórios com o formato de ``new_job_id()``. Arquivos soltos
    e diretórios com outros nomes são ignorados.
    """
    base = base_dir.resolve()
    if not base.is_dir():
        return []

    cutoff = None
    if older_than_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)

    orphans: list[OrphanOutput] = []
    for entry in sorted(base.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        if not GENERATED_JOB_ID.fullmatch(entry.name):
            continue
        if entry.name in known_job_ids:
            continue

        modified_at = datetime.fromtimestamp(entry.stat().st_mtime, tz=UTC)
        if cutoff is not None and modified_at > cutoff:
            continue

        orphans.append(
            OrphanOutput(
                path=entry,
                size_bytes=_directory_size(entry),
                modified_at=modified_at,
            )
        )
    return orphans


def delete_orphan_outputs(
    base_dir: Path, orphans: list[OrphanOutput]
) -> int:
    """Remove os diretórios informados, validando que estão dentro da base."""
    base = base_dir.resolve()
    deleted = 0
    for orphan in orphans:
        if orphan.path.is_symlink():
            continue
        path = orphan.path.resolve()
        if path.parent != base or not path.is_dir():
            continue
        shutil.rmtree(path)
        deleted += 1
    return deleted
