"""Resolução segura dos arquivos gerados pela classificação."""

import re
import shutil
from pathlib import Path

from backend.app.config import get_settings

SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_\-]+$")

# Formato gerado por new_job_id(); usado pela limpeza de órfãos.
GENERATED_JOB_ID = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")

# Allowlist: o nome do arquivo pedido nunca é usado diretamente no caminho.
ALLOWED_OUTPUT_FILES = frozenset(
    {
        "classification.png",
        "rgb.png",
        "overlay.png",
        "slope.png",
        "relief.png",
        "crop.png",
    }
)


def resolve_output_file(job_id: str, filename: str) -> Path | None:
    """Retorna o caminho de um PNG permitido do job, se ele existir.

    O ``job_id`` deve vir do banco (já validado contra o usuário) e o nome do
    arquivo precisa estar na allowlist. Qualquer tentativa de path traversal
    retorna ``None``.
    """
    if not SAFE_JOB_ID.fullmatch(job_id):
        return None
    if filename not in ALLOWED_OUTPUT_FILES:
        return None

    base = Path(get_settings().OUTPUTS_DIR).resolve()
    filepath = (base / job_id / filename).resolve()
    if base not in filepath.parents or not filepath.is_file():
        return None
    return filepath


def resolve_output_directory(job_id: str) -> Path | None:
    """Retorna o diretório de saída do job, se ele estiver dentro da base."""
    if not SAFE_JOB_ID.fullmatch(job_id):
        return None

    base = Path(get_settings().OUTPUTS_DIR).resolve()
    candidate = base / job_id
    if candidate.is_symlink():
        return None

    directory = candidate.resolve()
    if directory.parent != base:
        return None
    return directory


def delete_output_directory(job_id: str) -> bool:
    """Remove a pasta de saída do job; retorna ``False`` se ela não existir."""
    directory = resolve_output_directory(job_id)
    if directory is None or not directory.is_dir():
        return False
    shutil.rmtree(directory)
    return True

