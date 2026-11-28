"""Resolução segura dos arquivos gerados pela classificação."""

import re
from pathlib import Path

from backend.app.config import get_settings

SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_\-]+$")

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
