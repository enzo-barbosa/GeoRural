"""Entrega de imagens geradas pela classificação, protegidas por usuário."""

import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.db.models import User
from backend.app.db.repositories import get_analysis_for_user
from backend.app.db.session import get_db
from backend.app.dependencies.auth import get_current_user

router = APIRouter(tags=["outputs"])

_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_\-]+$")

# Allowlist: o nome do arquivo pedido nunca é usado diretamente no caminho.
_ALLOWED_FILES = {
    "classification.png": "classification.png",
    "rgb.png": "rgb.png",
    "overlay.png": "overlay.png",
    "slope.png": "slope.png",
    "relief.png": "relief.png",
    "crop.png": "crop.png",
}

_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND, detail="Arquivo não encontrado"
)


@router.get("/outputs/{job_id}/{filename}")
def get_output_file(
    job_id: str,
    filename: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> FileResponse:
    """Serve um PNG apenas para o dono da análise que o gerou."""
    if not _SAFE_JOB_ID.fullmatch(job_id):
        raise _NOT_FOUND

    safe_filename = _ALLOWED_FILES.get(filename)
    if safe_filename is None:
        raise _NOT_FOUND

    analysis = get_analysis_for_user(db, job_id, current_user.id)
    if analysis is None:
        raise _NOT_FOUND

    # Usa o id vindo do banco (já validado contra o usuário) em vez do path param.
    safe_job_id = str(analysis.id)
    if not _SAFE_JOB_ID.fullmatch(safe_job_id):
        raise _NOT_FOUND

    base = Path(get_settings().OUTPUTS_DIR).resolve()
    filepath = (base / safe_job_id / safe_filename).resolve()
    if base not in filepath.parents or not filepath.is_file():
        raise _NOT_FOUND

    return FileResponse(filepath, media_type="image/png")
