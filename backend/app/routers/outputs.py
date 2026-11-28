"""Entrega de imagens geradas pela classificação, protegidas por usuário."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.db.models import User
from backend.app.db.repositories import get_analysis_for_user
from backend.app.db.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.services.output_files import resolve_output_file

router = APIRouter(tags=["outputs"])

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
    analysis = get_analysis_for_user(db, job_id, current_user.id)
    if analysis is None:
        raise _NOT_FOUND

    # Usa o id vindo do banco (já validado contra o usuário) em vez do path param.
    filepath = resolve_output_file(str(analysis.id), filename)
    if filepath is None:
        raise _NOT_FOUND

    return FileResponse(filepath, media_type="image/png")
