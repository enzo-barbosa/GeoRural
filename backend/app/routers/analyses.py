"""Rotas de histórico e detalhe das análises do usuário."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.db.models import Analysis, User
from backend.app.db.repositories import get_analysis_for_user, list_analyses
from backend.app.db.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.schemas.analysis import AnalysisRead, AnalysisSummary

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


def _summary_fields(analysis: Analysis) -> dict[str, Any]:
    """Campos derivados usados na listagem e no detalhe."""
    result = analysis.result or {}
    return {
        "id": analysis.id,
        "name": analysis.name,
        "status": analysis.status,
        "stage": analysis.stage,
        "progress": analysis.progress,
        "error": analysis.error,
        "satellite": analysis.satellite,
        "classifier": analysis.classifier,
        "coverage": analysis.coverage,
        "start_date": analysis.start_date,
        "end_date": analysis.end_date,
        "area_id": analysis.area_id,
        "area_name": analysis.area.name if analysis.area else None,
        "area_ha": analysis.area_ha,
        "num_images": analysis.num_images,
        "confidence_mean": analysis.confidence_mean,
        "image_url": result.get("classification_image_url"),
        "created_at": analysis.created_at,
        "updated_at": analysis.updated_at,
    }


def _analysis_read(analysis: Analysis) -> AnalysisRead:
    return AnalysisRead(
        **_summary_fields(analysis),
        label_source=analysis.label_source,
        max_cloud=analysis.max_cloud,
        include_topography=analysis.include_topography,
        include_crop=analysis.include_crop,
        postprocess=analysis.postprocess,
        models_used=analysis.models_used,
        weights=analysis.weights,
        geojson=analysis.geojson,
        result=analysis.result,
    )


@router.get("", response_model=list[AnalysisSummary])
def list_user_analyses(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AnalysisSummary]:
    """Histórico de análises do usuário, mais recentes primeiro."""
    analyses = list_analyses(
        db, current_user.id, limit=limit, offset=offset
    )
    return [AnalysisSummary(**_summary_fields(a)) for a in analyses]


@router.get("/{job_id}", response_model=AnalysisRead)
def get_user_analysis(
    job_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> AnalysisRead:
    """Detalhe completo de uma análise do usuário."""
    analysis = get_analysis_for_user(db, job_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Análise não encontrada"
        )
    return _analysis_read(analysis)
