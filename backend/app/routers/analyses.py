"""Rotas de histórico e detalhe das análises do usuário."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from backend.app.db.models import Analysis, User
from backend.app.db.repositories import (
    create_analysis,
    delete_analysis,
    get_analysis_for_user,
    list_analyses,
)
from backend.app.db.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.dependencies.models import get_gee
from backend.app.schemas.analysis import AnalysisRead, AnalysisSummary
from backend.app.schemas.classify import ClassificationPending
from backend.app.services.classification_service import new_job_id
from backend.app.services.job_queue import (
    JobQueue,
    enqueue_job,
    get_job_queue,
)
from backend.app.services.output_files import delete_output_directory
from backend.app.services.report_service import build_analysis_report

logger = logging.getLogger(__name__)

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


@router.get("/{job_id}/report.pdf", response_class=Response)
def download_analysis_report(
    job_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    """Gera e entrega o relatório PDF de uma análise concluída."""
    analysis = get_analysis_for_user(db, job_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Análise não encontrada"
        )
    if analysis.status != "completed" or not analysis.result:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A análise ainda não foi concluída.",
        )

    pdf = build_analysis_report(analysis, user_name=current_user.name)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="georural_{analysis.id}.pdf"'
            )
        },
    )


@router.delete("/{job_id}", status_code=204)
def delete_user_analysis(
    job_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> None:
    """Remove uma análise, seus arquivos e cancela o job se ainda estiver na fila."""
    analysis = get_analysis_for_user(db, job_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Análise não encontrada"
        )

    analysis_id = str(analysis.id)

    if analysis.status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A análise está em processamento e não pode ser excluída.",
        )

    if analysis.status == "queued":
        if not queue.cancel(analysis_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "A análise já começou a ser processada ou não está mais "
                    "na fila."
                ),
            )
        # O job nem chegou a rodar; a linha pode sair antes dos arquivos.
        delete_analysis(db, analysis)
        try:
            delete_output_directory(analysis_id)
        except OSError:
            logger.exception(
                "Falha ao remover arquivos parciais do job %s", job_id
            )
        return

    try:
        deleted_files = delete_output_directory(analysis_id)
    except OSError as exc:
        logger.exception("Falha ao remover arquivos do job %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Não foi possível remover os arquivos da análise.",
        ) from exc

    delete_analysis(db, analysis)
    logger.info(
        "Análise %s removida do histórico (arquivos removidos: %s)",
        job_id,
        deleted_files,
    )


@router.post(
    "/{job_id}/retry",
    response_model=ClassificationPending,
    status_code=202,
)
def retry_user_analysis(
    job_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    _: Annotated[None, Depends(get_gee)],
) -> ClassificationPending:
    """Recria a análise com os mesmos parâmetros e a enfileira novamente."""
    original = get_analysis_for_user(db, job_id, current_user.id)
    if original is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Análise não encontrada"
        )

    if original.status in {"queued", "running"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A análise original ainda está em processamento.",
        )

    if not original.geojson:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Análise antiga sem polígono salvo. Refaça a classificação.",
        )

    new_id = new_job_id()
    name = (
        f"{original.name} (reprocessada)" if original.name else None
    )
    analysis = create_analysis(
        db,
        job_id=new_id,
        satellite=original.satellite,
        classifier=original.classifier,
        user_id=current_user.id,
        area_id=original.area_id,
        name=name,
        geojson=original.geojson,
        start_date=original.start_date,
        end_date=original.end_date,
        max_cloud=original.max_cloud,
        include_topography=original.include_topography,
        include_crop=original.include_crop,
        postprocess=original.postprocess,
        status="queued",
        stage="queued",
        progress=0,
    )

    try:
        enqueue_job(db, analysis, queue)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return ClassificationPending(job_id=new_id, status="pending")
