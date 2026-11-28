"""Classification API router."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.db.models import User
from backend.app.db.repositories import create_analysis, get_area
from backend.app.db.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.dependencies.models import get_gee, get_models_status
from backend.app.schemas.classify import (
    ClassificationPending,
    ClassifyRequest,
    ClassifyResponse,
    CoverageRequest,
    CoverageResponse,
    ModelsStatusResponse,
)
from backend.app.services.classification_service import new_job_id
from backend.app.services.geometry_validation import (
    GeometryValidationError,
    validate_polygon,
)
from backend.app.services.job_queue import (
    JobQueue,
    enqueue_job,
    get_job_queue,
)
from backend.app.services.limits import area_limit_message
from src.gee.coverage import get_coverage, get_label_source

router = APIRouter(prefix="/api/classify", tags=["classification"])


@router.post("", response_model=ClassifyResponse, status_code=202)
async def classify_endpoint(
    request: ClassifyRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    _: Annotated[None, Depends(get_gee)],
) -> ClassificationPending:
    """Enfileira a classificação de um polígono e responde imediatamente.

    O processamento acontece em segundo plano (1 job por vez). Acompanhe por
    ``GET /api/analyses/{job_id}``.
    """
    try:
        validated = validate_polygon(request.polygon.model_dump())
    except GeometryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    scale = 10 if request.satellite == "sentinel2" else 30
    limit_message = area_limit_message(
        validated.area_ha, scale, get_settings().MAX_CLASSIFY_PIXELS
    )
    if limit_message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=limit_message,
        )

    area = None
    if request.area_id is not None:
        area = get_area(db, request.area_id, user_id=current_user.id)
        if area is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Área não encontrada",
            )

    job_id = new_job_id()
    analysis = create_analysis(
        db,
        job_id=job_id,
        satellite=request.satellite,
        classifier=request.classifier,
        user_id=current_user.id,
        area_id=area.id if area else None,
        name=request.name or (area.name if area else None),
        geojson=validated.geojson,
        area_ha=round(validated.area_ha, 4),
        start_date=str(request.start_date),
        end_date=str(request.end_date),
        max_cloud=request.max_cloud,
        include_topography=request.include_topography,
        include_crop=request.include_crop,
        postprocess=request.postprocess,
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

    return ClassificationPending(job_id=job_id, status="pending")


@router.get("/models", response_model=ModelsStatusResponse)
async def models_status(
    _: Annotated[None, Depends(get_gee)],
    __: Annotated[User, Depends(get_current_user)],
):
    """Get availability status of all ML models."""
    return ModelsStatusResponse(**get_models_status())


@router.post("/coverage", response_model=CoverageResponse)
async def coverage_endpoint(
    request: CoverageRequest,
    _: Annotated[None, Depends(get_gee)],
    __: Annotated[User, Depends(get_current_user)],
):
    """Retorna a cobertura (Brasil/global) do polígono."""
    polygon_coords = request.polygon.coordinates[0]
    coverage = get_coverage(polygon_coords)
    return CoverageResponse(
        coverage=coverage, label_source=get_label_source(coverage)
    )
