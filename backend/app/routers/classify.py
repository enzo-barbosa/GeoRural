"""Classification API router."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.models import User
from backend.app.db.repositories import create_analysis, get_area
from backend.app.db.session import get_db
from backend.app.dependencies.auth import get_current_user
from backend.app.dependencies.models import get_gee, get_models_status
from backend.app.schemas.classify import (
    ClassificationError,
    ClassificationResult,
    ClassifyRequest,
    ClassifyResponse,
    CoverageRequest,
    CoverageResponse,
    ModelsStatusResponse,
)
from backend.app.services.classification_service import classify_area, new_job_id
from src.gee.coverage import get_coverage, get_label_source

router = APIRouter(prefix="/api/classify", tags=["classification"])


@router.post("", response_model=ClassifyResponse)
async def classify_endpoint(
    request: ClassifyRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    _: None = Depends(get_gee),
):
    """
    Classify a polygon area using satellite imagery and ML models.

    This is a synchronous endpoint - for large areas it may take 30-120 seconds.
    Cada execução é registrada no histórico do usuário autenticado.
    """
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
        geojson=request.polygon.model_dump(),
        start_date=str(request.start_date),
        end_date=str(request.end_date),
        max_cloud=request.max_cloud,
        include_topography=request.include_topography,
        include_crop=request.include_crop,
        postprocess=request.postprocess,
        status="running",
        stage="starting",
        progress=0,
    )

    polygon_coords = request.polygon.coordinates[0]

    try:
        result = await classify_area(
            polygon_coords=polygon_coords,
            satellite=request.satellite,
            classifier_name=request.classifier,
            start_date=str(request.start_date),
            end_date=str(request.end_date),
            max_cloud=request.max_cloud,
            include_topography=request.include_topography,
            include_crop=request.include_crop,
            postprocess=request.postprocess,
            job_id=job_id,
        )
    except Exception as exc:
        _mark_failed(db, analysis, f"Falha inesperada: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha inesperada na classificação: {exc}",
        ) from exc

    if result["status"] == "failed":
        _mark_failed(db, analysis, result.get("error", "Falha na classificação"))
        return ClassificationError(job_id=result["job_id"], error=result["error"])

    _mark_completed(db, analysis, result)
    return ClassificationResult(**result)


def _mark_failed(db: Session, analysis, error: str) -> None:
    analysis.status = "failed"
    analysis.stage = "failed"
    analysis.error = error
    db.commit()


def _mark_completed(db: Session, analysis, result: dict) -> None:
    measurements = result.get("measurements") or {}
    analysis.status = "completed"
    analysis.stage = "done"
    analysis.progress = 100
    analysis.coverage = result.get("coverage")
    analysis.label_source = result.get("label_source")
    analysis.num_images = result.get("num_images")
    analysis.confidence_mean = result.get("confidence_mean")
    analysis.models_used = result.get("models_used")
    analysis.weights = result.get("weights")
    analysis.area_ha = measurements.get("total_area_ha")
    analysis.result = result
    db.commit()


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
