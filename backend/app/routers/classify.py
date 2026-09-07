"""Classification API router."""

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from backend.app.schemas.classify import (
    ClassifyRequest,
    ClassifyResponse,
    ClassificationError,
    ClassificationPending,
    ClassificationResult,
    ModelsStatusResponse,
)
from backend.app.services.classification_service import classify_area
from backend.app.dependencies.models import get_gee, get_models_status

router = APIRouter(prefix="/api/classify", tags=["classification"])


@router.post("", response_model=ClassifyResponse)
async def classify_endpoint(
    request: ClassifyRequest,
    _: None = Depends(get_gee),
):
    """
    Classify a polygon area using satellite imagery and ML models.
    
    This is a synchronous endpoint - for large areas it may take 30-120 seconds.
    """
    # Convert polygon to list of coordinates
    polygon_coords = request.polygon.coordinates[0]

    result = await classify_area(
        polygon_coords=polygon_coords,
        satellite=request.satellite,
        classifier_name=request.classifier,
        start_date=str(request.start_date),
        end_date=str(request.end_date),
        max_cloud=request.max_cloud,
    )

    if result["status"] == "failed":
        return ClassificationError(job_id=result["job_id"], error=result["error"])

    return ClassificationResult(**result)


@router.get("/models", response_model=ModelsStatusResponse)
async def models_status(_: None = Depends(get_gee)):
    """Get availability status of all ML models."""
    status = get_models_status()
    return ModelsStatusResponse(**status)


@router.get("/outputs/{job_id}/{filename}")
async def get_output_file(job_id: str, filename: str):
    """Serve generated output files."""
    from backend.app.config import get_settings
    settings = get_settings()
    filepath = Path(settings.OUTPUTS_DIR) / job_id / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(filepath)