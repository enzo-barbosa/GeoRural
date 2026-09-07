"""Health check API router."""

from fastapi import APIRouter, Depends

from backend.app.schemas.classify import HealthResponse
from backend.app.services.health import check_health
from backend.app.dependencies.models import get_gee

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check(_: None = Depends(get_gee)):
    """Health check endpoint."""
    result = await check_health()
    return HealthResponse(**result)