"""Health check API router."""

from fastapi import APIRouter, Depends

from backend.app.dependencies.models import get_gee
from backend.app.schemas.classify import HealthResponse, LivenessResponse
from backend.app.services.health import check_health, check_liveness

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check(_: None = Depends(get_gee)):
    """Health check completo (inclui inicialização do GEE)."""
    result = await check_health()
    return HealthResponse(**result)


@router.get("/live", response_model=LivenessResponse)
async def liveness_check():
    """Liveness check usado por Docker/CI (não chama o GEE)."""
    result = await check_liveness()
    return LivenessResponse(**result)
