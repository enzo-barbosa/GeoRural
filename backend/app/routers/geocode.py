"""API de geocodificação (proxy do OpenStreetMap Nominatim)."""

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.geocode import GeocodeResponse
from backend.app.services.geocoding import GeocodingError, search_places

router = APIRouter(prefix="/api/geocode", tags=["geocoding"])


@router.get("", response_model=GeocodeResponse)
def geocode_endpoint(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(5, ge=1, le=10),
):
    """Converte um endereço ou local em coordenadas (lat/lon)."""
    try:
        results = search_places(q, limit=limit)
    except GeocodingError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return GeocodeResponse(results=results)
