"""Schemas da API de geocodificação."""

from pydantic import BaseModel, Field


class GeocodeResult(BaseModel):
    """Um resultado de geocodificação (Nominatim)."""

    display_name: str
    lat: float
    lon: float
    # Nominatim devolve [south, north, west, east].
    bbox: list[float] | None = None


class GeocodeResponse(BaseModel):
    """Lista de resultados para uma consulta de endereço."""

    results: list[GeocodeResult] = Field(default_factory=list)
