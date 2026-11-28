"""Validação e normalização de polígonos recebidos pela API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from src.geo.measurements import polygon_metrics

MIN_AREA_HA = 0.001


class GeometryValidationError(ValueError):
    """Erro de geometria com mensagem adequada ao usuário."""


@dataclass(frozen=True)
class ValidatedPolygon:
    """Polígono normalizado e suas medidas reais."""

    geojson: dict[str, Any]
    area_ha: float
    perimeter_m: float


def _point(
    value: Any, ring_index: int, point_index: int
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise GeometryValidationError(
            "Coordenada inválida no anel "
            f"{ring_index + 1}, ponto {point_index + 1}."
        )

    lon, lat = value[0], value[1]
    if isinstance(lon, bool) or isinstance(lat, bool):
        raise GeometryValidationError(
            "As coordenadas precisam ser números."
        )
    try:
        lon_f, lat_f = float(lon), float(lat)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(
            "As coordenadas precisam ser números."
        ) from exc

    if not math.isfinite(lon_f) or not math.isfinite(lat_f):
        raise GeometryValidationError("As coordenadas precisam ser finitas.")
    if not -180.0 <= lon_f <= 180.0:
        raise GeometryValidationError(
            f"Longitude fora do intervalo permitido: {lon_f}."
        )
    if not -90.0 <= lat_f <= 90.0:
        raise GeometryValidationError(
            f"Latitude fora do intervalo permitido: {lat_f}."
        )
    return lon_f, lat_f


def _normalize_ring(ring: Any, ring_index: int) -> list[list[float]]:
    if not isinstance(ring, list) or len(ring) < 3:
        raise GeometryValidationError(
            f"O anel {ring_index + 1} precisa ter pelo menos 3 pontos."
        )

    points = [
        _point(point, ring_index, point_index)
        for point_index, point in enumerate(ring)
    ]

    # Remove pontos consecutivos repetidos (comuns ao fechar o anel).
    cleaned = [points[0]]
    for point in points[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    if len(cleaned) < 3:
        raise GeometryValidationError(
            f"O anel {ring_index + 1} precisa ter pelo menos 3 pontos "
            "distintos."
        )

    closed = [*cleaned, cleaned[0]]
    return [[lon, lat] for lon, lat in closed]


def validate_polygon(geojson: dict[str, Any]) -> ValidatedPolygon:
    """Valida, fecha o anel externo e calcula área/perímetro do polígono.

    Levanta ``GeometryValidationError`` com uma mensagem em português quando a
    geometria não pode ser usada na classificação.
    """
    if not isinstance(geojson, dict) or geojson.get("type") != "Polygon":
        raise GeometryValidationError("O polígono precisa ser um GeoJSON Polygon.")

    rings = geojson.get("coordinates")
    if not isinstance(rings, list) or not rings:
        raise GeometryValidationError("O polígono precisa ter um anel externo.")

    normalized = [
        _normalize_ring(ring, index) for index, ring in enumerate(rings)
    ]
    geometry = Polygon(normalized[0], normalized[1:])
    if geometry.is_empty:
        raise GeometryValidationError("O polígono está vazio.")

    if not geometry.is_valid:
        reason = explain_validity(geometry)
        if "Self-intersection" in reason:
            reason = "auto-interseção"
        raise GeometryValidationError(f"Polígono inválido: {reason}.")

    try:
        metrics = polygon_metrics(normalized[0])
    except Exception as exc:
        raise GeometryValidationError(
            "Não foi possível calcular a área do polígono."
        ) from exc

    area_ha = float(metrics.get("area_ha") or 0.0)
    perimeter_m = float(metrics.get("perimeter_m") or 0.0)
    if area_ha < MIN_AREA_HA:
        raise GeometryValidationError(
            f"Área muito pequena ({area_ha:.6f} ha). "
            f"O mínimo é {MIN_AREA_HA} ha."
        )

    return ValidatedPolygon(
        geojson={"type": "Polygon", "coordinates": normalized},
        area_ha=area_ha,
        perimeter_m=perimeter_m,
    )
