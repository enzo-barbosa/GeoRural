"""Geocodificação de endereços via OpenStreetMap Nominatim.

O Nominatim é gratuito e não exige chave de API, mas impõe regras de uso:
- no máximo ~1 requisição por segundo;
- User-Agent identificável;
- cache de consultas repetidas.

Por isso o acesso é feito pelo backend, com ``User-Agent`` próprio,
throttle entre chamadas e cache em memória.

Política: https://operations.osmfoundation.org/policies/nominatim/
"""

import os
import threading
import time
from math import asin, cos, radians, sin, sqrt

import httpx

from backend.app.schemas.geocode import GeocodeResult

NOMINATIM_URL = os.getenv(
    "NOMINATIM_URL", "https://nominatim.openstreetmap.org/search"
)
USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "GeoRural/0.1 (projeto academico de classificacao de solo)",
)
TIMEOUT_S = float(os.getenv("NOMINATIM_TIMEOUT_S", "10.0"))
_MIN_INTERVAL_S = float(os.getenv("NOMINATIM_MIN_INTERVAL_S", "1.1"))
_MAX_RESULTS = 10
_CACHE_MAX = 256

_lock = threading.Lock()
_last_request = 0.0
_cache: dict[tuple[str, int], list[GeocodeResult]] = {}


class GeocodingError(RuntimeError):
    """Falha ao consultar o serviço de geocodificação."""


def clear_cache() -> None:
    """Limpa o cache (útil em testes)."""
    with _lock:
        _cache.clear()


def _throttle() -> None:
    """Respeita o intervalo mínimo entre chamadas ao Nominatim."""
    global _last_request
    with _lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL_S - (now - _last_request)
        _last_request = now + max(wait, 0.0)
    if wait > 0:
        time.sleep(wait)


def _normalize(query: str) -> str:
    return " ".join(query.split()).strip()


def _fetch(query: str, limit: int) -> list[dict]:
    _throttle()
    try:
        response = httpx.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": limit,
                "addressdetails": 0,
                "accept-language": "pt-BR",
            },
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as error:
        raise GeocodingError(
            "Serviço de endereços indisponível no momento."
        ) from error

    if not isinstance(payload, list):
        raise GeocodingError("Resposta inesperada do serviço de endereços.")
    return payload


def _parse(item: dict) -> GeocodeResult | None:
    try:
        lat = float(item["lat"])
        lon = float(item["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    raw_bbox = item.get("boundingbox")
    bbox = None
    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        try:
            bbox = [float(value) for value in raw_bbox]
        except (TypeError, ValueError):
            bbox = None

    return GeocodeResult(
        display_name=str(item.get("display_name") or f"{lat}, {lon}"),
        lat=lat,
        lon=lon,
        bbox=bbox,
    )


def _distance_km(a: GeocodeResult, b: GeocodeResult) -> float:
    """Distância aproximada (haversine) entre dois resultados."""
    lat1, lon1, lat2, lon2 = map(radians, (a.lat, a.lon, b.lat, b.lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(min(1.0, sqrt(h)))


def _short_name(result: GeocodeResult) -> str:
    return result.display_name.split(",")[0].strip().casefold()


def _dedupe(results: list[GeocodeResult]) -> list[GeocodeResult]:
    """Remove duplicatas do Nominatim (o mesmo local retornado duas vezes).

    Mantém o primeiro resultado (mais relevante) e descarta outro com o
    mesmo nome curto a menos de 10 km. Homônimos distantes são preservados.
    """
    kept: list[GeocodeResult] = []
    for result in results:
        name = _short_name(result)
        duplicate = any(
            _short_name(other) == name and _distance_km(result, other) < 10.0
            for other in kept
        )
        if not duplicate:
            kept.append(result)
    return kept


def search_places(query: str, limit: int = 5) -> list[GeocodeResult]:
    """Converte um endereço/texto em coordenadas.

    Consultas com menos de 3 caracteres não chamam o serviço externo.
    Resultados repetidos vêm do cache em memória.
    """
    normalized = _normalize(query)
    if len(normalized) < 3:
        return []

    limit = max(1, min(int(limit), _MAX_RESULTS))
    cache_key = (normalized.lower(), limit)

    with _lock:
        cached = _cache.get(cache_key)
    if cached is not None:
        return list(cached)

    payload = _fetch(normalized, min(limit * 2, _MAX_RESULTS))
    results = []
    for item in payload:
        parsed = _parse(item)
        if parsed is not None:
            results.append(parsed)

    results = _dedupe(results)[:limit]

    with _lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[cache_key] = list(results)

    return results
