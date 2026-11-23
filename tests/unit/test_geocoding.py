"""Testes da geocodificação (Nominatim)."""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.routers import geocode as geocode_router
from backend.app.schemas.geocode import GeocodeResult
from backend.app.services import geocoding


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _reset_geocoding(monkeypatch):
    geocoding.clear_cache()
    monkeypatch.setattr(geocoding, "_MIN_INTERVAL_S", 0.0)


def test_short_query_skips_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("não deveria chamar o Nominatim")

    monkeypatch.setattr(geocoding.httpx, "get", fail)
    assert geocoding.search_places("ab") == []


def test_search_places_parses_results(monkeypatch):
    payload = [
        {
            "display_name": "Piracicaba, São Paulo, Brasil",
            "lat": "-22.7253",
            "lon": "-47.6492",
            "boundingbox": ["-22.83", "-22.60", "-47.80", "-47.50"],
        },
        {
            "display_name": "Piracicaba, Minas Gerais, Brasil",
            "lat": "-19.9",
            "lon": "-43.1",
            "boundingbox": None,
        },
    ]
    monkeypatch.setattr(
        geocoding.httpx, "get", lambda *a, **k: _FakeResponse(payload)
    )

    results = geocoding.search_places("Piracicaba")

    assert len(results) == 2
    assert results[0].display_name.startswith("Piracicaba, São Paulo")
    assert results[0].lat == pytest.approx(-22.7253)
    assert results[0].lon == pytest.approx(-47.6492)
    assert results[0].bbox == [-22.83, -22.60, -47.80, -47.50]
    assert results[1].bbox is None


def test_search_places_caches_repeated_query(monkeypatch):
    calls = {"count": 0}

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        return _FakeResponse(
            [{"display_name": "Campinas", "lat": "-22.9", "lon": "-47.06"}]
        )

    monkeypatch.setattr(geocoding.httpx, "get", fake_get)

    first = geocoding.search_places("Campinas")
    second = geocoding.search_places("  Campinas  ")

    assert calls["count"] == 1
    assert [r.display_name for r in first] == [r.display_name for r in second]


def test_search_places_ignores_invalid_items(monkeypatch):
    payload = [
        {"display_name": "sem coordenadas"},
        {"display_name": "válido", "lat": "-22.0", "lon": "-47.0"},
    ]
    monkeypatch.setattr(
        geocoding.httpx, "get", lambda *a, **k: _FakeResponse(payload)
    )

    results = geocoding.search_places("teste")

    assert len(results) == 1
    assert results[0].display_name == "válido"


def test_search_places_deduplicates_same_city(monkeypatch):
    payload = [
        {
            "display_name": "Piracicaba, São Paulo, Brasil",
            "lat": "-22.72517",
            "lon": "-47.64933",
        },
        {
            "display_name": "Piracicaba, São Paulo, Brasil",
            "lat": "-22.73981",
            "lon": "-47.66801",
        },
    ]
    monkeypatch.setattr(
        geocoding.httpx, "get", lambda *a, **k: _FakeResponse(payload)
    )

    results = geocoding.search_places("Piracicaba", limit=5)

    assert len(results) == 1
    assert results[0].lat == pytest.approx(-22.72517)


def test_search_places_keeps_homonyms_far_apart(monkeypatch):
    payload = [
        {
            "display_name": "Piracicaba, São Paulo, Brasil",
            "lat": "-22.72",
            "lon": "-47.65",
        },
        {
            "display_name": "Piracicaba, Minas Gerais, Brasil",
            "lat": "-19.90",
            "lon": "-43.10",
        },
    ]
    monkeypatch.setattr(
        geocoding.httpx, "get", lambda *a, **k: _FakeResponse(payload)
    )

    results = geocoding.search_places("Piracicaba", limit=5)

    assert len(results) == 2


def test_search_places_raises_geocoding_error(monkeypatch):
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(geocoding.httpx, "get", fake_get)

    with pytest.raises(geocoding.GeocodingError):
        geocoding.search_places("Piracicaba")


def test_geocode_endpoint_returns_results(monkeypatch):
    monkeypatch.setattr(
        geocode_router,
        "search_places",
        lambda q, limit=5: [
            GeocodeResult(
                display_name="Piracicaba, SP",
                lat=-22.72,
                lon=-47.65,
                bbox=[-22.83, -22.60, -47.80, -47.50],
            )
        ],
    )

    app = FastAPI()
    app.include_router(geocode_router.router)
    client = TestClient(app)

    response = client.get("/api/geocode", params={"q": "Piracicaba"})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["display_name"] == "Piracicaba, SP"
    assert body["results"][0]["lat"] == pytest.approx(-22.72)


def test_geocode_endpoint_maps_service_error(monkeypatch):
    def fail(q, limit=5):
        raise geocoding.GeocodingError("indisponível")

    monkeypatch.setattr(geocode_router, "search_places", fail)

    app = FastAPI()
    app.include_router(geocode_router.router)
    client = TestClient(app)

    response = client.get("/api/geocode", params={"q": "Piracicaba"})

    assert response.status_code == 502
