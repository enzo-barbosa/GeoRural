"""Testes da validação e normalização de polígonos."""

import pytest

from backend.app.services.geometry_validation import (
    GeometryValidationError,
    validate_polygon,
)

VALID = {
    "type": "Polygon",
    "coordinates": [
        [
            [-50.30, -20.25],
            [-50.20, -20.25],
            [-50.20, -20.30],
            [-50.30, -20.30],
            [-50.30, -20.25],
        ]
    ],
}


def test_valid_polygon_is_measured():
    result = validate_polygon(VALID)

    assert result.area_ha > 0
    assert result.perimeter_m > 0
    assert result.geojson["type"] == "Polygon"
    ring = result.geojson["coordinates"][0]
    assert ring[0] == ring[-1]


def test_unclosed_ring_is_closed():
    open_polygon = {
        "type": "Polygon",
        "coordinates": [VALID["coordinates"][0][:-1]],
    }

    result = validate_polygon(open_polygon)

    ring = result.geojson["coordinates"][0]
    assert len(ring) == len(VALID["coordinates"][0])
    assert ring[0] == ring[-1]


def test_consecutive_duplicate_points_are_removed():
    duplicated = {
        "type": "Polygon",
        "coordinates": [
            [
                [-50.30, -20.25],
                [-50.30, -20.25],
                [-50.20, -20.25],
                [-50.20, -20.30],
                [-50.30, -20.30],
                [-50.30, -20.25],
            ]
        ],
    }

    result = validate_polygon(duplicated)

    assert len(result.geojson["coordinates"][0]) == 5


def test_rejects_too_few_points():
    with pytest.raises(GeometryValidationError, match="pelo menos 3 pontos"):
        validate_polygon(
            {
                "type": "Polygon",
                "coordinates": [[[-50.0, -20.0], [-49.0, -19.0]]],
            }
        )


def test_rejects_tiny_area():
    tiny = {
        "type": "Polygon",
        "coordinates": [
            [
                [-50.0, -20.0],
                [-49.99999, -20.0],
                [-49.99999, -19.99999],
                [-50.0, -19.99999],
                [-50.0, -20.0],
            ]
        ],
    }

    with pytest.raises(GeometryValidationError, match="Área muito pequena"):
        validate_polygon(tiny)


def test_rejects_collinear_ring():
    with pytest.raises(GeometryValidationError, match="Polígono inválido"):
        validate_polygon(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-50.0, -20.0],
                        [-49.5, -20.0],
                        [-49.0, -20.0],
                        [-50.0, -20.0],
                    ]
                ],
            }
        )


def test_rejects_out_of_range_coordinate():
    with pytest.raises(GeometryValidationError, match="Latitude fora"):
        validate_polygon(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-50.0, -95.0],
                        [-49.0, -20.0],
                        [-50.0, -20.0],
                        [-50.0, -95.0],
                    ]
                ],
            }
        )


def test_rejects_self_intersection():
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [
                [-50.0, -20.0],
                [-49.9, -19.9],
                [-49.9, -20.0],
                [-50.0, -19.9],
                [-50.0, -20.0],
            ]
        ],
    }

    with pytest.raises(GeometryValidationError, match="auto-interseção"):
        validate_polygon(bowtie)


def test_rejects_non_polygon():
    with pytest.raises(GeometryValidationError, match="GeoJSON Polygon"):
        validate_polygon({"type": "Point", "coordinates": [-50.0, -20.0]})


def test_rejects_non_numeric_coordinate():
    with pytest.raises(GeometryValidationError, match="números"):
        validate_polygon(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        ["a", -20.0],
                        [-49.0, -20.0],
                        [-50.0, -21.0],
                        ["a", -20.0],
                    ]
                ],
            }
        )
