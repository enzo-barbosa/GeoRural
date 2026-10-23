"""Testes para métricas geométricas (área, perímetro, talhões)."""

import numpy as np
from rasterio.transform import from_origin

from src.geo.measurements import (
    compute_raster_metrics,
    polygon_mask,
    polygon_metrics,
    raster_bounds_lonlat,
)


def _transform():
    return from_origin(0, 0, 10, 10)


def test_area_projected_pixels():
    pred_map = np.zeros((10, 10), dtype=np.uint8)
    metrics = compute_raster_metrics(pred_map, _transform(), "EPSG:32622")
    assert metrics["total_area_ha"] == 1.0
    assert metrics["total_pixels"] == 100
    assert metrics["classes"][0]["area_ha"] == 1.0
    assert metrics["classes"][0]["parcels"] == 1


def test_parcels_count():
    pred_map = np.zeros((10, 10), dtype=np.uint8)
    pred_map[0:2, 0:2] = 1
    pred_map[7:9, 7:9] = 1
    metrics = compute_raster_metrics(pred_map, _transform(), "EPSG:32622")
    class_one = next(c for c in metrics["classes"] if c["class_id"] == 1)
    assert class_one["parcels"] == 2


def test_valid_mask_limits_area():
    pred_map = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :5] = True
    metrics = compute_raster_metrics(
        pred_map, _transform(), "EPSG:32622", valid_mask=mask
    )
    assert metrics["total_area_ha"] == 0.5


def test_polygon_metrics_square():
    coords = [[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]]
    metrics = polygon_metrics(coords)
    assert 115 < metrics["area_ha"] < 130
    assert 4300 < metrics["perimeter_m"] < 4600


def test_polygon_mask_geographic():
    transform = from_origin(-50, -20, 0.001, 0.001)
    coords = [[-50, -20], [-49.99, -20], [-49.99, -20.01], [-50, -20.01]]
    mask = polygon_mask(coords, transform, "EPSG:4326", (10, 10))
    assert mask.all()


def test_raster_bounds_lonlat():
    transform = from_origin(-50, -20, 0.001, 0.001)
    bounds = raster_bounds_lonlat(transform, "EPSG:4326", (10, 10))
    south_west, north_east = bounds
    assert south_west[0] == -20.01
    assert south_west[1] == -50
    assert north_east[0] == -20
    assert north_east[1] == -49.99
