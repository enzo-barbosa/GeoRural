"""Testes para classificação de declividade e resumo topográfico."""

import numpy as np

from src.gee.topography import classify_slope_values, summarize


def test_classify_slope_values():
    slope = np.array([1, 3, 8, 20, 45, 75, 100], dtype=np.float32)
    classes = classify_slope_values(slope)
    assert classes.tolist() == [0, 1, 2, 3, 4, 5, 5]


def test_classify_slope_flat():
    slope = np.zeros((5, 5), dtype=np.float32)
    classes = classify_slope_values(slope)
    assert (classes == 0).all()


def test_summarize():
    stats = {
        "elevation_min": 400.0,
        "elevation_max": 520.0,
        "elevation_mean": 460.0,
        "slope_min": 1.0,
        "slope_mean": 6.0,
        "slope_max": 18.0,
    }
    slope_metrics = {
        "classes": [
            {"class_id": 0, "area_ha": 2.0, "percent": 20.0},
            {"class_id": 2, "area_ha": 8.0, "percent": 80.0},
        ]
    }
    summary = summarize(stats, slope_metrics)
    assert summary["relief_m"] == 120.0
    assert summary["dominant_class"] == "Ondulado"
    assert len(summary["classes"]) == 2
