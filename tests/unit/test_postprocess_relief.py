"""Testes de pós-processamento (maioria) e mapa de relevo."""

import numpy as np

from src.geo.postprocess import majority_filter
from src.geo.relief import elevation_legend, make_relief_rgb


def test_majority_filter_removes_isolated_pixel():
    class_map = np.zeros((5, 5), dtype=np.uint8)
    class_map[2, 2] = 1
    smoothed = majority_filter(class_map, num_classes=2, kernel_size=3)
    assert smoothed[2, 2] == 0


def test_majority_filter_preserves_outside_mask():
    class_map = np.zeros((5, 5), dtype=np.uint8)
    class_map[2, 2] = 1
    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = False
    class_map[0, 0] = 3
    smoothed = majority_filter(class_map, mask, num_classes=4, kernel_size=3)
    assert smoothed[0, 0] == 3


def test_relief_rgb_shape_and_dtype():
    elevation = np.array([[100, 200], [300, 400]], dtype=np.float32)
    hillshade = np.array([[0, 128], [200, 255]], dtype=np.float32)
    rgb = make_relief_rgb(elevation, hillshade)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8


def test_elevation_legend_size():
    legend = elevation_legend(100, 600, n_bins=5)
    assert len(legend) == 5
    assert all(item["color"].startswith("#") for item in legend)
