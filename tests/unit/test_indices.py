"""Testes para índices espectrais."""

import numpy as np

from src.processing.indices import calculate_evi, calculate_ndvi


def test_ndvi_positive():
    nir = np.array([0.5, 0.6, 0.7])
    red = np.array([0.1, 0.1, 0.1])
    ndvi = calculate_ndvi(nir, red)
    assert all(ndvi > 0)


def test_ndvi_range():
    nir = np.random.uniform(0, 1, (100, 100))
    red = np.random.uniform(0, 1, (100, 100))
    ndvi = calculate_ndvi(nir, red)
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1)


def test_ndvi_water():
    nir = np.array([0.01])
    red = np.array([0.02])
    ndvi = calculate_ndvi(nir, red)
    assert ndvi[0] < 0


def test_evi_positive():
    nir = np.array([0.5, 0.6, 0.7])
    red = np.array([0.1, 0.1, 0.1])
    blue = np.array([0.05, 0.05, 0.05])
    evi = calculate_evi(nir, red, blue)
    assert all(evi > 0)


def test_evi_range():
    nir = np.random.uniform(0, 1, (100, 100))
    red = np.random.uniform(0, 1, (100, 100))
    blue = np.random.uniform(0, 1, (100, 100))
    evi = calculate_evi(nir, red, blue)
    assert np.all(np.isfinite(evi))
