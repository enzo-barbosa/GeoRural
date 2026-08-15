"""Testes para módulos de processamento."""

import numpy as np
import pytest
from src.processing.indices import calculate_ndvi


def test_ndvi_positive():
    """Teste: NDVI deve ser positivo para vegetação saudável."""
    nir = np.array([0.5, 0.6, 0.7])
    red = np.array([0.1, 0.1, 0.1])
    ndvi = calculate_ndvi(nir, red)
    assert all(ndvi > 0)


def test_ndvi_range():
    """Teste: NDVI deve estar entre -1 e 1."""
    nir = np.random.uniform(0, 1, (100, 100))
    red = np.random.uniform(0, 1, (100, 100))
    ndvi = calculate_ndvi(nir, red)
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1)


def test_ndvi_water():
    """Teste: NDVI deve ser negativo para água."""
    nir = np.array([0.01])
    red = np.array([0.02])
    ndvi = calculate_ndvi(nir, red)
    assert ndvi[0] < 0
