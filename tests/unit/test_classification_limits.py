"""Testes dos limites de área/download da classificação."""

import pytest

from backend.app.services.limits import (
    area_limit_message,
    bands_per_download,
    estimated_pixels,
)


def test_estimated_pixels_sentinel():
    # 100 ha = 1 km²; a 10 m -> 10.000 pixels.
    assert estimated_pixels(100, 10) == 10_000


def test_area_limit_allows_small_area():
    assert area_limit_message(5_000, 10, 6_000_000) is None


def test_area_limit_rejects_large_area():
    message = area_limit_message(200_000, 10, 6_000_000)

    assert message is not None
    assert "excede o limite" in message


def test_bands_per_download_uses_groups_for_small_area():
    # 1M pixels -> 6 MB por banda; limite de 40 MB -> 6 bandas.
    assert bands_per_download(1_000_000, 40_000_000) == 6


def test_bands_per_download_falls_back_to_one_band():
    # 5M pixels -> 30 MB por banda; limite de 40 MB -> 1 banda.
    assert bands_per_download(5_000_000, 40_000_000) == 1


def test_bands_per_download_rejects_huge_area():
    with pytest.raises(RuntimeError):
        bands_per_download(20_000_000, 40_000_000)
