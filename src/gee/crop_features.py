"""Série temporal de NDVI/EVI para classificação de tipo de cultivo."""

from datetime import date

import ee

from src.gee.collector import get_composite_image, search_images
from src.gee.preprocessor import preprocess_image
from src.processing.indices import calculate_evi_ee, calculate_ndvi_ee


def month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Divide um intervalo em janelas mensais (YYYY-MM-DD)."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    ranges = []
    year, month = start.year, start.month
    while date(year, month, 1) <= end:
        next_year = year + (1 if month == 12 else 0)
        next_month = 1 if month == 12 else month + 1
        window_start = max(date(year, month, 1), start)
        window_end = min(date(next_year, next_month, 1), end)
        if window_start < window_end:
            ranges.append((window_start.isoformat(), window_end.isoformat()))
        year, month = next_year, next_month
    return ranges


def crop_feature_names(num_months: int) -> list[str]:
    """Nomes das features na ordem NDVI/EVI por mês."""
    names = []
    for i in range(num_months):
        names.append(f"NDVI_{i:02d}")
        names.append(f"EVI_{i:02d}")
    return names


def get_crop_feature_image(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    satellite: str = "sentinel2",
    max_cloud_pct: int = 80,
) -> ee.Image:
    """Monta imagem multi-banda com NDVI/EVI mensal (série temporal).

    Meses sem imagens válidas são preenchidos com banda constante zero,
    mantendo o número fixo de features (2 por mês).
    """
    windows = month_ranges(start_date, end_date)[:12]

    bands = []
    for index in range(12):
        if index >= len(windows):
            bands.append(ee.Image.constant(0).rename(f"NDVI_{index:02d}"))
            bands.append(ee.Image.constant(0).rename(f"EVI_{index:02d}"))
            continue

        window_start, window_end = windows[index]
        collection = search_images(
            geometry, window_start, window_end, satellite, max_cloud_pct
        )
        if collection.size().getInfo() == 0:
            bands.append(ee.Image.constant(0).rename(f"NDVI_{index:02d}"))
            bands.append(ee.Image.constant(0).rename(f"EVI_{index:02d}"))
            continue

        composite = get_composite_image(collection, geometry)
        composite = preprocess_image(composite, satellite)

        ndvi = calculate_ndvi_ee(composite, satellite)
        ndvi = ndvi.select("NDVI").rename(f"NDVI_{index:02d}")
        evi = calculate_evi_ee(composite, satellite)
        evi = evi.select("EVI").rename(f"EVI_{index:02d}")

        bands.extend([ndvi, evi])

    return ee.Image.cat(bands)
