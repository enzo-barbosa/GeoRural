"""Análise topográfica com DEM (NASADEM/GLO-30) via Google Earth Engine."""

import os

import ee
import numpy as np

# Classes de declividade (EMBRAPA), ids 0..5
SLOPE_CLASS_NAMES = [
    "Plano",
    "Suave ondulado",
    "Ondulado",
    "Forte ondulado",
    "Montanhoso",
    "Escarpado",
]

SLOPE_CLASS_RANGES = [
    (0, 3),
    (3, 8),
    (8, 20),
    (20, 45),
    (45, 75),
    (75, 9999),
]

SLOPE_CLASS_COLORS = [
    "#2ecc71",
    "#a3d977",
    "#f4d03f",
    "#f39c12",
    "#e67e22",
    "#e74c3c",
]


def get_dem(geometry: ee.Geometry, source: str | None = None) -> ee.Image:
    """Retorna o DEM recortado na área de interesse.

    ``GEORURAL_DEM_SOURCE=GLO30`` usa Copernicus GLO-30; o padrão é NASADEM.
    """
    source = (source or os.environ.get("GEORURAL_DEM_SOURCE", "NASADEM")).upper()
    if source == "GLO30":
        dem = ee.Image("COPERNICUS/DEM/GLO30").select("DEM")
    else:
        dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    return dem.clip(geometry)


def dem_source_name(source: str | None = None) -> str:
    """Nome legível da fonte do DEM."""
    source = (source or os.environ.get("GEORURAL_DEM_SOURCE", "NASADEM")).upper()
    return "Copernicus GLO-30" if source == "GLO30" else "NASADEM"


def compute_terrain(dem: ee.Image) -> ee.Image:
    """Adiciona declividade, aspecto e hillshade ao DEM.

    Usa ee.Terrain.products, que gera as bandas:
    'elevation', 'slope', 'aspect' e 'hillshade'.
    """
    return ee.Terrain.products(dem)


def classify_slope_values(slope: np.ndarray) -> np.ndarray:
    """Classifica declividade (graus) nas classes EMBRAPA (numpy)."""
    slope = np.asarray(slope, dtype=np.float32)
    classes = np.zeros(slope.shape, dtype=np.uint8)
    for idx, (low, _) in enumerate(SLOPE_CLASS_RANGES[1:], start=1):
        classes[slope >= low] = idx
    return classes


def classify_slope(slope: ee.Image) -> ee.Image:
    """Classifica a declividade no GEE nas classes EMBRAPA (0..5)."""
    cls = ee.Image(0)
    for idx, (low, _) in enumerate(SLOPE_CLASS_RANGES[1:], start=1):
        cls = cls.where(slope.gte(low), idx)
    return cls.rename("slope_class").toByte()


def elevation_stats(terrain: ee.Image, geometry: ee.Geometry, scale: int = 30) -> dict:
    """Estatísticas de elevação e declividade via reduceRegion."""
    reducer = ee.Reducer.minMax().combine(
        ee.Reducer.mean(), sharedInputs=True
    )
    values = terrain.select(["elevation", "slope"]).reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        maxPixels=1_000_000_000,
        bestEffort=True,
    ).getInfo()
    return values or {}


def summarize(
    stats: dict, slope_metrics: dict
) -> dict:
    """Monta o resumo topográfico a partir de stats e áreas por classe."""
    def _num(key):
        value = stats.get(key)
        return float(value) if value is not None else None

    elevation_min = _num("elevation_min")
    elevation_max = _num("elevation_max")
    relief = None
    if elevation_min is not None and elevation_max is not None:
        relief = elevation_max - elevation_min

    classes = []
    dominant = None
    for item in slope_metrics["classes"]:
        cid = item["class_id"]
        if cid >= len(SLOPE_CLASS_NAMES):
            continue
        classes.append(
            {
                "class_name": SLOPE_CLASS_NAMES[cid],
                "area_ha": round(item["area_ha"], 4),
                "percent": round(item["percent"], 2),
            }
        )

    if classes:
        dominant = max(classes, key=lambda c: c["area_ha"])["class_name"]

    return {
        "elevation_min": elevation_min,
        "elevation_max": elevation_max,
        "elevation_mean": _num("elevation_mean"),
        "relief_m": relief,
        "slope_min": _num("slope_min"),
        "slope_mean": _num("slope_mean"),
        "slope_max": _num("slope_max"),
        "dominant_class": dominant,
        "classes": classes,
    }
