"""Métricas geométricas reais (área e perímetro) para propriedades rurais."""

import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon, mapping


def _utm_crs(lon: float, lat: float) -> CRS:
    """Retorna o CRS UTM (metros) adequado para uma coordenada."""
    zone = int((lon + 180) / 6) + 1
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": lat < 0})


def _pixel_area_by_row(transform, crs, height: int) -> np.ndarray:
    """Área (m²) de cada pixel por linha do raster.

    Em CRS geográfico a largura de um grau de longitude varia com a
    latitude, então a área do pixel é calculada linha a linha.
    """
    crs = CRS.from_user_input(crs)
    dx = abs(transform.a)
    dy = abs(transform.e)

    if crs.is_geographic:
        rows = np.arange(height)
        lats = transform.f + (rows + 0.5) * transform.e
        phi = np.deg2rad(lats)
        mlat = (
            111132.92
            - 559.82 * np.cos(2 * phi)
            + 1.175 * np.cos(4 * phi)
            - 0.0023 * np.cos(6 * phi)
        )
        mlon = (
            111412.84 * np.cos(phi)
            - 93.5 * np.cos(3 * phi)
            + 0.118 * np.cos(5 * phi)
        )
        return (dx * mlon) * (dy * mlat)

    factor = 1.0
    try:
        factor = crs.axis_info[0].unit_conversion_factor or 1.0
    except (IndexError, AttributeError):
        factor = 1.0
    return np.full(height, (dx * factor) * (dy * factor), dtype=np.float64)


def _count_parcels(mask: np.ndarray) -> int:
    """Conta talhões (componentes conexos) em uma máscara binária."""
    if not mask.any():
        return 0
    import cv2

    num_labels, _ = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8
    )
    return max(0, num_labels - 1)


def polygon_mask(polygon_coords: list, transform, crs, shape: tuple) -> np.ndarray:
    """Máscara booleana (True dentro do polígono) alinhada ao raster."""
    height, width = shape[:2]
    crs_obj = CRS.from_user_input(crs)
    coords = list(polygon_coords)

    if not crs_obj.is_geographic:
        transformer = Transformer.from_crs(
            "EPSG:4326", crs_obj, always_xy=True
        )
        xs, ys = transformer.transform(
            [c[0] for c in coords], [c[1] for c in coords]
        )
        coords = list(zip(xs, ys))

    geom = Polygon(coords)
    if not geom.is_valid:
        geom = geom.buffer(0)

    return geometry_mask(
        [mapping(geom)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )


def compute_raster_metrics(
    pred_map: np.ndarray, transform, crs, valid_mask: np.ndarray | None = None
) -> dict:
    """Calcula área real (ha) e talhões por classe.

    Args:
        pred_map: Mapa de classes (height, width) com ids 0..N.
        transform: Affine do raster (rasterio).
        crs: CRS do raster.
        valid_mask: Máscara opcional (True = dentro da área de interesse).

    Returns:
        Dict com área total, total de pixels e métricas por classe.
    """
    pred_map = np.asarray(pred_map)
    height, width = pred_map.shape[:2]
    row_areas = _pixel_area_by_row(transform, crs, height)

    if valid_mask is None:
        valid_mask = np.ones((height, width), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (height, width):
            valid_mask = np.ones((height, width), dtype=bool)

    total_m2 = float((valid_mask.sum(axis=1) * row_areas).sum())

    classes = []
    for cid in np.unique(pred_map):
        if cid < 0:
            continue
        mask = (pred_map == cid) & valid_mask
        area_m2 = float((mask.sum(axis=1) * row_areas).sum())
        classes.append(
            {
                "class_id": int(cid),
                "pixels": int(mask.sum()),
                "area_m2": area_m2,
                "area_ha": area_m2 / 10000.0,
                "percent": (area_m2 / total_m2 * 100.0) if total_m2 else 0.0,
                "parcels": _count_parcels(mask),
            }
        )

    return {
        "height": height,
        "width": width,
        "total_area_ha": total_m2 / 10000.0,
        "total_pixels": int(pred_map.size),
        "classes": classes,
    }


def polygon_metrics(coordinates: list) -> dict:
    """Calcula área (ha) e perímetro (m) de um polígono lon/lat.

    Args:
        coordinates: Lista de pares [lon, lat] (anel externo).

    Returns:
        Dict com area_ha, perimeter_m e bounds.
    """
    if not coordinates:
        return {"area_ha": 0.0, "perimeter_m": 0.0}

    geom = Polygon(coordinates)
    if not geom.is_valid:
        geom = geom.buffer(0)
    centroid = geom.centroid
    utm = _utm_crs(centroid.x, centroid.y)
    transformer = Transformer.from_crs("EPSG:4326", utm, always_xy=True)

    projected = [transformer.transform(x, y) for x, y in coordinates]
    polygon_m = Polygon(projected)
    if not polygon_m.is_valid:
        polygon_m = polygon_m.buffer(0)

    return {
        "area_ha": float(polygon_m.area / 10000.0),
        "perimeter_m": float(polygon_m.length),
    }


def raster_bounds_lonlat(transform, crs, shape: tuple) -> list[list[float]]:
    """Bounds [[south, west], [north, east]] em EPSG:4326 (formato Leaflet)."""
    height, width = shape[:2]
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)

    if CRS.from_user_input(crs).is_geographic:
        west, south, east, north = left, bottom, right, top
    else:
        west, south, east, north = transform_bounds(
            crs, "EPSG:4326", left, bottom, right, top
        )

    return [[float(south), float(west)], [float(north), float(east)]]
