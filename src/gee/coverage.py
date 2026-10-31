"""Detecção de cobertura (Brasil vs global) para roteamento de modelos."""

import ee

BRAZIL_NAME = "Brazil"


def get_coverage(polygon_coords: list[list[float]]) -> str:
    """Retorna 'brazil' ou 'global' para um polígono lon/lat.

    Usa a fronteira oficial (USDOS/LSIB_SIMPLE/2017). Se mais de 50% da área
    estiver no Brasil, usa o especialista brasileiro; caso contrário, a base
    global.
    """
    geometry = ee.Geometry.Polygon(polygon_coords)
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

    matches = countries.filterBounds(geometry)
    names = matches.aggregate_array("country_na").getInfo()

    if not names:
        return "global"
    if len(names) == 1:
        return "brazil" if names[0] == BRAZIL_NAME else "global"

    brazil_fc = countries.filter(ee.Filter.eq("country_na", BRAZIL_NAME))
    if brazil_fc.size().getInfo() == 0:
        return "global"

    brazil_geometry = brazil_fc.geometry()
    intersection = geometry.intersection(brazil_geometry, 1)
    fraction = intersection.area(1).divide(geometry.area(1)).getInfo()
    return "brazil" if (fraction or 0.0) >= 0.5 else "global"


def get_label_source(coverage: str) -> str:
    """Rótulo de referência esperado para a cobertura."""
    return "MapBiomas" if coverage == "brazil" else "ESA WorldCover"
