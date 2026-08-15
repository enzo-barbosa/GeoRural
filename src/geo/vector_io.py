"""Manipulação de dados vetoriais."""

import ee
import geopandas as gpd
from shapely.geometry import mapping, shape


def geojson_to_geometry(geojson: dict):
    """Converte GeoJSON (do Streamlit) para geometria Shapely.

    Args:
        geojson: Dicionário GeoJSON

    Returns:
        Objeto Shapely Geometry
    """
    return shape(geojson)


def geometry_to_ee(geometry) -> ee.Geometry:
    """Converte geometria Shapely para formato GEE.

    Args:
        geometry: Objeto Shapely Geometry

    Returns:
        Objeto ee.Geometry
    """
    geojson = mapping(geometry)
    return ee.Geometry(geojson)


def save_shapefile(
    geometry, filepath: str, crs: str = "EPSG:4326", class_name: str = "unclassified"
):
    """Salva uma geometria como shapefile.

    Args:
        geometry: Geometria Shapely
        filepath: Caminho de saída (.shp)
        crs: Sistema de coordenadas
        class_name: Nome da classe
    """
    gdf = gpd.GeoDataFrame({"class": [class_name]}, geometry=[geometry], crs=crs)
    gdf.to_file(filepath, driver="ESRI Shapefile")
