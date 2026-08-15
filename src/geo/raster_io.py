"""Leitura e escrita de dados raster."""

import numpy as np
import rasterio


def read_geotiff(filepath: str) -> tuple[np.ndarray, dict]:
    """Lê um arquivo GeoTIFF.

    Args:
        filepath: Caminho do arquivo GeoTIFF

    Returns:
        Tupla (array_dados, metadados)
        - array_dados: numpy array de forma (bandas, altura, largura)
        - metadados: dict com transform, crs, shape
    """
    with rasterio.open(filepath) as src:
        data = src.read()
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "shape": data.shape,
            "dtype": data.dtype,
            "bounds": src.bounds,
        }

    return data, meta


def write_geotiff(
    filepath: str,
    data: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: str,
    dtype: str = "float32",
):
    """Escreve um array como GeoTIFF.

    Args:
        filepath: Caminho de saída
        data: numpy array de forma (bandas, altura, largura)
        transform: Transformação geográfica
        crs: Sistema de coordenadas (ex: "EPSG:4326")
        dtype: Tipo de dado de saída
    """
    height, width = data.shape[1], data.shape[2]
    num_bands = data.shape[0]

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(num_bands):
            dst.write(data[i], i + 1)

    print(f"GeoTIFF salvo em: {filepath}")
