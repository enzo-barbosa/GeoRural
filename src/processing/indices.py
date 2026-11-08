"""Índices espectrais para classificação do uso do solo."""

import ee
import numpy as np


def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Calcula o NDVI (Normalized Difference Vegetation Index).

    Fórmula: NDVI = (NIR - RED) / (NIR + RED)

    Interpretação:
    - NDVI < 0: Água, nuvens, solo exposto
    - NDVI 0 - 0.2: Solo exposto, vegetação rala
    - NDVI 0.2 - 0.5: Vegetação moderada (pastagem)
    - NDVI 0.5 - 1.0: Vegetação densa (mata nativa, lavoura sadia)

    Args:
        nir_band: Banda do infravermelho próximo
        red_band: Banda do vermelho

    Returns:
        Array com valores NDVI (-1 a 1)
    """
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)

    denominator = nir + red
    denominator[denominator == 0] = np.nan

    ndvi = (nir - red) / denominator

    return np.clip(ndvi, -1.0, 1.0)


def calculate_ndvi_ee(image: ee.Image, satellite: str = "sentinel2") -> ee.Image:
    """Calcula o NDVI diretamente no GEE.

    Args:
        image: Imagem GEE pré-processada
        satellite: 'sentinel2' ou 'landsat8'

    Returns:
        Imagem com banda NDVI adicionada
    """
    if satellite == "sentinel2":
        ndvi = image.expression(
            "(b('B8') - b('B4')) / (b('B8') + b('B4'))"
        ).rename("NDVI").clamp(-1.0, 1.0)
    elif satellite == "landsat8":
        ndvi = image.expression(
            "(b('SR_B5') - b('SR_B4')) / (b('SR_B5') + b('SR_B4'))"
        ).rename("NDVI").clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Satélite não suportado: {satellite}")

    return image.addBands(ndvi)


def calculate_evi(
    nir_band: np.ndarray, red_band: np.ndarray, blue_band: np.ndarray
) -> np.ndarray:
    """Calcula o EVI (Enhanced Vegetation Index).

    Fórmula: EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)

    O EVI é mais sensível a variações de alta densidade vegetacional
    e minimiza influências atmosféricas e de solo.

    Args:
        nir_band: Banda do infravermelho próximo
        red_band: Banda do vermelho
        blue_band: Banda do azul

    Returns:
        Array com valores EVI (-1 a 1)
    """
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)
    blue = blue_band.astype(np.float32)

    denominator = nir + 6 * red - 7.5 * blue + 1
    denominator[denominator == 0] = np.nan

    evi = 2.5 * (nir - red) / denominator

    return np.clip(evi, -1.0, 1.0)


def calculate_evi_ee(image: ee.Image, satellite: str = "sentinel2") -> ee.Image:
    """Calcula o EVI diretamente no GEE.

    Args:
        image: Imagem GEE pré-processada
        satellite: 'sentinel2' ou 'landsat8'

    Returns:
        Imagem com banda EVI adicionada
    """
    if satellite == "sentinel2":
        evi = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": image.select("B8"),
                "RED": image.select("B4"),
                "BLUE": image.select("B2"),
            },
        ).rename("EVI").clamp(-1.0, 1.0)
    elif satellite == "landsat8":
        evi = image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {
                "NIR": image.select("SR_B5"),
                "RED": image.select("SR_B4"),
                "BLUE": image.select("SR_B2"),
            },
        ).rename("EVI").clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Satélite não suportado: {satellite}")

    return image.addBands(evi)
