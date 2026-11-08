"""Índices espectrais calculados localmente a partir das bandas já extraídas.

Não fazem novas chamadas ao GEE: reutilizam B2/B3/B4/B8/B11/B12.
"""

import numpy as np
import pandas as pd

LOCAL_INDEX_NAMES = [
    "NDWI",
    "MNDWI",
    "SAVI",
    "EVI",
    "NBR",
    "brightness",
]

BASE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def _safe_divide(numerator, denominator):
    denominator = np.where(np.abs(denominator) < 1e-6, np.nan, denominator)
    result = numerator / denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _indices_from_arrays(blue, green, red, nir, swir1, swir2):
    ndwi = _safe_divide(green - nir, green + nir)
    mndwi = _safe_divide(green - swir1, green + swir1)
    savi = _safe_divide(1.5 * (nir - red), nir + red + 0.5)
    evi = _safe_divide(
        2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0
    )
    nbr = _safe_divide(nir - swir2, nir + swir2)
    brightness = np.mean([blue, green, red, nir, swir1, swir2], axis=0)
    return {
        "NDWI": ndwi,
        "MNDWI": mndwi,
        "SAVI": savi,
        "EVI": evi,
        "NBR": nbr,
        "brightness": brightness,
    }


def add_local_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona os índices locais a um DataFrame de amostras."""
    if not set(BASE_BANDS).issubset(df.columns):
        return df

    values = _indices_from_arrays(
        df["B2"].to_numpy(dtype=np.float32),
        df["B3"].to_numpy(dtype=np.float32),
        df["B4"].to_numpy(dtype=np.float32),
        df["B8"].to_numpy(dtype=np.float32),
        df["B11"].to_numpy(dtype=np.float32),
        df["B12"].to_numpy(dtype=np.float32),
    )
    result = df.copy()
    for name, array in values.items():
        result[name] = array.astype(np.float32)
    return result


def add_local_indices_array(image: np.ndarray) -> np.ndarray:
    """Adiciona índices a um array (H, W, C) cujas 6 primeiras bandas são S2."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] < 6:
        return image

    values = _indices_from_arrays(
        image[..., 0],
        image[..., 1],
        image[..., 2],
        image[..., 3],
        image[..., 4],
        image[..., 5],
    )
    indices = np.stack(
        [values[name] for name in LOCAL_INDEX_NAMES], axis=-1
    ).astype(np.float32)
    return np.concatenate([image, indices], axis=-1)
