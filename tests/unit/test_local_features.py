"""Testes dos índices locais."""

import numpy as np
import pandas as pd

from src.processing.local_features import (
    LOCAL_INDEX_NAMES,
    add_local_indices,
    add_local_indices_array,
)


def test_add_local_indices_dataframe():
    df = pd.DataFrame(
        {
            "B2": [0.05],
            "B3": [0.08],
            "B4": [0.10],
            "B8": [0.40],
            "B11": [0.20],
            "B12": [0.15],
        }
    )
    result = add_local_indices(df)
    for name in LOCAL_INDEX_NAMES:
        assert name in result.columns
    assert result.loc[0, "NDWI"] < 0  # vegetação: NIR > verde
    assert result.loc[0, "NBR"] > 0


def test_add_local_indices_array_shape():
    image = np.ones((4, 4, 9), dtype=np.float32)
    result = add_local_indices_array(image)
    assert result.shape == (4, 4, 9 + len(LOCAL_INDEX_NAMES))


def test_add_local_indices_without_bands_is_noop():
    df = pd.DataFrame({"x": [1, 2]})
    result = add_local_indices(df)
    assert list(result.columns) == ["x"]
