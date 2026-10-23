"""Divisão espacial de amostras para validação honesta de modelos."""

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def assign_spatial_blocks(
    lon, lat, block_size: float = 0.01
) -> np.ndarray:
    """Atribui um id de bloco espacial a cada amostra (grade lon/lat).

    Args:
        lon: Longitudes das amostras.
        lat: Latitudes das amostras.
        block_size: Tamanho do bloco em graus.

    Returns:
        Array de ids inteiros (um por amostra).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    bx = np.floor(lon / block_size).astype(np.int64)
    by = np.floor(lat / block_size).astype(np.int64)

    stacked = np.stack([bx, by], axis=1)
    _, ids = np.unique(stacked, axis=0, return_inverse=True)
    return ids


def spatial_holdout_indices(
    groups: np.ndarray, test_size: float = 0.3, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Gera índices de treino/teste mantendo blocos inteiros em cada lado."""
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    dummy = np.zeros((len(groups), 1))
    return next(splitter.split(dummy, groups=groups))


def spatial_kfold_indices(
    groups: np.ndarray, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Gera folds de validação cruzada com blocos espaciais disjuntos."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    effective = max(2, min(n_splits, len(unique)))

    splitter = GroupKFold(n_splits=effective)
    dummy = np.zeros((len(groups), 1))
    return list(splitter.split(dummy, groups=groups))
