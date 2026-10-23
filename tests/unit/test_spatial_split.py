"""Testes para divisão espacial de amostras e patches."""

import numpy as np

from src.classifiers.unet_cnn import spatial_patch_split
from src.evaluation.spatial_split import (
    assign_spatial_blocks,
    spatial_holdout_indices,
    spatial_kfold_indices,
)


def test_assign_spatial_blocks_groups_nearby_points():
    lon = np.array([-50.001, -50.002, -50.05])
    lat = np.array([-20.001, -20.002, -20.05])
    blocks = assign_spatial_blocks(lon, lat, block_size=0.01)
    assert blocks[0] == blocks[1]
    assert blocks[0] != blocks[2]


def test_spatial_kfold_keeps_groups_disjoint():
    groups = np.repeat(np.arange(10), 5)
    splits = spatial_kfold_indices(groups, n_splits=5)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_spatial_holdout_keeps_groups_disjoint():
    groups = np.repeat(np.arange(8), 4)
    train_idx, test_idx = spatial_holdout_indices(groups, test_size=0.3)
    assert len(train_idx) + len(test_idx) == len(groups)
    assert not set(groups[train_idx]) & set(groups[test_idx])


def test_spatial_patch_split_shapes():
    rng = np.random.default_rng(0)
    image = rng.random((128, 128, 3)).astype(np.float32)
    labels = np.zeros((128, 128), dtype=np.int64)

    X_train, y_train, X_val, y_val = spatial_patch_split(
        image, labels, patch_size=32, stride=16, tiles=2, val_tiles=1
    )

    assert X_train.ndim == 4 and X_train.shape[1:] == (32, 32, 3)
    assert y_train.shape[1:] == (32, 32)
    assert y_val.shape[1:] == (32, 32)
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_train) + len(X_val) == len(range(0, 128 - 32 + 1, 16)) ** 2
