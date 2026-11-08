"""Testes do dataset de patches sob demanda."""

import numpy as np
import torch

from src.classifiers.unet_cnn import make_patch_datasets
from src.config import NUM_CLASSES


def _tiles():
    image = np.ones((128, 128, 9), dtype=np.float32)
    labels = np.zeros((128, 128), dtype=np.int64)
    labels[64:, :] = 1
    return {
        "train": (image, labels),
        "val": (image.copy(), labels.copy()),
    }


def test_patch_dataset_shapes_and_indices():
    train, val = make_patch_datasets(
        _tiles(), "val", patch_size=64, stride=64, include_indices=True
    )
    assert len(train) == 4
    assert len(val) == 4
    image, label = train[0]
    assert image.shape == (15, 64, 64)
    assert label.shape == (64, 64)
    assert isinstance(image, torch.Tensor)


def test_patch_dataset_class_weights_length():
    train, _ = make_patch_datasets(
        _tiles(), "val", patch_size=32, stride=32, include_indices=False
    )
    weights = train.class_weights()
    assert weights.shape == (NUM_CLASSES,)
    assert np.all(weights > 0)
