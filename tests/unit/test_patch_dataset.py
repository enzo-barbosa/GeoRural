"""Testes do dataset de patches sob demanda."""

import numpy as np
import pytest
import torch

from src.classifiers.unet_cnn import (
    FocalLoss,
    PatchDataset,
    _sample_weights,
    make_patch_datasets,
)
from src.config import IGNORE_LABEL, NUM_CLASSES


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


def test_class_weights_exponent_normalizes_mean():
    train, _ = make_patch_datasets(
        _tiles(), "val", patch_size=32, stride=32, include_indices=False
    )
    exact = train.class_weights(exponent=1.0)
    smoothed = train.class_weights(exponent=0.5)

    assert exact.mean() == pytest.approx(1.0)
    assert smoothed.mean() == pytest.approx(1.0)
    # O expoente menor suaviza o upweight das classes raras.
    assert smoothed.max() < exact.max()


def _rare_tiles():
    """Tile de treino com pixels raros agrupados e tile de validação."""
    image = np.ones((128, 128, 9), dtype=np.float32)
    labels = np.zeros((128, 128), dtype=np.int64)
    labels[10:26, 10:26] = 3  # Solo Exposto
    labels[40:56, 40:56] = 4  # Área Construída
    labels[80:96, 80:96] = 5  # Água
    return {
        "train": (image, labels),
        "val": (image.copy(), labels.copy()),
    }


def test_rare_sampling_adds_centered_patches():
    tiles = _rare_tiles()
    plain = PatchDataset(
        tiles, "val", split="train", patch_size=32, stride=32,
        include_indices=False,
    )
    sampled = PatchDataset(
        tiles, "val", split="train", patch_size=32, stride=32,
        include_indices=False, rare_patches_per_class=2,
    )

    assert len(sampled) > len(plain)

    # Cada patch extra precisa conter a classe rara que o originou.
    extras = sampled.samples[len(plain.samples):]
    labels = tiles["train"][1]
    for _, y, x in extras:
        patch = labels[y:y + 32, x:x + 32]
        assert np.isin(patch, (3, 4, 5)).any()


def test_rare_sampling_not_applied_to_validation():
    tiles = _rare_tiles()
    val = PatchDataset(
        tiles, "val", split="val", patch_size=32, stride=32,
        include_indices=False, rare_patches_per_class=5,
    )
    # 128x128 com patch 32 e stride 32 -> 4x4 = 16 patches de grade.
    assert len(val) == 16


def test_multi_tile_validation_split():
    image = np.ones((128, 128, 9), dtype=np.float32)
    labels = np.zeros((128, 128), dtype=np.int64)
    labels[64:, :] = 1
    tiles = {
        "a": (image, labels),
        "b": (image.copy(), labels.copy()),
        "c": (image.copy(), labels.copy()),
    }
    train, val = make_patch_datasets(
        tiles, ["b", "c"], patch_size=64, stride=64, include_indices=False
    )
    assert {name for name, _, _ in train.samples} == {"a"}
    assert {name for name, _, _ in val.samples} == {"b", "c"}


def test_unknown_validation_tile_raises():
    with pytest.raises(KeyError):
        make_patch_datasets(_tiles(), "missing", include_indices=False)


def _empty_patch_tiles():
    """Tile com uma região válida e três patches 100% ignorados."""
    image = np.ones((128, 128, 9), dtype=np.float32)
    labels = np.full((128, 128), IGNORE_LABEL, dtype=np.int64)
    labels[:64, :32] = 1
    labels[:64, 32:64] = 2
    return {
        "train": (image, labels),
        "val": (image.copy(), labels.copy()),
    }


def test_sample_weights_zero_for_empty_patches():
    tiles = _empty_patch_tiles()
    dataset = PatchDataset(
        tiles, "val", split="train", patch_size=64, stride=64,
        include_indices=False,
    )
    weights = dataset.sample_weights()
    labels = tiles["train"][1]

    assert len(weights) == len(dataset)
    for (_, y, x), weight in zip(dataset.samples, weights):
        patch = labels[y:y + 64, x:x + 64]
        if (patch != IGNORE_LABEL).any():
            assert weight > 0
        else:
            assert weight == 0.0

    positive = weights[weights > 0]
    assert positive.size == 1
    assert positive.mean() == pytest.approx(1.0)


def test_array_sample_weights_zero_for_empty_patches():
    labels = np.full((2, 64, 64), IGNORE_LABEL, dtype=np.int64)
    labels[0, :32, :] = 0
    labels[0, 32:, :] = 1

    weights = _sample_weights(labels)

    assert weights.shape == (2,)
    assert weights[0] > 0
    assert weights[1] == 0.0
    assert weights[0] == pytest.approx(1.0)


def test_focal_loss_ignores_label_255():
    loss = FocalLoss(gamma=2.0, ignore_index=IGNORE_LABEL)
    logits = torch.zeros((1, NUM_CLASSES, 8, 8), requires_grad=True)
    target = torch.full((1, 8, 8), IGNORE_LABEL, dtype=torch.long)

    value = loss(logits, target)

    assert torch.isfinite(value)
    assert value.item() == pytest.approx(0.0)
    value.backward()
    assert logits.grad is not None


def test_focal_loss_weights_rare_errors_more():
    loss = FocalLoss(gamma=0.0)
    logits = torch.zeros((1, NUM_CLASSES, 4, 4))

    # Erro em uma classe rara (5) versus classe dominante (0).
    target_rare = torch.full((1, 4, 4), 5, dtype=torch.long)
    target_common = torch.full((1, 4, 4), 0, dtype=torch.long)
    common_loss = loss(logits, target_common)
    rare_loss = loss(logits, target_rare)

    # Sem gamma e sem pesos, ambas as classes têm o mesmo erro inicial.
    assert rare_loss.item() == pytest.approx(common_loss.item())

    weights = torch.ones(NUM_CLASSES)
    weights[5] = 10.0
    weighted = FocalLoss(gamma=0.0, weight=weights)
    assert weighted(logits, target_rare).item() == pytest.approx(
        10.0 * rare_loss.item()
    )
