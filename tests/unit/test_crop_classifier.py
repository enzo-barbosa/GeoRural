"""Testes para o classificador de tipo de cultivo."""

import numpy as np

from src.classifiers.crop_classifier import (
    CROP_NAMES,
    build_labeled_dataset,
    crop_feature_names,
    predict_crop_map,
    train_crop_classifier,
)


def test_dataset_shape_and_labels():
    X, y = build_labeled_dataset(samples_per_crop=20, seed=1)
    assert X.shape == (20 * len(CROP_NAMES), 24)
    assert set(np.unique(y).tolist()) == set(range(len(CROP_NAMES)))


def test_feature_names():
    names = crop_feature_names(12)
    assert len(names) == 24
    assert names[0] == "NDVI_00"
    assert names[1] == "EVI_00"


def test_train_and_predict():
    X, y = build_labeled_dataset(samples_per_crop=60, seed=7)
    model, metrics = train_crop_classifier(X, y, n_trees=50, max_depth=10)
    assert metrics["accuracy"] > 0.8

    sample = np.repeat(X[:4], 1, axis=0).reshape(2, 2, 24)
    prediction = predict_crop_map(model, sample)
    assert prediction.shape == (2, 2)
    assert prediction.max() < len(CROP_NAMES)
