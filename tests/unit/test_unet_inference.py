"""Testes de inferência da U-Net em áreas menores que o patch."""

import numpy as np
import pytest

from src.classifiers.unet_cnn import (
    _pad_to_grid,
    build_unet,
    predict_unet_proba,
)
from src.config import NUM_CLASSES


@pytest.fixture(scope="module")
def model():
    return build_unet(input_shape=(64, 64, 15), num_classes=NUM_CLASSES)


def test_pad_to_grid_accepts_small_images():
    padded, original_h, original_w = _pad_to_grid(
        np.zeros((24, 83, 15), dtype=np.float32), patch_size=64, stride=32
    )
    assert (original_h, original_w) == (24, 83)
    assert padded.shape[:2] == (64, 96)


def test_pad_to_grid_keeps_large_images():
    padded, original_h, original_w = _pad_to_grid(
        np.zeros((64, 64, 15), dtype=np.float32), patch_size=64, stride=32
    )
    assert (original_h, original_w) == (64, 64)
    assert padded.shape[:2] == (64, 64)


def test_predict_unet_proba_small_image(model):
    image = np.random.default_rng(0).random((24, 83, 15), dtype=np.float32)
    proba = predict_unet_proba(model, image, patch_size=64, stride=32)

    assert proba.shape == (24, 83, NUM_CLASSES)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=-1), 1.0, atol=1e-5)


def test_predict_unet_proba_tiny_image(model):
    image = np.random.default_rng(1).random((1, 1, 15), dtype=np.float32)
    proba = predict_unet_proba(model, image, patch_size=64, stride=32)

    assert proba.shape == (1, 1, NUM_CLASSES)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=-1), 1.0, atol=1e-5)


def test_batched_inference_matches_single_patch(model):
    image = np.random.default_rng(2).random((48, 96, 15), dtype=np.float32)
    single = predict_unet_proba(
        model, image, patch_size=64, stride=32, batch_size=1
    )
    batched = predict_unet_proba(
        model, image, patch_size=64, stride=32, batch_size=8
    )

    np.testing.assert_allclose(single, batched, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(
        np.argmax(single, axis=-1), np.argmax(batched, axis=-1)
    )


class _DummyClassifier:
    n_features_in_ = 9
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        proba = np.zeros((len(features), 2), dtype=np.float32)
        proba[:, 0] = 0.7
        proba[:, 1] = 0.3
        return proba


def test_unet_failure_keeps_other_models(monkeypatch):
    from backend.app.services import classification_service as service

    monkeypatch.setattr(service, "check_model_exists", lambda name: True)
    monkeypatch.setattr(service, "get_cached_model", lambda name: _DummyClassifier())

    def fail(*args, **kwargs):
        raise ValueError("Imagem (1x1) menor que patch_size (64).")

    monkeypatch.setattr(service, "predict_unet_proba", fail)

    image = np.ones((24, 83, 15), dtype=np.float32)
    pred, _proba, used, weights = service._predict_probability_maps(
        image, "sentinel2", "brazil", "auto"
    )

    assert used == ["brazil_rf", "brazil_svm"]
    assert pred.shape == (24, 83)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
