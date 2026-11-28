"""Testes de alinhamento e acumulação de probabilidades entre modelos."""

import numpy as np

from backend.app.services.classification_service import (
    _align_probabilities,
    _domain_model_name,
)


class _DummyClassifier:
    n_features_in_ = 9
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        proba = np.zeros((len(features), 2), dtype=np.float32)
        proba[:, 0] = 0.7
        proba[:, 1] = 0.3
        return proba


def test_align_legacy_five_class_model_to_six():
    probabilities = np.array([[0.1, 0.2, 0.3, 0.2, 0.2]], dtype=np.float32)
    aligned = _align_probabilities(probabilities, [0, 1, 2, 3, 4], num_classes=6)
    assert aligned.shape == (1, 6)
    assert aligned[0, 5] == 0.0
    assert np.isclose(aligned.sum(), 1.0)


def test_align_reorders_columns_by_class_id():
    probabilities = np.array([[0.9, 0.1]], dtype=np.float32)
    aligned = _align_probabilities(probabilities, [3, 5], num_classes=6)
    assert aligned[0, 3] == 0.9
    assert aligned[0, 5] == 0.1


def test_domain_model_name_maps_aliases():
    assert _domain_model_name("random_forest", "brazil") == "brazil_rf"
    assert _domain_model_name("svm", "global") == "global_svm"
    assert _domain_model_name("unet_cnn", "brazil") == "brazil_unet"


def test_probability_maps_accumulate_in_chunks(monkeypatch):
    from backend.app.config import get_settings
    from backend.app.services import classification_service as service

    settings = get_settings()
    monkeypatch.setattr(settings, "INFERENCE_CHUNK_SIZE", 2)
    monkeypatch.setattr(service, "check_model_exists", lambda name: True)
    monkeypatch.setattr(
        service, "get_cached_model", lambda name: _DummyClassifier()
    )
    monkeypatch.setattr(
        service,
        "_load_ensemble_weights",
        lambda coverage, models: {
            "brazil_rf": 0.6,
            "brazil_svm": 0.4,
            "brazil_unet": 0.0,
        },
    )

    def fail_unet(*args, **kwargs):
        raise ValueError("U-Net indisponível no teste")

    monkeypatch.setattr(service, "predict_unet_proba", fail_unet)

    image = np.ones((2, 3, 15), dtype=np.float32)
    stages: list[tuple[str, int]] = []
    pred, proba, used, weights = service._predict_probability_maps(
        image,
        "sentinel2",
        "brazil",
        "auto",
        on_progress=lambda stage, progress: stages.append((stage, progress)),
    )

    assert ("loading_models", 62) in stages
    assert ("inference", 70) in stages
    assert used == ["brazil_rf", "brazil_svm"]
    assert pred.shape == (2, 3)
    assert proba.shape == (2, 3, 6)
    np.testing.assert_allclose(proba[..., 0], 0.7, atol=1e-6)
    np.testing.assert_allclose(proba[..., 1], 0.3, atol=1e-6)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
