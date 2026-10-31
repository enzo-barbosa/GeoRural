"""Testes de alinhamento de probabilidades entre modelos."""

import numpy as np

from backend.app.services.classification_service import (
    _align_probabilities,
    _domain_model_name,
)


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
