"""Testes do ensemble por votação suave."""

import numpy as np

from src.classifiers.ensemble import (
    confidence_from_proba,
    normalize_weights,
    soft_vote,
    weights_from_metrics,
)


def test_normalize_weights_equal_when_absent():
    weights = normalize_weights(["a", "b", "c"], None)
    assert weights == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def test_normalize_weights_filters_and_normalizes():
    weights = normalize_weights(["a", "b"], {"a": 3, "b": 1, "c": 10})
    assert abs(weights["a"] - 0.75) < 1e-6
    assert abs(weights["b"] - 0.25) < 1e-6
    assert "c" not in weights


def test_soft_vote_weighted_average():
    proba_a = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    proba_b = np.array([[0.4, 0.6], [0.7, 0.3]], dtype=np.float32)

    combined = soft_vote({"a": proba_a, "b": proba_b}, {"a": 0.75, "b": 0.25})
    expected = proba_a * 0.75 + proba_b * 0.25

    assert np.allclose(combined, expected)
    assert np.allclose(combined.sum(axis=1), 1.0)


def test_confidence_is_max_probability():
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    assert np.allclose(confidence_from_proba(proba), [0.8, 0.6])


def test_weights_from_metrics_uses_kappa():
    metrics = {
        "brazil_rf": {"kappa_mean": 0.6},
        "brazil_svm": {"kappa_mean": 0.8},
    }
    weights = weights_from_metrics(metrics)
    assert weights == {"brazil_rf": 0.6, "brazil_svm": 0.8}
