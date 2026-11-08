"""Testes de calibração (ECE/Brier/NLL)."""

import numpy as np

from src.evaluation.calibration import (
    brier_score,
    calibrate_temperature,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_bins,
)


def test_perfect_probabilities_have_zero_error():
    y = np.array([0, 1, 2])
    probabilities = np.eye(3, dtype=np.float32)
    assert expected_calibration_error(y, probabilities) == 0.0
    assert brier_score(y, probabilities) < 1e-12
    assert negative_log_likelihood(y, probabilities) < 1e-6


def test_overconfident_wrong_prediction_has_positive_ece():
    y = np.array([1, 1])
    probabilities = np.array([[0.95, 0.05], [0.95, 0.05]], dtype=np.float32)
    assert expected_calibration_error(y, probabilities) > 0.5


def test_reliability_bins_cover_all_samples():
    y = np.array([0, 1, 1, 0])
    probabilities = np.array(
        [[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.9, 0.1]], dtype=np.float32
    )
    bins = reliability_bins(y, probabilities, n_bins=5)
    assert sum(item["count"] for item in bins) == len(y)


def test_temperature_scaling_keeps_probabilities_valid():
    y = np.array([0, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    temperature = fit_temperature(y, probabilities)
    calibrated = calibrate_temperature(probabilities, temperature)
    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)
