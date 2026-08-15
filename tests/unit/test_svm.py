"""Testes para Support Vector Machine."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.classifiers.svm import train_svm, predict_svm


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=5,
        n_informative=5,
        random_state=42,
    )
    return X, y


def test_svm_training(sample_data):
    X, y = sample_data
    model = train_svm(X[:120], y[:120])
    predictions = predict_svm(model, X[120:])
    assert len(predictions) == len(y[120:])


def test_svm_accuracy_above_random(sample_data):
    X, y = sample_data
    model = train_svm(X[:120], y[:120])
    predictions = predict_svm(model, X[120:])
    accuracy = np.mean(predictions == y[120:])
    assert accuracy > 0.3
