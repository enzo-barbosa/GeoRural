"""Testes para classificadores."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.classifiers.random_forest import (
    prepare_data_rf, train_random_forest, predict_random_forest
)


@pytest.fixture
def sample_data():
    """Gera dados sintéticos para teste."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=5,
        n_informative=5,
        random_state=42
    )
    return X, y


def test_random_forest_training(sample_data):
    """Teste: RF deve treinar sem erros."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = prepare_data_rf(X, y)
    model = train_random_forest(X_train, y_train, n_trees=10, max_depth=5)
    predictions = predict_random_forest(model, X_test)
    assert len(predictions) == len(y_test)


def test_random_forest_accuracy_above_random(sample_data):
    """Teste: RF deve ter acurácia acima do aleatório."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = prepare_data_rf(X, y)
    model = train_random_forest(X_train, y_train, n_trees=10, max_depth=5)
    predictions = predict_random_forest(model, X_test)
    accuracy = np.mean(predictions == y_test)
    assert accuracy > 0.3  # Acima do aleatório (1/5 = 0.2)
