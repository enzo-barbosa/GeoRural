"""Testes para métricas de avaliação."""

import numpy as np
import pytest
from src.evaluation.metrics import (
    overall_accuracy,
    precision_per_class,
    recall_per_class,
    f1_per_class,
    kappa_coefficient
)


def test_overall_accuracy_perfect():
    """Teste: acurácia perfeita deve retornar 1.0."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    assert overall_accuracy(y_true, y_pred) == 1.0


def test_overall_accuracy_zero():
    """Teste: nenhuma acerto deve retornar 0.0."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    assert overall_accuracy(y_true, y_pred) == 0.0


def test_kappa_perfect():
    """Teste: Kappa perfeito deve retornar 1.0."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert kappa_coefficient(y_true, y_pred) == 1.0


def test_precision_per_class():
    """Teste: verificar cálculo de precisão por classe."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    precision = precision_per_class(y_true, y_pred)
    assert precision["Agricultura"] == 0.5  # 1/2
