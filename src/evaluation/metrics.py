"""Métricas de avaliação de classificadores."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import CLASS_NAMES


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a Acurácia Global.

    Proporção do total de pixels corretamente classificados.
    """
    return float(accuracy_score(y_true, y_pred))


def precision_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula a Precisão para cada classe."""
    precisions = precision_score(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    return {name: float(p) for name, p in zip(CLASS_NAMES, precisions)}


def recall_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula o Recall para cada classe."""
    recalls = recall_score(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    return {name: float(r) for name, r in zip(CLASS_NAMES, recalls)}


def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula o F1-Score para cada classe."""
    f1s = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    return {name: float(f) for name, f in zip(CLASS_NAMES, f1s)}


def kappa_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Coeficiente Kappa."""
    return float(cohen_kappa_score(y_true, y_pred))


def full_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Gera relatório completo de avaliação.

    Returns:
        Dict com todas as métricas calculadas
    """
    return {
        "overall_accuracy": overall_accuracy(y_true, y_pred),
        "precision": precision_per_class(y_true, y_pred),
        "recall": recall_per_class(y_true, y_pred),
        "f1_score": f1_per_class(y_true, y_pred),
        "kappa": kappa_coefficient(y_true, y_pred),
    }


def print_evaluation(y_true: np.ndarray, y_pred: np.ndarray):
    """Imprime relatório de avaliação formatado."""
    metrics = full_evaluation(y_true, y_pred)

    print("=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO")
    print("=" * 60)
    print(f"\nAcurácia Global: {metrics['overall_accuracy']:.4f}")
    print(f"Coeficiente Kappa: {metrics['kappa']:.4f}")
    print("\nMétricas por classe:")
    print("-" * 60)
    print(f"{'Classe':<20} {'Precisão':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 60)

    for class_name in CLASS_NAMES:
        p = metrics["precision"].get(class_name, 0.0)
        r = metrics["recall"].get(class_name, 0.0)
        f = metrics["f1_score"].get(class_name, 0.0)
        print(f"{class_name:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    print("=" * 60)
