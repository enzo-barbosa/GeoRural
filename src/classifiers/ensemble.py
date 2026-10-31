"""Combinação de modelos por votação suave (soft voting)."""

import json
import os

import numpy as np


def normalize_weights(
    models: list[str], weights: dict[str, float] | None
) -> dict[str, float]:
    """Normaliza pesos para os modelos disponíveis (iguais se ausentes)."""
    if not models:
        return {}

    if not weights:
        return {name: 1.0 / len(models) for name in models}

    selected = {name: max(float(weights.get(name, 0.0)), 0.0) for name in models}
    total = sum(selected.values())
    if total <= 0:
        return {name: 1.0 / len(models) for name in models}
    return {name: value / total for name, value in selected.items()}


def soft_vote(
    probabilities: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Média ponderada de mapas de probabilidade (N, C) ou (H, W, C)."""
    if not probabilities:
        raise ValueError("Nenhuma probabilidade fornecida.")

    models = list(probabilities)
    normalized = normalize_weights(models, weights)

    combined = None
    for name in models:
        proba = np.asarray(probabilities[name], dtype=np.float32)
        weighted = proba * normalized[name]
        combined = weighted if combined is None else combined + weighted

    return combined.astype(np.float32)


def confidence_from_proba(probabilities: np.ndarray) -> np.ndarray:
    """Confiança por pixel = maior probabilidade."""
    return np.max(probabilities, axis=-1)


def load_weights(path: str) -> dict[str, float]:
    """Carrega pesos de um JSON (retorna vazio se ausente/inválido)."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items()}
    except (OSError, ValueError, TypeError):
        return {}


def weights_from_metrics(metrics: dict) -> dict[str, float]:
    """Deriva pesos a partir de Kappa médio em um JSON de métricas."""
    weights = {}
    for key, value in metrics.items():
        if not isinstance(value, dict):
            continue
        kappa = value.get("kappa_mean")
        if kappa is None:
            kappa = value.get("kappa")
        if kappa is not None:
            weights[key] = max(float(kappa), 0.0)
    return weights
