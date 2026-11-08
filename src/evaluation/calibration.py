"""Métricas de calibração de probabilidades (ECE, Brier, NLL)."""

import json
import os

import numpy as np

try:
    from scipy.optimize import minimize_scalar
except ImportError:  # pragma: no cover
    minimize_scalar = None


def _confidence_and_prediction(probabilities: np.ndarray):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    return confidence, prediction


def reliability_bins(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Faixas de confiança com acurácia observada."""
    y_true = np.asarray(y_true, dtype=int)
    confidence, prediction = _confidence_and_prediction(probabilities)
    correct = (prediction == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        mask = (confidence >= low) & (
            confidence < high if index < n_bins - 1 else confidence <= high
        )
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                }
            )
            continue
        bins.append(
            {
                "low": float(low),
                "high": float(high),
                "count": count,
                "confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
            }
        )
    return bins


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: média ponderada de |acurácia - confiança|."""
    bins = reliability_bins(y_true, probabilities, n_bins=n_bins)
    total = sum(item["count"] for item in bins)
    if total == 0:
        return 0.0
    return float(
        sum(
            item["count"] / total
            * abs((item["accuracy"] or 0.0) - (item["confidence"] or 0.0))
            for item in bins
            if item["count"] > 0
        )
    )


def maximum_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """MCE: maior desvio entre acurácia e confiança nas faixas."""
    bins = reliability_bins(y_true, probabilities, n_bins=n_bins)
    values = [
        abs(item["accuracy"] - item["confidence"])
        for item in bins
        if item["count"] > 0
        and item["accuracy"] is not None
        and item["confidence"] is not None
    ]
    return float(max(values)) if values else 0.0


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Brier score multiclasse (menor é melhor)."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def negative_log_likelihood(
    y_true: np.ndarray, probabilities: np.ndarray
) -> float:
    """NLL média da classe verdadeira."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0)
    return float(-np.mean(np.log(probabilities[np.arange(len(y_true)), y_true])))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(
    y_true: np.ndarray, probabilities: np.ndarray
) -> float:
    """Ajusta temperatura T para minimizar NLL (calibração pós-hoc)."""
    if minimize_scalar is None:
        return 1.0

    y_true = np.asarray(y_true, dtype=int)
    logits = np.log(np.clip(probabilities, 1e-9, 1.0))

    def objective(temperature):
        scaled = _softmax(logits / max(float(temperature), 1e-3))
        return negative_log_likelihood(y_true, scaled)

    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    return float(result.x) if result.success else 1.0


def calibrate_temperature(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """Aplica temperatura às probabilidades."""
    logits = np.log(np.clip(probabilities, 1e-9, 1.0))
    return _softmax(logits / max(float(temperature), 1e-3))


def calibration_summary(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> dict:
    """Resumo de calibração bruto + temperatura ajustada."""
    temperature = fit_temperature(y_true, probabilities)
    calibrated = calibrate_temperature(probabilities, temperature)
    return {
        "temperature": temperature,
        "raw": {
            "ece": expected_calibration_error(y_true, probabilities, n_bins),
            "mce": maximum_calibration_error(y_true, probabilities, n_bins),
            "brier": brier_score(y_true, probabilities),
            "nll": negative_log_likelihood(y_true, probabilities),
            "bins": reliability_bins(y_true, probabilities, n_bins),
        },
        "calibrated": {
            "ece": expected_calibration_error(y_true, calibrated, n_bins),
            "mce": maximum_calibration_error(y_true, calibrated, n_bins),
            "brier": brier_score(y_true, calibrated),
            "nll": negative_log_likelihood(y_true, calibrated),
            "bins": reliability_bins(y_true, calibrated, n_bins),
        },
    }


def plot_reliability(
    bins: list[dict], save_path: str, title: str = "Reliability diagram"
) -> None:
    """Gera o diagrama de confiabilidade (confiança × acurácia)."""
    import matplotlib.pyplot as plt

    points = [
        (item["confidence"], item["accuracy"], item["count"])
        for item in bins
        if item["count"] > 0
    ]
    if not points:
        return

    confidence = [point[0] for point in points]
    accuracy = [point[1] for point in points]
    counts = np.array([point[2] for point in points], dtype=float)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfeita calibração")
    plt.scatter(confidence, accuracy, s=40 + counts / counts.max() * 200)
    plt.plot(confidence, accuracy, color="#2e7d32", label="Modelo")
    plt.xlabel("Confiança")
    plt.ylabel("Acurácia observada")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    with open(
        os.path.splitext(save_path)[0] + ".json", "w", encoding="utf-8"
    ) as file:
        json.dump(bins, file, indent=2, ensure_ascii=False)
