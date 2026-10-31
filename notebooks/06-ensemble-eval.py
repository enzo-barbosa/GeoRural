"""Avaliação do ensemble (RF + SVM) em hold-out espacial por bioma.

Também gera ``outputs/ensemble_weights.json`` a partir dos Kappas da
validação espacial (usado pelo backend para ponderação).
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.classifiers.ensemble import soft_vote, weights_from_metrics
from src.classifiers.random_forest import train_random_forest
from src.classifiers.svm import train_svm
from src.evaluation.metrics import full_evaluation
from src.evaluation.spatial_split import spatial_holdout_indices
from src.evaluation.training import (
    SVM_MAX_SAMPLES,
    _cap_samples,
    load_domain_samples,
    train_domain,
)


def _proba(model, X, num_classes=6):
    proba = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    aligned = np.zeros((len(X), num_classes), dtype=np.float32)
    for column, class_id in enumerate(classes):
        aligned[:, class_id] = proba[:, column]
    return aligned


def build_weights(prefix: str) -> dict[str, float]:
    metrics_path = os.path.join(PROJECT_ROOT, f"outputs/metrics_{prefix}.json")
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    weights = weights_from_metrics(
        {k: v for k, v in metrics.items() if isinstance(v, dict)}
    )
    if weights:
        out_path = os.path.join(PROJECT_ROOT, "outputs/ensemble_weights.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
        print(f"Pesos salvos em {out_path}: {weights}")
    return weights


def main():
    csv_path = os.path.join(PROJECT_ROOT, "data/samples/brazil_samples.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "data/samples/brazil_samples.csv não encontrado. Rode a extração."
        )

    # Garante métricas/pesos atualizados (treina se necessário).
    metrics_path = os.path.join(PROJECT_ROOT, "outputs/metrics_brazil.json")
    if not os.path.exists(metrics_path):
        train_domain(csv_path, prefix="brazil")

    weights = build_weights("brazil")

    _, feature_cols, X, y, groups = load_domain_samples(csv_path)
    feature_cols = list(feature_cols)

    train_idx, test_idx = spatial_holdout_indices(
        groups, test_size=0.25, random_state=42
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"Features: {feature_cols}")

    rf = train_random_forest(X_train, y_train, verbose=0)
    X_svm, y_svm, _ = _cap_samples(
        X_train, y_train, groups[train_idx], SVM_MAX_SAMPLES
    )
    svm = train_svm(
        X_svm, y_svm, verbose=False, probability=True
    )

    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)
    rf_metrics = full_evaluation(y_test, rf_pred)
    svm_metrics = full_evaluation(y_test, svm_pred)

    proba = {
        "brazil_rf": _proba(rf, X_test),
        "brazil_svm": _proba(svm, X_test),
    }
    ensemble_proba = soft_vote(proba, weights)
    ensemble_pred = np.argmax(ensemble_proba, axis=-1)
    ensemble_metrics = full_evaluation(y_test, ensemble_pred)

    results = {
        "validation": "spatial_holdout",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "weights": weights,
        "brazil_rf": rf_metrics,
        "brazil_svm": svm_metrics,
        "ensemble": ensemble_metrics,
    }

    print("\n=== COMPARAÇÃO ===")
    for name in ("brazil_rf", "brazil_svm", "ensemble"):
        m = results[name]
        print(
            f"{name:<12} acc={m['overall_accuracy']:.4f} "
            f"kappa={m['kappa']:.4f}"
        )

    out_path = os.path.join(PROJECT_ROOT, "outputs/ensemble_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas salvas em {out_path}")


if __name__ == "__main__":
    main()
