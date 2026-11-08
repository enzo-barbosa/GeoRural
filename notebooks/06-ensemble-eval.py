"""Avaliação do ensemble (RF + SVM) em hold-out espacial por bioma."""

import argparse
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


def build_weights(prefix: str, output_dir: str) -> dict[str, float]:
    metrics_path = os.path.join(output_dir, f"metrics_{prefix}.json")
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    weights = weights_from_metrics(
        {k: v for k, v in metrics.items() if isinstance(v, dict)}
    )
    if weights:
        out_path = os.path.join(output_dir, "ensemble_weights.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
        print(f"Pesos salvos em {out_path}: {weights}")
    return weights


def main():
    parser = argparse.ArgumentParser(description="Avalia o ensemble RF+SVM")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--prefix", default="brazil")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--svm-max", type=int, default=20_000)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-depth", type=int, default=25)
    parser.add_argument("--filter-regions", nargs="+", default=None)
    parser.add_argument(
        "--group-strategy",
        default="auto",
        choices=["auto", "block", "region", "biome"],
    )
    args = parser.parse_args()

    prefix = args.prefix
    models_dir = args.models_dir or os.path.join(PROJECT_ROOT, "models")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs")
    csv_path = args.csv or os.path.join(
        PROJECT_ROOT, f"data/samples/{prefix}_samples.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    metrics_path = os.path.join(output_dir, f"metrics_{prefix}.json")
    if not os.path.exists(metrics_path):
        train_domain(
            csv_path,
            prefix=prefix,
            svm_max_samples=args.svm_max,
            models_dir=models_dir,
            output_dir=output_dir,
            rf_n_trees=args.rf_trees,
            rf_max_depth=args.rf_depth,
            filter_regions=args.filter_regions,
            group_strategy=args.group_strategy,
        )

    weights = build_weights(prefix, output_dir)

    _, _feature_cols, X, y, groups = load_domain_samples(
        csv_path,
        filter_regions=args.filter_regions,
        group_strategy=args.group_strategy,
    )
    train_idx, test_idx = spatial_holdout_indices(
        groups, test_size=0.25, random_state=42
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    rf = train_random_forest(
        X_train,
        y_train,
        n_trees=args.rf_trees,
        max_depth=args.rf_depth,
        verbose=0,
    )
    X_svm, y_svm, _ = _cap_samples(
        X_train, y_train, groups[train_idx], args.svm_max
    )
    svm = train_svm(
        X_svm, y_svm, verbose=False, probability=True
    )

    rf_name = f"{prefix}_rf"
    svm_name = f"{prefix}_svm"
    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)
    rf_metrics = full_evaluation(y_test, rf_pred)
    svm_metrics = full_evaluation(y_test, svm_pred)

    proba = {
        rf_name: _proba(rf, X_test),
        svm_name: _proba(svm, X_test),
    }
    ensemble_proba = soft_vote(proba, weights)
    ensemble_pred = np.argmax(ensemble_proba, axis=-1)
    ensemble_metrics = full_evaluation(y_test, ensemble_pred)

    results = {
        "validation": "spatial_holdout",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "weights": weights,
        rf_name: rf_metrics,
        svm_name: svm_metrics,
        "ensemble": ensemble_metrics,
    }

    print("\n=== COMPARAÇÃO ===")
    for name in (rf_name, svm_name, "ensemble"):
        metrics = results[name]
        print(
            f"{name:<14} acc={metrics['overall_accuracy']:.4f} "
            f"kappa={metrics['kappa']:.4f}"
        )

    out_path = os.path.join(output_dir, "ensemble_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas salvas em {out_path}")


if __name__ == "__main__":
    main()
