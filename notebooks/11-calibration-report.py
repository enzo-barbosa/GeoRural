"""Relatório de calibração (ECE, Brier, NLL, reliability diagram)."""

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
from src.evaluation.calibration import (
    calibration_summary,
    plot_reliability,
)
from src.evaluation.metrics import full_evaluation
from src.evaluation.spatial_split import spatial_holdout_indices
from src.evaluation.training import _cap_samples, load_domain_samples


def _proba(model, X, num_classes=6):
    proba = model.predict_proba(X)
    aligned = np.zeros((len(X), num_classes), dtype=np.float32)
    for column, class_id in enumerate(model.classes_):
        aligned[:, int(class_id)] = proba[:, column]
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Relatório de calibração")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prefix", default="sp")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--filter-regions", nargs="+", default=None)
    parser.add_argument(
        "--group-strategy",
        default="block",
        choices=["auto", "block", "region", "biome"],
    )
    parser.add_argument("--svm-max", type=int, default=20_000)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-depth", type=int, default=25)
    parser.add_argument(
        "--no-local-indices",
        action="store_true",
        help="Usa apenas as 9 features originais (modelos de produção).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs/full")
    os.makedirs(output_dir, exist_ok=True)

    _, _, X, y, groups = load_domain_samples(
        args.csv,
        filter_regions=args.filter_regions,
        group_strategy=args.group_strategy,
        include_indices=not args.no_local_indices,
    )
    train_idx, test_idx = spatial_holdout_indices(
        groups, test_size=0.25, random_state=42
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"{args.prefix}: treino={len(X_train)} teste={len(X_test)}")

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
    svm = train_svm(X_svm, y_svm, verbose=False, probability=True)

    rf_name = f"{args.prefix}_rf"
    svm_name = f"{args.prefix}_svm"
    probabilities = {
        rf_name: _proba(rf, X_test),
        svm_name: _proba(svm, X_test),
    }

    metrics_path = os.path.join(output_dir, f"metrics_{args.prefix}.json")
    weights = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as file:
            metrics = json.load(file)
        weights = weights_from_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, dict)}
        )

    ensemble_proba = soft_vote(probabilities, weights)
    predictions = ensemble_proba.argmax(axis=1)

    evaluation = full_evaluation(y_test, predictions)
    calibration = calibration_summary(y_test, ensemble_proba)

    report = {
        "prefix": args.prefix,
        "validation": "spatial_holdout",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "weights": weights,
        "accuracy": evaluation["overall_accuracy"],
        "kappa": evaluation["kappa"],
        "f1_score": evaluation["f1_score"],
        "confidence_mean_raw": float(ensemble_proba.max(axis=1).mean()),
        "calibration": calibration,
    }

    report_path = os.path.join(output_dir, f"calibration_{args.prefix}.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    raw_png = os.path.join(output_dir, f"reliability_{args.prefix}_raw.png")
    calibrated_png = os.path.join(
        output_dir, f"reliability_{args.prefix}_calibrated.png"
    )
    plot_reliability(
        calibration["raw"]["bins"],
        raw_png,
        title=f"Calibração bruta — {args.prefix}",
    )
    plot_reliability(
        calibration["calibrated"]["bins"],
        calibrated_png,
        title=f"Calibração com temperatura — {args.prefix}",
    )

    print(f"Acurácia: {report['accuracy']:.4f} | Kappa: {report['kappa']:.4f}")
    print(
        f"Confiança média: {report['confidence_mean_raw']:.4f} | "
        f"ECE bruto: {calibration['raw']['ece']:.4f} | "
        f"ECE calibrado: {calibration['calibrated']['ece']:.4f} | "
        f"T={calibration['temperature']:.3f}"
    )
    print(f"Relatório: {report_path}")


if __name__ == "__main__":
    main()
