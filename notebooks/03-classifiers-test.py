"""Treino e avaliação dos classificadores clássicos.

Avaliação com validação espacial (blocos geográficos) quando o CSV possui
longitude/latitude; caso contrário, usa Hold-Out aleatório.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.classifiers.random_forest import (
    predict_random_forest,
    prepare_data_rf,
    train_random_forest,
)
from src.classifiers.random_forest import save_model as save_rf
from src.classifiers.svm import predict_svm, train_svm
from src.classifiers.svm import save_model as save_svm
from src.evaluation.confusion import plot_confusion_matrix, plot_feature_importance
from src.evaluation.metrics import full_evaluation, print_evaluation
from src.evaluation.spatial_split import (
    assign_spatial_blocks,
    spatial_kfold_indices,
)


def _evaluate_cv(model_fn, predict_fn, splits, X, y):
    """Avalia um modelo com folds espaciais e retorna métricas + OOF."""
    accs, kappas = [], []
    oof_true = np.full(len(y), -1, dtype=y.dtype)
    oof_pred = np.full(len(y), -1, dtype=y.dtype)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = model_fn(X[train_idx], y[train_idx])
        predictions = predict_fn(model, X[test_idx])

        oof_true[test_idx] = y[test_idx]
        oof_pred[test_idx] = predictions

        metrics = full_evaluation(y[test_idx], predictions)
        accs.append(metrics["overall_accuracy"])
        kappas.append(metrics["kappa"])
        print(
            f"  Fold {fold}: Acurácia={metrics['overall_accuracy']:.4f} "
            f"Kappa={metrics['kappa']:.4f}"
        )

    mask = oof_true >= 0
    return np.array(accs), np.array(kappas), oof_true[mask], oof_pred[mask]


def main():
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data/samples/training_samples.csv")
    )

    feature_cols = [c for c in df.columns if c in (
        "B2", "B3", "B4", "B8", "B11", "B12", "NDVI",
        "contrast", "dissimilarity",
    )]
    X = df[feature_cols].values
    y = df["class_id"].values

    print(f"Total de amostras: {len(X)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    if {"longitude", "latitude"}.issubset(df.columns):
        groups = assign_spatial_blocks(df["longitude"], df["latitude"])
        splits = spatial_kfold_indices(groups, n_splits=5)
        print(
            f"Validação espacial: {len(splits)} folds, "
            f"{len(np.unique(groups))} blocos"
        )
    else:
        groups = None
        splits = None
        print("Aviso: coordenadas ausentes — usando Hold-Out aleatório.")

    metrics_summary = {"validation": "spatial_kfold" if groups is not None else "holdout"}

    if splits is not None:
        print("\n" + "=" * 60)
        print("RANDOM FOREST (validação espacial)")
        print("=" * 60)
        accs, kappas, rf_true, rf_pred = _evaluate_cv(
            lambda a, b: train_random_forest(a, b, verbose=0),
            predict_random_forest,
            splits,
            X,
            y,
        )
        print(f"RF Acurácia: {accs.mean():.4f} ± {accs.std():.4f}")
        print(f"RF Kappa:    {kappas.mean():.4f} ± {kappas.std():.4f}")
        print_evaluation(rf_true, rf_pred)
        plot_confusion_matrix(
            rf_true,
            rf_pred,
            save_path=os.path.join(
                PROJECT_ROOT, "outputs/rf_spatial_confusion_matrix.png"
            ),
        )

        print("\n" + "=" * 60)
        print("SVM (validação espacial)")
        print("=" * 60)
        svm_accs, svm_kappas, svm_true, svm_pred = _evaluate_cv(
            lambda a, b: train_svm(a, b, verbose=False),
            predict_svm,
            splits,
            X,
            y,
        )
        print(f"SVM Acurácia: {svm_accs.mean():.4f} ± {svm_accs.std():.4f}")
        print(f"SVM Kappa:    {svm_kappas.mean():.4f} ± {svm_kappas.std():.4f}")
        print_evaluation(svm_true, svm_pred)
        plot_confusion_matrix(
            svm_true,
            svm_pred,
            save_path=os.path.join(
                PROJECT_ROOT, "outputs/svm_spatial_confusion_matrix.png"
            ),
        )

        metrics_summary.update(
            {
                "random_forest": {
                    "accuracy_mean": float(accs.mean()),
                    "accuracy_std": float(accs.std()),
                    "kappa_mean": float(kappas.mean()),
                    "kappa_std": float(kappas.std()),
                },
                "svm": {
                    "accuracy_mean": float(svm_accs.mean()),
                    "accuracy_std": float(svm_accs.std()),
                    "kappa_mean": float(svm_kappas.mean()),
                    "kappa_std": float(svm_kappas.std()),
                },
            }
        )
    else:
        X_train, X_test, y_train, y_test = prepare_data_rf(X, y)
        rf_model = train_random_forest(X_train, y_train)
        svm_model = train_svm(X_train, y_train)
        rf_pred = predict_random_forest(rf_model, X_test)
        svm_pred = predict_svm(svm_model, X_test)
        print_evaluation(y_test, rf_pred)
        print_evaluation(y_test, svm_pred)

    # Modelos finais treinados em todos os dados (para produção)
    print("\nTreinando modelos finais em todos os dados...")
    final_rf = train_random_forest(X, y, verbose=0)
    final_svm = train_svm(X, y, verbose=False)

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    save_rf(final_rf, os.path.join(PROJECT_ROOT, "models/random_forest.pkl"))
    save_svm(final_svm, os.path.join(PROJECT_ROOT, "models/svm.pkl"))

    plot_feature_importance(
        final_rf.feature_importances_,
        feature_cols,
        save_path=os.path.join(PROJECT_ROOT, "outputs/rf_feature_importance.png"),
    )

    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    with open(
        os.path.join(PROJECT_ROOT, "outputs/classifier_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print("\nMétricas salvas em outputs/classifier_metrics.json")


if __name__ == "__main__":
    main()
