"""Treino e avaliação de RF/SVM por domínio (Brasil ou global)."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.classifiers.random_forest import (
    predict_random_forest,
    train_random_forest,
)
from src.classifiers.random_forest import (
    save_model as save_rf,
)
from src.classifiers.svm import predict_svm, train_svm
from src.classifiers.svm import save_model as save_svm
from src.evaluation.confusion import plot_confusion_matrix, plot_feature_importance
from src.evaluation.metrics import full_evaluation, print_evaluation
from src.evaluation.spatial_split import (
    assign_spatial_blocks,
    spatial_kfold_indices,
)
from src.processing.local_features import LOCAL_INDEX_NAMES, add_local_indices

BASE_FEATURE_COLUMNS = (
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "contrast",
    "dissimilarity",
)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + tuple(LOCAL_INDEX_NAMES)
SVM_MAX_SAMPLES = 20_000


def load_domain_samples(
    path: str,
    filter_regions: list[str] | None = None,
    group_strategy: str = "auto",
    include_indices: bool = True,
):
    """Carrega CSV e devolve (df, features, X, y, groups).

    Args:
        path: CSV de amostras.
        filter_regions: mantém apenas estes ``region_id`` (ex.: polos de SP).
        group_strategy: ``auto`` (bioma/região), ``block`` (blocos espaciais),
            ``region`` ou ``biome``.
    """
    df = pd.read_csv(path)
    if filter_regions:
        if "region_id" not in df.columns:
            raise ValueError("filter_regions exige a coluna region_id.")
        df = df[df["region_id"].isin(filter_regions)].reset_index(drop=True)

    if include_indices:
        df = add_local_indices(df)
        expected_features = FEATURE_COLUMNS
    else:
        expected_features = BASE_FEATURE_COLUMNS

    feature_cols = [c for c in expected_features if c in df.columns]
    if len(feature_cols) != len(expected_features):
        raise ValueError(
            f"Features ausentes em {path}. Colunas: {df.columns.tolist()}"
        )
    if "class_id" not in df.columns:
        raise ValueError(f"Coluna class_id ausente em {path}.")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["class_id"].to_numpy(dtype=np.int64)

    if group_strategy == "block" and {"longitude", "latitude"}.issubset(
        df.columns
    ):
        groups = assign_spatial_blocks(
            df["longitude"], df["latitude"], block_size=0.02
        )
    elif group_strategy == "region" and "region_id" in df.columns:
        groups = pd.factorize(df["region_id"].astype(str))[0]
    elif "biome" in df.columns and (
        group_strategy == "biome"
        or (group_strategy == "auto" and df["biome"].nunique() >= 2)
    ):
        groups = pd.factorize(df["biome"].astype(str))[0]
    elif (
        group_strategy == "auto"
        and "region_id" in df.columns
        and df["region_id"].nunique() >= 2
    ):
        groups = pd.factorize(df["region_id"].astype(str))[0]
    elif {"longitude", "latitude"}.issubset(df.columns):
        groups = assign_spatial_blocks(df["longitude"], df["latitude"])
    else:
        groups = np.arange(len(df), dtype=np.int64)

    return df, feature_cols, X, y, groups


def _cap_samples(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, max_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subamostra estratificada para viabilizar SVM em bases grandes."""
    if len(X) <= max_samples:
        return X, y, groups

    _, indices = train_test_split(
        np.arange(len(y)),
        test_size=max_samples / len(y),
        stratify=y,
        random_state=42,
    )
    return X[indices], y[indices], groups[indices]


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


def _evaluate_one(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    output_dir: str,
    rf_n_trees: int = 300,
    rf_max_depth: int = 25,
    rf_min_samples_leaf: int = 1,
):
    splits = spatial_kfold_indices(groups, n_splits=n_splits)
    print(f"\n{name}: {len(splits)} folds, {len(np.unique(groups))} grupos")

    if name == "Random Forest":
        accs, kappas, oof_true, oof_pred = _evaluate_cv(
            lambda a, b: train_random_forest(
                a,
                b,
                n_trees=rf_n_trees,
                max_depth=rf_max_depth,
                min_samples_leaf=rf_min_samples_leaf,
                verbose=0,
            ),
            predict_random_forest,
            splits,
            X,
            y,
        )
        prefix = "rf"
    else:
        accs, kappas, oof_true, oof_pred = _evaluate_cv(
            lambda a, b: train_svm(a, b, verbose=False),
            predict_svm,
            splits,
            X,
            y,
        )
        prefix = "svm"

    print(f"{name} Acurácia: {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"{name} Kappa:    {kappas.mean():.4f} ± {kappas.std():.4f}")
    print_evaluation(oof_true, oof_pred)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_confusion_matrix(
            oof_true,
            oof_pred,
            save_path=os.path.join(output_dir, f"{prefix}_spatial_confusion_matrix.png"),
        )

    return {
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "kappa_mean": float(kappas.mean()),
        "kappa_std": float(kappas.std()),
    }


def train_domain(
    csv_path: str,
    prefix: str,
    n_splits: int = 5,
    svm_max_samples: int = SVM_MAX_SAMPLES,
    models_dir: str = "models",
    output_dir: str = "outputs",
    rf_n_trees: int = 300,
    rf_max_depth: int = 25,
    rf_min_samples_leaf: int = 1,
    filter_regions: list[str] | None = None,
    group_strategy: str = "auto",
):
    """Treina RF/SVM de um domínio e salva modelos + métricas."""
    _df, feature_cols, X, y, groups = load_domain_samples(
        csv_path,
        filter_regions=filter_regions,
        group_strategy=group_strategy,
    )
    print(f"Total de amostras: {len(X)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Classes: {dict(pd.Series(y).value_counts().sort_index())}")

    metrics = {
        "validation": "leave_one_group_out",
        "n_samples": len(X),
        "features": feature_cols,
    }

    metrics[f"{prefix}_rf"] = _evaluate_one(
        "Random Forest",
        X,
        y,
        groups,
        n_splits,
        output_dir,
        rf_n_trees=rf_n_trees,
        rf_max_depth=rf_max_depth,
        rf_min_samples_leaf=rf_min_samples_leaf,
    )

    X_svm, y_svm, groups_svm = _cap_samples(
        X, y, groups, svm_max_samples
    )
    print(f"\nSVM limitado a {len(X_svm)} amostras (de {len(X)}).")
    metrics[f"{prefix}_svm"] = _evaluate_one(
        "SVM", X_svm, y_svm, groups_svm, n_splits, output_dir
    )

    # Modelos finais
    print("\nTreinando modelos finais...")
    final_rf = train_random_forest(
        X,
        y,
        n_trees=rf_n_trees,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        verbose=0,
    )
    final_svm = train_svm(
        X_svm, y_svm, verbose=False, probability=True
    )

    os.makedirs(models_dir, exist_ok=True)
    save_rf(final_rf, os.path.join(models_dir, f"{prefix}_rf.pkl"))
    save_svm(final_svm, os.path.join(models_dir, f"{prefix}_svm.pkl"))

    plot_feature_importance(
        final_rf.feature_importances_,
        feature_cols,
        save_path=os.path.join(output_dir, f"{prefix}_rf_feature_importance.png"),
    )

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"metrics_{prefix}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas salvas em {metrics_path}")
    return metrics
