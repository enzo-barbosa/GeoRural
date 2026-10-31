"""Treino do classificador de cultivo.

Se ``data/samples/crop_samples.csv`` existir, usa rótulos reais do MapBiomas
(20 Cana, 47 Citros, 15 Pastagem, 39 Soja, 41 Milho aproximado) com validação
espacial por bioma. Caso contrário, cai no dataset sintético de perfis.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.classifiers.crop_classifier import (
    CROP_NAMES,
    build_labeled_dataset,
    save_model,
    train_crop_classifier,
)
from src.evaluation.spatial_split import (
    assign_spatial_blocks,
    spatial_kfold_indices,
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("NDVI_", "EVI_"))]


def train_with_real_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_cols = _feature_columns(df)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["crop_id"].to_numpy(dtype=np.int64)

    if "region_id" in df.columns and df["region_id"].nunique() >= 2:
        groups = pd.factorize(df["region_id"].astype(str))[0]
    elif "biome" in df.columns and df["biome"].nunique() >= 2:
        groups = pd.factorize(df["biome"].astype(str))[0]
    elif {"longitude", "latitude"}.issubset(df.columns):
        groups = assign_spatial_blocks(df["longitude"], df["latitude"])
    else:
        groups = np.arange(len(df), dtype=np.int64)

    n_splits = min(5, len(np.unique(groups)))
    splits = spatial_kfold_indices(groups, n_splits=n_splits)

    print(f"Amostras reais: {len(X)} | features: {len(feature_cols)}")
    print(f"Distribuição: {dict(pd.Series(y).value_counts().sort_index())}")

    accs = []
    oof_true = np.full(len(y), -1, dtype=y.dtype)
    oof_pred = np.full(len(y), -1, dtype=y.dtype)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        oof_true[test_idx] = y[test_idx]
        oof_pred[test_idx] = pred
        accuracy = accuracy_score(y[test_idx], pred)
        accs.append(accuracy)
        print(f"  Fold {fold}: acurácia={accuracy:.4f}")

    mask = oof_true >= 0
    report = classification_report(
        oof_true[mask],
        oof_pred[mask],
        labels=list(range(len(CROP_NAMES))),
        target_names=CROP_NAMES,
        output_dict=True,
        zero_division=0,
    )
    print(f"\nAcurácia espacial: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    final = RandomForestClassifier(
        n_estimators=300, max_depth=18, random_state=42, n_jobs=-1
    )
    final.fit(X, y)

    metrics = {
        "validation": "spatial_kfold",
        "n_samples": len(X),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "report": report,
    }
    return final, metrics


def main():
    print("=" * 60)
    print("CLASSIFICADOR DE TIPO DE CULTIVO")
    print("=" * 60)
    print("Culturas:", ", ".join(CROP_NAMES))

    real_csv = os.path.join(PROJECT_ROOT, "data/samples/crop_samples.csv")
    if os.path.exists(real_csv):
        print(f"Fonte: rótulos MapBiomas ({real_csv})")
        model, metrics = train_with_real_labels(real_csv)
    else:
        print("Fonte: dataset sintético (perfis fenológicos)")
        X, y = build_labeled_dataset(samples_per_crop=400, seed=42)
        print(f"Dataset sintético: {X.shape[0]} amostras")
        model, metrics = train_crop_classifier(X, y)

    print(f"\nAcurácia: {metrics['accuracy'] if 'accuracy' in metrics else metrics['accuracy_mean']:.4f}")
    report = metrics["report"]
    print("\nMétricas por cultura:")
    print("-" * 60)
    print(f"{'Cultura':<18} {'Precisão':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for name in CROP_NAMES:
        row = report.get(name, {})
        print(
            f"{name:<18} "
            f"{row.get('precision', 0):>10.4f} "
            f"{row.get('recall', 0):>10.4f} "
            f"{row.get('f1-score', 0):>10.4f}"
        )

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "models/crop_classifier.pkl")
    save_model(model, save_path)
    print(f"\nModelo salvo em: {save_path}")

    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    metrics_path = os.path.join(PROJECT_ROOT, "outputs/crop_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"Métricas salvas em: {metrics_path}")


if __name__ == "__main__":
    main()
