"""Treino e avaliação do classificador de tipo de cultivo (série NDVI/EVI)."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.classifiers.crop_classifier import (
    CROP_NAMES,
    build_labeled_dataset,
    save_model,
    train_crop_classifier,
)


def main():
    print("=" * 60)
    print("CLASSIFICADOR DE TIPO DE CULTIVO")
    print("=" * 60)
    print("Culturas:", ", ".join(CROP_NAMES))

    X, y = build_labeled_dataset(samples_per_crop=400, seed=42)
    print(f"Dataset sintético (perfis fenológicos): {X.shape[0]} amostras")
    print(f"Features por amostra: {X.shape[1]} (NDVI/EVI x 12 meses)")

    model, metrics = train_crop_classifier(X, y)

    print(f"\nAcurácia: {metrics['accuracy']:.4f}")
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
    print("-" * 60)

    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "models/crop_classifier.pkl")
    save_model(model, save_path)
    print(f"\nModelo salvo em: {save_path}")


if __name__ == "__main__":
    main()
