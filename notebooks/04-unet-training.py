"""Treino da U-Net (Brasil, 6 classes).

Prefere tiles multi-bioma em ``data/processed/unet/`` (leave-one-tile-out).
Se não existirem, usa os rasters únicos ``data/processed/unet_image.npy`` com
split espacial interno.

Para treinar na GPU, use ``notebooks/04-unet-training-colab.ipynb``.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from src.classifiers.unet_cnn import (
    build_unet,
    leave_one_tile_out,
    load_unet_tiles,
    spatial_patch_split,
    train_unet,
)
from src.config import IGNORE_LABEL, NUM_CLASSES
from src.processing.features import get_feature_bands

TILES_DIR = os.path.join(PROJECT_ROOT, "data/processed/unet")
VAL_TILE = os.environ.get("UNET_VAL_TILE", "unet_piracicaba")


def class_weights_from(labels: np.ndarray) -> np.ndarray:
    valid = labels != IGNORE_LABEL
    counts = np.bincount(
        labels[valid].astype(int).ravel(), minlength=NUM_CLASSES
    )[:NUM_CLASSES].astype(float)
    counts[counts == 0] = 1.0
    return counts.sum() / (counts * len(counts))


def main():
    print(f"Classes ({NUM_CLASSES}): {get_feature_bands('sentinel2')}")
    print(f"Label de ignore: {IGNORE_LABEL}")

    if os.path.isdir(TILES_DIR) and os.listdir(TILES_DIR):
        tiles = load_unet_tiles(TILES_DIR)
        print(f"Tiles encontrados: {list(tiles)}")

        val_tile = VAL_TILE if VAL_TILE in tiles else next(iter(tiles))
        print(f"Validação: {val_tile} (leave-one-tile-out)")

        X_train, y_train, X_val, y_val = leave_one_tile_out(
            tiles, val_tile, patch_size=64, stride=32
        )
    else:
        image_path = os.path.join(PROJECT_ROOT, "data/processed/unet_image.npy")
        labels_path = os.path.join(PROJECT_ROOT, "data/processed/unet_labels.npy")
        if not os.path.exists(image_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                "Rasters da U-Net não encontrados. Rode:\n"
                "  python src/gee/build_unet_dataset.py --regions all "
                "--year 2023 --start 2023-01-01 --end 2023-12-31"
            )

        image = np.load(image_path)
        labels = np.load(labels_path)
        print(f"Imagem: {image.shape} | labels: {labels.shape}")
        X_train, y_train, X_val, y_val = spatial_patch_split(
            image, labels, patch_size=64, stride=32, tiles=3, val_tiles=1
        )

    print(f"Treino: {X_train.shape} | Validação: {X_val.shape}")

    class_weights = class_weights_from(y_train)
    print(f"Pesos de classe: {np.round(class_weights, 4).tolist()}")

    model = build_unet(
        input_shape=(64, 64, X_train.shape[-1]), num_classes=NUM_CLASSES
    )
    history = train_unet(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        batch_size=16,
        learning_rate=0.001,
        save_path=os.path.join(PROJECT_ROOT, "models/brazil_unet.pth"),
        class_weights=class_weights,
        ignore_index=IGNORE_LABEL,
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, "outputs/unet_training_curves.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Curvas salvas em {out_path}")


if __name__ == "__main__":
    main()
