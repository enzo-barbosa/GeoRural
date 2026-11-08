"""Valida um único tile da U-Net (imagem + labels)."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.processing.dataset_validation import analyze_tile


def main():
    parser = argparse.ArgumentParser(description="Valida um tile da U-Net")
    parser.add_argument("--dir", default="data/processed/unet_full")
    parser.add_argument("--tile", default="unet_piracicaba_t0")
    parser.add_argument("--min-valid-fraction", type=float, default=0.01)
    parser.add_argument("--min-classes", type=int, default=2)
    args = parser.parse_args()

    directory = os.path.join(PROJECT_ROOT, args.dir)
    image_path = os.path.join(directory, f"{args.tile}_image.npy")
    labels_path = os.path.join(directory, f"{args.tile}_labels.npy")

    image = np.load(image_path)
    labels = np.load(labels_path)
    report = analyze_tile(
        image,
        labels,
        min_valid_fraction=args.min_valid_fraction,
        min_classes=args.min_classes,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
