"""Avaliação em raster de SP: acurácia e ganho do pós-processamento."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.classifiers.random_forest import train_random_forest
from src.evaluation.metrics import full_evaluation
from src.evaluation.training import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.geo.postprocess import majority_filter
from src.processing.local_features import add_local_indices_array


def main():
    parser = argparse.ArgumentParser(description="Avalia raster SP e pós-processamento")
    parser.add_argument("--csv", default="data/samples/full/brazil_samples.csv")
    parser.add_argument("--holdout-region", default="piracicaba")
    parser.add_argument(
        "--tile-dir", default="data/processed/unet_full"
    )
    parser.add_argument("--output-dir", default="outputs/v2/sp")
    parser.add_argument("--no-local-indices", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(os.path.join(PROJECT_ROOT, args.csv))
    sp_regions = ["piracicaba", "ribeirao_preto", "barretos"]
    sp = df[df["region_id"].isin(sp_regions)]
    train = sp[sp["region_id"] != args.holdout_region]

    feature_cols = (
        BASE_FEATURE_COLUMNS if args.no_local_indices else FEATURE_COLUMNS
    )
    if not args.no_local_indices:
        from src.processing.local_features import add_local_indices

        train = add_local_indices(train)

    X_train = train[list(feature_cols)].to_numpy(np.float32)
    y_train = train["class_id"].to_numpy(np.int64)

    model = train_random_forest(
        X_train, y_train, n_trees=300, max_depth=25, verbose=0
    )

    image_path = os.path.join(
        PROJECT_ROOT, args.tile_dir, f"unet_{args.holdout_region}_t0_image.npy"
    )
    labels_path = os.path.join(
        PROJECT_ROOT, args.tile_dir, f"unet_{args.holdout_region}_t0_labels.npy"
    )
    image = np.load(image_path)
    labels = np.load(labels_path)
    if not args.no_local_indices:
        image = add_local_indices_array(image)

    prediction = model.predict(image.reshape(-1, image.shape[-1]))
    prediction = prediction.reshape(labels.shape).astype(np.uint8)
    valid = labels != 255

    before = full_evaluation(labels[valid], prediction[valid])
    smoothed = majority_filter(
        prediction, valid, kernel_size=3, num_classes=6
    )
    after = full_evaluation(labels[valid], smoothed[valid])

    report = {
        "holdout_region": args.holdout_region,
        "train_regions": sorted(train["region_id"].unique().tolist()),
        "n_train": len(train),
        "tile": image_path,
        "before": before,
        "after": after,
        "gain_accuracy_pp": (after["overall_accuracy"] - before["overall_accuracy"]) * 100,
        "gain_kappa": after["kappa"] - before["kappa"],
    }

    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raster_postprocess_eval.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Acurácia antes:  {before['overall_accuracy']:.4f} | Kappa {before['kappa']:.4f}")
    print(f"Acurácia depois: {after['overall_accuracy']:.4f} | Kappa {after['kappa']:.4f}")
    print(f"Ganho: {report['gain_accuracy_pp']:.2f} pp | Kappa {report['gain_kappa']:.4f}")
    print(f"Relatório: {output_path}")


if __name__ == "__main__":
    main()
