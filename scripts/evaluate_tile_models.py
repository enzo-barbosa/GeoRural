"""Compara RF, SVM e U-Net (e ensemble) em um tile da U-Net."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.classifiers.ensemble import soft_vote
from src.classifiers.unet_cnn import predict_unet_proba
from src.config import IGNORE_LABEL, NUM_CLASSES
from src.evaluation.metrics import full_evaluation
from src.geo.postprocess import majority_filter
from src.processing.local_features import add_local_indices_array


def _proba(model, image):
    features = image.reshape(-1, image.shape[-1])
    n_features = getattr(model, "n_features_in_", features.shape[1])
    proba = model.predict_proba(features[:, :n_features])
    aligned = np.zeros((len(features), NUM_CLASSES), dtype=np.float32)
    for column, class_id in enumerate(model.classes_):
        aligned[:, int(class_id)] = proba[:, column]
    return aligned.reshape(*image.shape[:2], NUM_CLASSES)


def _evaluate(name, prediction, labels):
    valid = labels != IGNORE_LABEL
    metrics = full_evaluation(labels[valid], prediction[valid].astype(np.uint8))
    return {
        "name": name,
        "accuracy": round(metrics["overall_accuracy"], 4),
        "kappa": round(metrics["kappa"], 4),
        "macro_f1": round(
            float(np.mean(list(metrics["f1_score"].values()))), 4
        ),
        "f1": {key: round(value, 4) for key, value in metrics["f1_score"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Compara modelos em um tile")
    parser.add_argument("--tile", default="unet_piracicaba_t0")
    parser.add_argument("--tile-dir", default="data/processed/unet_full")
    parser.add_argument("--output", default="outputs/tile_model_comparison.json")
    args = parser.parse_args()

    from backend.app.dependencies.models import get_cached_model

    image = np.load(
        os.path.join(PROJECT_ROOT, args.tile_dir, f"{args.tile}_image.npy")
    )
    labels = np.load(
        os.path.join(PROJECT_ROOT, args.tile_dir, f"{args.tile}_labels.npy")
    )
    image = add_local_indices_array(image)

    probabilities = {}
    results = []

    for name in ("brazil_rf", "brazil_svm", "sp_rf", "sp_svm"):
        model = get_cached_model(name)
        proba = _proba(model, image)
        probabilities[name] = proba
        prediction = proba.argmax(axis=-1)
        results.append(_evaluate(name, prediction, labels))
        smoothed = majority_filter(prediction, labels != IGNORE_LABEL)
        results.append(_evaluate(f"{name}+3x3", smoothed, labels))

    unet = get_cached_model("brazil_unet")
    unet_proba = predict_unet_proba(
        unet, image, patch_size=64, stride=32, num_classes=NUM_CLASSES
    )
    probabilities["brazil_unet"] = unet_proba
    unet_pred = unet_proba.argmax(axis=-1)
    results.append(_evaluate("brazil_unet", unet_pred, labels))
    results.append(
        _evaluate(
            "brazil_unet+3x3",
            majority_filter(unet_pred, labels != IGNORE_LABEL),
            labels,
        )
    )

    for label, names in (
        ("ensemble_brazil", ["brazil_rf", "brazil_svm", "brazil_unet"]),
        ("ensemble_sp", ["sp_rf", "sp_svm", "brazil_unet"]),
    ):
        combined = soft_vote({name: probabilities[name] for name in names})
        prediction = combined.argmax(axis=-1)
        results.append(_evaluate(label, prediction, labels))
        results.append(
            _evaluate(
                f"{label}+3x3",
                majority_filter(prediction, labels != IGNORE_LABEL),
                labels,
            )
        )

        for unet_weight in (0.4, 0.5, 0.6, 0.7):
            remainder = (1.0 - unet_weight) / (len(names) - 1)
            weights = {
                name: (unet_weight if name == "brazil_unet" else remainder)
                for name in names
            }
            weighted = soft_vote(
                {name: probabilities[name] for name in names}, weights
            )
            prediction = weighted.argmax(axis=-1)
            results.append(
                _evaluate(
                    f"{label}_unet{unet_weight:.1f}",
                    prediction,
                    labels,
                )
            )
            results.append(
                _evaluate(
                    f"{label}_unet{unet_weight:.1f}+3x3",
                    majority_filter(prediction, labels != IGNORE_LABEL),
                    labels,
                )
            )

    os.makedirs(os.path.dirname(os.path.join(PROJECT_ROOT, args.output)), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, args.output), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    for item in results:
        print(
            f"{item['name']:<24} acc={item['accuracy']:.4f} "
            f"kappa={item['kappa']:.4f} macroF1={item['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
