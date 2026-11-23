"""Sanitização e validação dos tiles da U-Net."""

import json
import os

import numpy as np

from src.config import IGNORE_LABEL, NUM_CLASSES

FEATURE_NAMES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "contrast",
    "dissimilarity",
]
REFLECTANCE_BANDS = {"B2", "B3", "B4", "B8", "B11", "B12"}
TEXTURE_BANDS = {"contrast", "dissimilarity"}


def sanitize_features(
    image: np.ndarray,
    band_names: list[str] | None = None,
    reflectance_range: tuple[float, float] = (-0.5, 1.5),
    texture_max: float = 1000.0,
) -> np.ndarray:
    """Remove NaN/Inf e limita cada banda a faixas físicas."""
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    names = band_names or FEATURE_NAMES

    for index, name in enumerate(names[: image.shape[-1]]):
        band = image[..., index]
        if name == "NDVI":
            band = np.clip(band, -1.0, 1.0)
        elif name in REFLECTANCE_BANDS:
            band = np.clip(band, *reflectance_range)
        elif name in TEXTURE_BANDS:
            band = np.clip(band, 0.0, texture_max)
        image[..., index] = band

    return image


def sanitize_labels(
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_label: int = IGNORE_LABEL,
) -> np.ndarray:
    """Mantém apenas classes 0..N-1; todo o resto vira ignore_label."""
    labels = np.asarray(labels)
    labels = np.nan_to_num(labels, nan=ignore_label, posinf=ignore_label, neginf=ignore_label)
    labels = labels.astype(np.int64)
    return np.where(
        (labels >= 0) & (labels < num_classes), labels, ignore_label
    )


def analyze_tile(
    image: np.ndarray,
    labels: np.ndarray,
    min_valid_fraction: float = 0.01,
    min_classes: int = 2,
) -> dict:
    """Resumo de qualidade de um tile e decisão de validade."""
    image = np.asarray(image)
    labels = np.asarray(labels)

    unique, counts = np.unique(labels, return_counts=True)
    distribution = {
        int(value): int(count) for value, count in zip(unique, counts)
    }
    valid_labels = labels[(labels >= 0) & (labels < NUM_CLASSES)]
    valid_fraction = float(valid_labels.size / labels.size) if labels.size else 0.0
    classes = sorted(int(value) for value in np.unique(valid_labels))

    band_stats = []
    for index, name in enumerate(FEATURE_NAMES[: image.shape[-1]]):
        band = image[..., index]
        band_stats.append(
            {
                "band": name,
                "min": float(np.nanmin(band)),
                "max": float(np.nanmax(band)),
                "outliers": int(
                    np.sum(
                        ~np.isfinite(band)
                        | (np.abs(band) > 1e6)
                    )
                ),
            }
        )

    valid = (
        image.shape[:2] == labels.shape
        and bool(np.isfinite(image).all())
        and valid_fraction >= min_valid_fraction
        and len(classes) >= min_classes
    )

    return {
        "image_shape": list(image.shape),
        "labels_shape": list(labels.shape),
        "finite": bool(np.isfinite(image).all()),
        "band_stats": band_stats,
        "label_distribution": distribution,
        "valid_fraction": round(valid_fraction, 4),
        "classes": classes,
        "valid": valid,
    }


def repair_tile(
    image: np.ndarray,
    labels: np.ndarray,
    band_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sanitiza imagem/labels e devolve o relatório do tile reparado."""
    image = sanitize_features(image, band_names=band_names)
    labels = sanitize_labels(labels)
    report = analyze_tile(image, labels)
    return image, labels, report


def patch_coverage(
    labels: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    ignore_label: int = IGNORE_LABEL,
) -> dict:
    """Conta quantos patches da grade de treino contêm cada classe.

    Mede o desbalanceamento no nível em que a U-Net realmente aprende: um
    patch só contribui para uma classe se ela estiver presente nele.
    """
    labels = np.asarray(labels)
    height, width = labels.shape

    total = 0
    counts = {class_id: 0 for class_id in range(NUM_CLASSES)}
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = labels[y : y + patch_size, x : x + patch_size]
            present = np.unique(patch[patch != ignore_label])
            if present.size == 0:
                continue
            total += 1
            for value in present:
                if 0 <= int(value) < NUM_CLASSES:
                    counts[int(value)] += 1

    fractions = {
        class_id: round(count / total, 4) if total else 0.0
        for class_id, count in counts.items()
    }
    return {
        "patch_size": patch_size,
        "stride": stride,
        "total_patches": total,
        "patches_with_class": counts,
        "fraction": fractions,
    }


def write_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
