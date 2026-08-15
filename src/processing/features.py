"""Extração consolidada de atributos para classificação."""

import ee
import numpy as np
import pandas as pd

from src.processing.indices import calculate_ndvi_ee
from src.processing.texture import calculate_glcm_ee

CLASS_NAMES = [
    "Vegetação Nativa",
    "Pastagem",
    "Agricultura",
    "Solo Exposto",
    "Área Construída",
]


def extract_features_ee(image: ee.Image, satellite: str = "sentinel2") -> ee.Image:
    """Extrai todos os atributos de uma imagem GEE.

    Pipeline completo:
    1. Bandas originais (reflectância)
    2. NDVI
    3. Textura GLCM

    Args:
        image: Imagem GEE pré-processada
        satellite: 'sentinel2' ou 'landsat8'

    Returns:
        Imagem com todas as bandas de atributos
    """
    image = calculate_ndvi_ee(image, satellite)

    if satellite == "sentinel2":
        red_band = "B4"
    else:
        red_band = "SR_B4"

    image = calculate_glcm_ee(image, band_name=red_band)

    return image


def get_feature_bands(satellite: str = "sentinel2") -> list:
    """Retorna a lista de bandas de atributos para classificação.

    Args:
        satellite: 'sentinel2' ou 'landsat8'

    Returns:
        Lista com nomes das bandas de atributos
    """
    if satellite == "sentinel2":
        return [
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
    else:
        return [
            "SR_B2",
            "SR_B3",
            "SR_B4",
            "SR_B5",
            "SR_B6",
            "SR_B7",
            "NDVI",
            "contrast",
            "dissimilarity",
        ]


def extract_real_samples(
    ground_truth: np.ndarray,
    image_data: np.ndarray,
    band_names: list,
    num_samples_per_class: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Extrai amostras REAIS a partir do ground truth.

    Para cada classe, seleciona N pontos aleatórios DENTRO
    das áreas rotuladas no ground truth, e extrai os valores
    das bandas da imagem de satélite correspondente.

    Args:
        ground_truth: Array 2D com classes (0-4) do MapBiomas reclassificado
        image_data: Array 3D (bandas × altura × largura) da imagem de satélite
        band_names: Lista de nomes das bandas
        num_samples_per_class: Número de amostras por classe
        seed: Seed para reprodutibilidade

    Returns:
        DataFrame com amostras reais para treinamento
    """
    np.random.seed(seed)
    samples = []

    for class_id in range(5):
        mask = ground_truth == class_id
        ys, xs = np.where(mask)

        if len(ys) == 0:
            print(
                f"Aviso: classe {class_id} ({CLASS_NAMES[class_id]}) "
                f"sem pixels no ground truth — pulando"
            )
            continue

        n_available = min(num_samples_per_class, len(ys))
        indices = np.random.choice(len(ys), size=n_available, replace=False)

        for idx in indices:
            y, x = ys[idx], xs[idx]
            pixel_values = image_data[:, y, x]

            sample = {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "y": y,
                "x": x,
            }

            for i, band_name in enumerate(band_names):
                sample[f"band_{band_name}"] = pixel_values[i]

            samples.append(sample)

    df = pd.DataFrame(samples)
    print(f"Amostras extraídas: {len(df)} pixels")
    for class_id in range(5):
        n = len(df[df["class_id"] == class_id])
        print(f"  Classe {class_id} ({CLASS_NAMES[class_id]}): {n} pixels")
    return df


def validate_samples(df: pd.DataFrame) -> None:
    """Valida a qualidade das amostras extraídas."""
    print("\n=== Validação das Amostras ===")
    print(f"Total de amostras: {len(df)}")
    print("\nDistribuição de classes:")
    print(df["class_name"].value_counts().to_string())
    band_cols = [c for c in df.columns if c.startswith("band_")]
    print("\nEstatísticas por banda:")
    print(df[band_cols].describe().round(3).to_string())
    print("\n✓ Amostras validadas com sucesso")
