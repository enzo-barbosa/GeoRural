"""Classificador de tipo de cultivo baseado em série temporal NDVI/EVI.

A rotulagem de referência é construída por perfis fenológicos por cultura
(regras), e o Random Forest "destila" esses perfis em um modelo
supervisionado que opera sobre a série temporal de cada pixel.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

CROP_NAMES = [
    "Cana-de-açúcar",
    "Citros",
    "Pastagem",
    "Soja",
    "Milho",
]

# NDVI de referência por mês (Jan..Dez) para o hemisfério sul.
CROP_PROFILES_NDVI = {
    "Cana-de-açúcar": [
        0.80, 0.82, 0.80, 0.72, 0.65, 0.60,
        0.58, 0.60, 0.65, 0.72, 0.78, 0.80,
    ],
    "Citros": [
        0.72, 0.74, 0.75, 0.73, 0.70, 0.68,
        0.67, 0.68, 0.70, 0.72, 0.73, 0.72,
    ],
    "Pastagem": [
        0.55, 0.58, 0.60, 0.55, 0.48, 0.42,
        0.38, 0.35, 0.38, 0.45, 0.50, 0.53,
    ],
    "Soja": [
        0.85, 0.88, 0.80, 0.55, 0.35, 0.25,
        0.22, 0.22, 0.25, 0.45, 0.70, 0.82,
    ],
    "Milho": [
        0.80, 0.85, 0.82, 0.70, 0.55, 0.40,
        0.30, 0.25, 0.25, 0.30, 0.55, 0.72,
    ],
}


def crop_feature_names(num_months: int = 12) -> list[str]:
    """Nomes das features (NDVI/EVI por mês) na ordem esperada."""
    names = []
    for i in range(num_months):
        names.append(f"NDVI_{i:02d}")
        names.append(f"EVI_{i:02d}")
    return names


def _profile_to_features(ndvi_profile: np.ndarray) -> np.ndarray:
    """Converte um perfil NDVI mensal em features intercaladas NDVI/EVI."""
    evi_profile = np.clip(0.85 * ndvi_profile + 0.08, -1.0, 1.0)
    features = np.empty(ndvi_profile.size * 2, dtype=np.float32)
    features[0::2] = ndvi_profile
    features[1::2] = evi_profile
    return features


def build_labeled_dataset(
    samples_per_crop: int = 400, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Gera dataset sintético a partir dos perfis fenológicos.

    Adiciona ruído gaussiano e variação de escala para tornar o modelo
    robusto a pequenas diferenças de vigor entre talhões.
    """
    rng = np.random.default_rng(seed)
    features = []
    labels = []

    for label, crop in enumerate(CROP_NAMES):
        base = _profile_to_features(np.array(CROP_PROFILES_NDVI[crop]))
        for _ in range(samples_per_crop):
            scale = rng.uniform(0.90, 1.10)
            noise = rng.normal(0.0, 0.035, size=base.size)
            sample = np.clip(base * scale + noise, -1.0, 1.0).astype(np.float32)
            features.append(sample)
            labels.append(label)

    return np.vstack(features), np.array(labels, dtype=np.int64)


def train_crop_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    n_trees: int = 300,
    max_depth: int = 18,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Treina o Random Forest de tipo de cultivo e retorna métricas."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test,
        predictions,
        target_names=CROP_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return model, {"accuracy": accuracy, "report": report}


def predict_crop_map(model, features: np.ndarray) -> np.ndarray:
    """Classifica um raster de features (H, W, n_features) em culturas."""
    height, width = features.shape[:2]
    flat = features.reshape(-1, features.shape[-1])
    predictions = model.predict(flat)
    return predictions.reshape(height, width).astype(np.uint8)


def save_model(model, filepath: str) -> None:
    """Salva o modelo treinado em disco."""
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Carrega o modelo treinado do disco."""
    return joblib.load(filepath)
