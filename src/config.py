"""Configurações do projeto."""

# Ordem unificada (bate com src/gee/ground_truth.py e src/processing/features.py):
# 0: Vegetação Nativa, 1: Pastagem, 2: Agricultura, 3: Solo Exposto, 4: Área Construída
CLASS_NAMES = [
    "Vegetação Nativa",
    "Pastagem",
    "Agricultura",
    "Solo Exposto",
    "Área Construída",
]

CLASS_IDS = {
    "vegetação_nativa": 0,
    "pastagem": 1,
    "agricultura": 2,
    "solo_exposto": 3,
    "área_construída": 4,
}

CLASS_COLORS = {
    "Vegetação Nativa": "#1a5276",
    "Pastagem": "#27ae60",
    "Agricultura": "#2ecc71",
    "Solo Exposto": "#f39c12",
    "Área Construída": "#e74c3c",
}

RF_PARAMS = {"n_estimators": 200, "max_depth": 15, "random_state": 42}

SVM_PARAMS = {"kernel": "rbf", "C": 100, "gamma": "scale"}

UNET_PARAMS = {
    "input_shape": (64, 64, 10),
    "num_classes": 5,
    "epochs": 100,
    "learning_rate": 0.001,
    "batch_size": 16,
}

SENTINEL2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
LANDSAT8_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
