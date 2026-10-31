"""Configurações do projeto."""

# Ordem unificada (bate com src/gee/ground_truth.py e src/processing/features.py):
# 0: Vegetação Nativa, 1: Pastagem, 2: Agricultura,
# 3: Solo Exposto, 4: Área Construída, 5: Água
CLASS_NAMES = [
    "Vegetação Nativa",
    "Pastagem",
    "Agricultura",
    "Solo Exposto",
    "Área Construída",
    "Água",
]

# Fora do Brasil o rótulo "Pastagem" cobre grassland (pasto manejado ou campo
# natural), pois WorldCover/Dynamic World não separam essas classes.
CLASS_NAMES_GLOBAL = [
    "Vegetação Nativa",
    "Pastagem / Veg. Herbácea",
    "Agricultura",
    "Solo Exposto",
    "Área Construída",
    "Água",
]

CLASS_IDS = {
    "vegetação_nativa": 0,
    "pastagem": 1,
    "agricultura": 2,
    "solo_exposto": 3,
    "área_construída": 4,
    "água": 5,
}

CLASS_COLORS = {
    "Vegetação Nativa": "#1a5276",
    "Pastagem": "#27ae60",
    "Pastagem / Veg. Herbácea": "#27ae60",
    "Agricultura": "#2ecc71",
    "Solo Exposto": "#f39c12",
    "Área Construída": "#e74c3c",
    "Água": "#2e86de",
}

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_LABEL = 255

CROP_NAMES = [
    "Cana-de-açúcar",
    "Citros",
    "Pastagem",
    "Soja",
    "Milho",
]

CROP_COLORS = {
    "Cana-de-açúcar": "#8e44ad",
    "Citros": "#f1c40f",
    "Pastagem": "#27ae60",
    "Soja": "#16a085",
    "Milho": "#e67e22",
}

# Classes de declividade (EMBRAPA)
SLOPE_CLASS_NAMES = [
    "Plano",
    "Suave ondulado",
    "Ondulado",
    "Forte ondulado",
    "Montanhoso",
    "Escarpado",
]

SLOPE_CLASS_COLORS = [
    "#2ecc71",
    "#a3d977",
    "#f4d03f",
    "#f39c12",
    "#e67e22",
    "#e74c3c",
]

RF_PARAMS = {"n_estimators": 200, "max_depth": 15, "random_state": 42}

SVM_PARAMS = {"kernel": "rbf", "C": 100, "gamma": "scale"}

UNET_PARAMS = {
    "input_shape": (64, 64, 9),  # 9 canais = 6 bandas + NDVI + 2 texturas
    "num_classes": NUM_CLASSES,
    "epochs": 100,
    "learning_rate": 0.001,
    "batch_size": 16,
}

SENTINEL2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
LANDSAT8_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
