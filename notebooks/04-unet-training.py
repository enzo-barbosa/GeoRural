# %% [markdown]
# # Treinamento da U-Net (PyTorch)
#
# Este script executa:
# 1. Carga de imagem + ground truth (`data/processed/`)
# 2. Extração de patches 64x64
# 3. Treino da U-Net
# 4. Avaliação qualitativa e curvas de aprendizado
#
# Rode localmente (GPU recomendada, ex.: RTX 5070):
#     python notebooks/04-unet-training.py

# %% [markdown]
# ## 0. Gerar os rasters (uma única vez, na máquina de origem)
#
#     python src/gee/build_unet_dataset.py --satellite sentinel2 \
#         --start 2023-01-01 --end 2023-12-31 --max-cloud 80
#
# Isso cria `data/processed/unet_image.npy` e `data/processed/unet_labels.npy`
# e `data/processed/meta.json`. Se recebeu o `unet_dataset.zip`, basta extraí-lo
# dentro de `data/processed/`.

# %%
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from src.classifiers.unet_cnn import build_unet, spatial_patch_split, train_unet
from src.processing.features import get_feature_bands

# %%
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

image_path = "data/processed/unet_image.npy"
labels_path = "data/processed/unet_labels.npy"

if not os.path.exists(image_path) or not os.path.exists(labels_path):
    raise FileNotFoundError(
        "Rasters da U-Net não encontrados em data/processed/.\n"
        "Rode antes: python src/gee/build_unet_dataset.py --satellite sentinel2 "
        "ou extraia o unet_dataset.zip em data/processed/."
    )

image = np.load(image_path)
labels = np.load(labels_path)

print(f"Imagem carregada: {image.shape} dtype={image.dtype}")
print(f"Ground truth carregado: {labels.shape} dtype={labels.dtype}")
assert image.ndim == 3 and labels.ndim == 2, "Formato inesperado dos rasters."
assert image.shape[:2] == labels.shape, "Imagem e ground truth com dimensões diferentes."

# %% [markdown]
# ## 1. Verificar bandas e distribuição

# %%
print(f"Bandas: {get_feature_bands('sentinel2')}")
unique, counts = np.unique(labels, return_counts=True)
print("Distribuição de classes no ground truth:")
for cls_id, count in zip(unique, counts):
    print(f"  Classe {cls_id}: {count} pixels")

# %% [markdown]
# ## 2. Patches com split espacial
#
# Faixas espaciais disjuntas entre treino e validação evitam vazamento
# causado pela sobreposição de patches (patch 64 / stride 32).

# %%
X_train, y_train, X_val, y_val = spatial_patch_split(
    image, labels, patch_size=64, stride=32, tiles=3, val_tiles=1
)

print(f"Treino: {X_train.shape[0]} patches {X_train.shape}")
print(f"Validação: {X_val.shape[0]} patches {X_val.shape}")

# %% [markdown]
# ## 3. Pesos de classe (desbalanceamento)

# %%
counts = np.bincount(y_train.astype(int).ravel(), minlength=5).astype(float)
counts[counts == 0] = 1.0
class_weights = counts.sum() / (counts * len(counts))
print(f"Pesos de classe: {np.round(class_weights, 4).tolist()}")

# %% [markdown]
# ## 4. Treinar U-Net

# %%
N_FEATURES = image.shape[2]
model = build_unet(input_shape=(64, 64, N_FEATURES), num_classes=5)

history = train_unet(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=16,
    learning_rate=0.001,
    save_path="models/unet_model.pth",
    class_weights=class_weights,
)

# %% [markdown]
# ## 5. Curvas de aprendizado

# %%
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Época")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Época")
plt.legend()

plt.tight_layout()
plt.show()