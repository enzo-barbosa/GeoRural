"""Rede Neural Convolucional U-Net para segmentação semântica."""

import glob
import os
from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
    WeightedRandomSampler,
)

from src.config import CLASS_NAMES, IGNORE_LABEL, NUM_CLASSES
from src.processing.local_features import add_local_indices_array


class DoubleConv(nn.Module):
    """Bloco convolucional duplo: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Bloco do encoder: DoubleConv → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bloco do decoder: UpConv → Concat → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Camada de saída: Conv 1x1."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Arquitetura U-Net para segmentação semântica de imagens de satélite."""

    def __init__(
        self,
        n_channels: int = 9,
        n_classes: int = NUM_CLASSES,
        bilinear: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def build_unet(
    input_shape: tuple[int, int, int] = (64, 64, 9),
    num_classes: int = NUM_CLASSES,
):
    """Constrói a arquitetura U-Net.

    Args:
        input_shape: Formato da entrada (altura, largura, canais)
                     10 canais = 6 bandas + NDVI + 2 texturas + 1 extra
        num_classes: Número de classes de saída (5)

    Returns:
        Modelo U-Net
    """
    model = UNet(n_channels=input_shape[2], n_classes=num_classes)
    return model


def prepare_patches(
    image: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
):
    """Extrai patches (janelas) da imagem para treinamento da CNN.

    A U-Net recebe patches de patch_size x patch_size como entrada.
    O stride controla a sobreposição entre patches.

    Args:
        image: Array da imagem (height, width, channels)
        labels: Array dos rótulos (height, width)
        patch_size: Tamanho do patch
        stride: Passo entre patches

    Returns:
        Tupla (patches_imagem, patches_rótulo)
    """
    patches_img = []
    patches_label = []

    h, w = image.shape[:2]

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_img = image[y : y + patch_size, x : x + patch_size]
            patch_label = labels[y : y + patch_size, x : x + patch_size]

            patches_img.append(patch_img)
            patches_label.append(patch_label)

    if not patches_img:
        raise ValueError(
            f"Imagem ({h}x{w}) menor que patch_size ({patch_size}). "
            "Aumente a imagem ou reduza o patch_size."
        )

    patches_img = np.array(patches_img)
    patches_label = np.array(patches_label)

    print(f"Gerados {len(patches_img)} patches de {patch_size}x{patch_size}")

    return patches_img, patches_label


def spatial_patch_split(
    image: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    tiles: int = 3,
    val_tiles: int = 1,
    include_indices: bool = False,
):
    """Separa patches em treino/validação por faixas espaciais disjuntas.

    As faixas (tiles) são bandas horizontais do raster. A validação usa as
    últimas ``val_tiles`` faixas, garantindo que treino e validação não
    compartilhem vizinhança espacial (evita vazamento por patches sobrepostos).

    Args:
        image: Imagem (height, width, channels).
        labels: Rótulos (height, width).
        patch_size: Tamanho do patch.
        stride: Passo entre patches.
        tiles: Número de faixas horizontais.
        val_tiles: Número de faixas reservadas para validação.

    Returns:
        Tupla (X_train, y_train, X_val, y_val).
    """
    if include_indices:
        image = add_local_indices_array(image)
    height, width = image.shape[:2]
    ys = list(range(0, height - patch_size + 1, stride))
    xs = list(range(0, width - patch_size + 1, stride))
    if not ys or not xs:
        raise ValueError(
            f"Imagem ({height}x{width}) menor que patch_size ({patch_size})."
        )

    tiles = max(2, tiles)
    val_tiles = max(1, min(val_tiles, tiles - 1))

    train_img, train_lbl, val_img, val_lbl = [], [], [], []
    for iy, y in enumerate(ys):
        tile = min(int(iy / len(ys) * tiles), tiles - 1)
        is_val = tile >= (tiles - val_tiles)
        for x in xs:
            patch_img = image[y : y + patch_size, x : x + patch_size]
            patch_lbl = labels[y : y + patch_size, x : x + patch_size]
            if is_val:
                val_img.append(patch_img)
                val_lbl.append(patch_lbl)
            else:
                train_img.append(patch_img)
                train_lbl.append(patch_lbl)

    return (
        np.array(train_img),
        np.array(train_lbl),
        np.array(val_img),
        np.array(val_lbl),
    )


def load_unet_tiles(directory: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Carrega todos os tiles ``*_image.npy`` de um diretório."""
    tiles = {}
    for image_path in sorted(glob.glob(os.path.join(directory, "*_image.npy"))):
        name = os.path.basename(image_path).replace("_image.npy", "")
        labels_path = os.path.join(directory, f"{name}_labels.npy")
        if not os.path.exists(labels_path):
            continue
        tiles[name] = (np.load(image_path), np.load(labels_path))
    if not tiles:
        raise FileNotFoundError(f"Nenhum tile encontrado em {directory}")
    return tiles


def leave_one_tile_out(
    tiles: dict[str, tuple[np.ndarray, np.ndarray]],
    val_tile: str,
    patch_size: int = 64,
    stride: int = 32,
    include_indices: bool = False,
):
    """Monta patches de treino/validação deixando um tile inteiro de fora."""
    if val_tile not in tiles:
        raise KeyError(f"Tile de validação desconhecido: {val_tile}")

    train_img, train_lbl, val_img, val_lbl = [], [], [], []
    for name, (image, labels) in tiles.items():
        if include_indices:
            image = add_local_indices_array(image)
        patches_img, patches_lbl = prepare_patches(
            image, labels, patch_size=patch_size, stride=stride
        )
        if name == val_tile:
            val_img.append(patches_img)
            val_lbl.append(patches_lbl)
        else:
            train_img.append(patches_img)
            train_lbl.append(patches_lbl)

    return (
        np.concatenate(train_img, axis=0),
        np.concatenate(train_lbl, axis=0),
        np.concatenate(val_img, axis=0),
        np.concatenate(val_lbl, axis=0),
    )


class PatchDataset(Dataset):
    """Dataset de patches sob demanda (evita materializar 20 mil patches)."""

    def __init__(
        self,
        tiles: dict[str, tuple[np.ndarray, np.ndarray]],
        val_tile: str | Iterable[str],
        split: str = "train",
        patch_size: int = 64,
        stride: int = 32,
        include_indices: bool = True,
        min_valid_fraction: float = 0.01,
        min_classes: int = 2,
        rare_classes: tuple[int, ...] = (3, 4, 5),
        rare_patches_per_class: int = 0,
        rare_min_fraction: float = 0.01,
        seed: int = 42,
    ):
        val_tiles = {val_tile} if isinstance(val_tile, str) else set(val_tile)
        unknown = val_tiles - set(tiles)
        if unknown:
            raise KeyError(
                f"Tiles de validação desconhecidos: {sorted(unknown)}"
            )

        self.tiles = tiles
        self.val_tiles = val_tiles
        self.patch_size = patch_size
        self.include_indices = include_indices
        self.samples: list[tuple[str, int, int]] = []
        self.rare_classes = rare_classes
        self.rare_patches_per_class = rare_patches_per_class
        self.rare_min_fraction = rare_min_fraction
        self.rng = np.random.default_rng(seed)

        for name, (image, labels) in tiles.items():
            is_val = name in val_tiles
            if (split == "val") != is_val:
                continue

            valid_labels = labels[
                (labels >= 0) & (labels < NUM_CLASSES)
            ]
            fraction = (
                float(valid_labels.size / labels.size) if labels.size else 0.0
            )
            classes = np.unique(valid_labels)
            if fraction < min_valid_fraction or len(classes) < min_classes:
                if is_val:
                    raise ValueError(
                        f"Tile de validação inválido: {name} "
                        f"(fração={fraction:.3f}, "
                        f"classes={classes.tolist()})."
                    )
                print(
                    f"[skip] {name}: fração válida={fraction:.3f}, "
                    f"classes={classes.tolist()}"
                )
                continue

            height, width = image.shape[:2]
            grid = [
                (name, y, x)
                for y in range(0, height - patch_size + 1, stride)
                for x in range(0, width - patch_size + 1, stride)
            ]
            self.samples.extend(grid)

            if split == "train" and rare_patches_per_class > 0:
                existing = {(y, x) for _, y, x in grid}
                self.samples.extend(
                    self._rare_samples(
                        name, labels, height, width, existing
                    )
                )

        if not self.samples:
            raise ValueError(f"Split '{split}' sem patches.")

    def _rare_samples(
        self,
        name: str,
        labels: np.ndarray,
        height: int,
        width: int,
        existing: set[tuple[int, int]],
    ) -> list[tuple[str, int, int]]:
        """Patches centrados em pixels das classes raras."""
        if height < self.patch_size or width < self.patch_size:
            return []

        samples: list[tuple[str, int, int]] = []
        for class_id in self.rare_classes:
            coords = np.argwhere(labels == class_id)
            if coords.size == 0:
                continue
            self.rng.shuffle(coords)

            added = 0
            for center_y, center_x in coords:
                if added >= self.rare_patches_per_class:
                    break
                y = int(
                    np.clip(
                        center_y - self.patch_size // 2,
                        0,
                        height - self.patch_size,
                    )
                )
                x = int(
                    np.clip(
                        center_x - self.patch_size // 2,
                        0,
                        width - self.patch_size,
                    )
                )
                if (y, x) in existing:
                    continue
                patch = labels[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
                fraction = float((patch == class_id).mean())
                if fraction < self.rare_min_fraction:
                    continue
                existing.add((y, x))
                samples.append((name, y, x))
                added += 1
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        name, y, x = self.samples[index]
        image, labels = self.tiles[name]
        patch = image[
            y : y + self.patch_size, x : x + self.patch_size
        ]
        if self.include_indices:
            patch = add_local_indices_array(patch)
        label = labels[y : y + self.patch_size, x : x + self.patch_size]
        return (
            torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float(),
            torch.from_numpy(label.astype(np.int64)),
        )

    def class_weights(
        self, ignore_index: int = IGNORE_LABEL, exponent: float = 1.0
    ) -> np.ndarray:
        """Pesos inversos à frequência, suavizados por ``exponent``.

        ``exponent=1`` é o inverso exato; valores menores (ex.: 0,5) reduzem
        o upweight das classes raras e estabilizam o treino.
        """
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for name, _y, _x in self.samples:
            _, labels = self.tiles[name]
            patch = labels[
                _y : _y + self.patch_size, _x : _x + self.patch_size
            ]
            valid = patch != ignore_index
            counts += np.bincount(
                patch[valid].astype(int).ravel(), minlength=NUM_CLASSES
            )[:NUM_CLASSES]
        counts[counts == 0] = 1.0
        weights = (counts.sum() / (counts * len(counts))) ** exponent
        return weights / weights.mean()

    def sample_weights(self, ignore_index: int = IGNORE_LABEL) -> np.ndarray:
        """Peso por patch para WeightedRandomSampler.

        Patches sem nenhum pixel válido recebem peso ``0`` para nunca serem
        amostrados. Os pesos inversos são normalizados para média ``1`` entre
        os patches válidos, evitando a escala de ``1e-8`` que fazia o sampler
        escolher apenas patches vazios (100% ``255``).
        """
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for name, _y, _x in self.samples:
            _, labels = self.tiles[name]
            patch = labels[
                _y : _y + self.patch_size, _x : _x + self.patch_size
            ]
            valid = patch != ignore_index
            counts += np.bincount(
                patch[valid].astype(int).ravel(), minlength=NUM_CLASSES
            )[:NUM_CLASSES]
        inverse = 1.0 / (counts + 1.0)
        inverse = inverse / inverse.mean()

        weights = np.zeros(len(self.samples), dtype=np.float64)
        for index, (name, _y, _x) in enumerate(self.samples):
            _, labels = self.tiles[name]
            patch = labels[
                _y : _y + self.patch_size, _x : _x + self.patch_size
            ]
            valid = patch != ignore_index
            if valid.any():
                weights[index] = float(inverse[patch[valid]].mean())

        non_empty = weights > 0
        if non_empty.any():
            weights[non_empty] /= weights[non_empty].mean()
        return weights


def make_patch_datasets(
    tiles: dict[str, tuple[np.ndarray, np.ndarray]],
    val_tile: str | Iterable[str],
    patch_size: int = 64,
    stride: int = 32,
    include_indices: bool = True,
    min_valid_fraction: float = 0.01,
    min_classes: int = 2,
    rare_classes: tuple[int, ...] = (3, 4, 5),
    rare_patches_per_class: int = 0,
    rare_min_fraction: float = 0.01,
    seed: int = 42,
) -> tuple[PatchDataset, PatchDataset]:
    """Cria datasets de treino/validação sem carregar todos os patches.

    ``val_tile`` aceita um nome ou uma lista de tiles de validação. Para o
    treino, ``rare_patches_per_class > 0`` adiciona patches centrados nas
    classes raras (Solo Exposto, Área Construída e Água por padrão).
    """
    train = PatchDataset(
        tiles,
        val_tile,
        split="train",
        patch_size=patch_size,
        stride=stride,
        include_indices=include_indices,
        min_valid_fraction=min_valid_fraction,
        min_classes=min_classes,
        rare_classes=rare_classes,
        rare_patches_per_class=rare_patches_per_class,
        rare_min_fraction=rare_min_fraction,
        seed=seed,
    )
    val = PatchDataset(
        tiles,
        val_tile,
        split="val",
        patch_size=patch_size,
        stride=stride,
        include_indices=include_indices,
        min_valid_fraction=min_valid_fraction,
        min_classes=min_classes,
        seed=seed,
    )
    return train, val


class FocalLoss(nn.Module):
    """Focal loss multiclasse, com pesos opcionais por classe.

    Reduz o peso dos pixels fáceis e concentra o gradiente nos difíceis,
    o que ajuda classes raras sem ignorar completamente as dominantes.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = IGNORE_LABEL,
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid] = 0

        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        loss = (1.0 - pt) ** self.gamma * (-log_pt)
        if self.weight is not None:
            loss = loss * self.weight[target_safe]
        loss = loss[valid]

        return loss.mean() if loss.numel() else logits.sum() * 0.0


def compile_model(
    model: nn.Module,
    learning_rate: float = 0.001,
    device: str | None = None,
    class_weights: np.ndarray | None = None,
    ignore_index: int = IGNORE_LABEL,
    focal_gamma: float | None = None,
):
    """Compila o modelo com otimizador Adam e loss apropriada.

    Args:
        model: Modelo U-Net
        learning_rate: Taxa de aprendizado
        device: Dispositivo ('cuda' ou 'cpu')
        class_weights: Pesos por classe (para classes desbalanceadas)
        ignore_index: Rótulo ignorado (255)
        focal_gamma: Se informado, usa FocalLoss com esse gamma em vez de
            CrossEntropyLoss.

    Returns:
        Modelo compilado, optimizer, criterion
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    weights = None
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if focal_gamma is not None and focal_gamma > 0:
        criterion = FocalLoss(
            weight=weights, gamma=focal_gamma, ignore_index=ignore_index
        )
    elif weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=weights, ignore_index=ignore_index
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    print(f"Modelo compilado no dispositivo: {device}")
    print(model)

    return model, optimizer, criterion


def _sample_weights(
    labels: np.ndarray, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_LABEL
) -> np.ndarray:
    """Peso por patch para amostragem balanceada de classes raras.

    Patches sem pixels válidos recebem peso ``0`` e os pesos inversos são
    normalizados para média ``1`` entre os patches válidos.
    """
    valid = labels != ignore_index
    counts = np.bincount(
        labels[valid].astype(int).ravel(), minlength=num_classes
    )[:num_classes].astype(float)
    inverse = 1.0 / (counts + 1.0)
    inverse = inverse / inverse.mean()

    weights = np.zeros(len(labels), dtype=np.float64)
    for index, label in enumerate(labels):
        mask = label != ignore_index
        if mask.any():
            weights[index] = float(inverse[label[mask]].mean())

    non_empty = weights > 0
    if non_empty.any():
        weights[non_empty] /= weights[non_empty].mean()
    return weights


def augment_batch(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Augmentation espacial e espectral para patches de satélite.

    Args:
        x: Tensor (B, C, H, W)
        y: Tensor (B, H, W)
    """
    horizontal = torch.rand(len(x), device=x.device) < 0.5
    if horizontal.any():
        x[horizontal] = torch.flip(x[horizontal], dims=[3])
        y[horizontal] = torch.flip(y[horizontal], dims=[2])

    vertical = torch.rand(len(x), device=x.device) < 0.5
    if vertical.any():
        x[vertical] = torch.flip(x[vertical], dims=[2])
        y[vertical] = torch.flip(y[vertical], dims=[1])

    k = int(torch.randint(0, 4, (1,), device=x.device).item())
    if k:
        x = torch.rot90(x, k, dims=[2, 3])
        y = torch.rot90(y, k, dims=[1, 2])

    scale = 1.0 + (torch.rand((1, x.shape[1], 1, 1), device=x.device) - 0.5) * 0.1
    noise = torch.randn_like(x) * 0.01
    x = x * scale + noise

    return x, y


def segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_LABEL,
) -> dict:
    """Calcula IoU/F1 por classe e acurácia pixel a pixel."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    valid = y_true != ignore_index
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom_iou = np.maximum(tp + fp + fn, 1.0)
    denom_f1 = np.maximum(2 * tp + fp + fn, 1.0)
    iou = tp / denom_iou
    f1 = 2 * tp / denom_f1

    accuracy = float(tp.sum() / max(cm.sum(), 1))
    present = (tp + fn) > 0
    return {
        "accuracy": accuracy,
        "miou": float(np.mean(iou[present])) if present.any() else 0.0,
        "macro_f1": float(np.mean(f1[present])) if present.any() else 0.0,
        "iou": {
            name: round(float(value), 4)
            for name, value in zip(CLASS_NAMES[:num_classes], iou)
        },
        "f1": {
            name: round(float(value), 4)
            for name, value in zip(CLASS_NAMES[:num_classes], f1)
        },
        "confusion_matrix": cm.tolist(),
    }


def train_unet(
    model: nn.Module,
    X_train: np.ndarray | Dataset,
    y_train: np.ndarray | None,
    X_val: np.ndarray | Dataset,
    y_val: np.ndarray | None,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    save_path: str = "models/unet_model.pth",
    class_weights: np.ndarray | None = None,
    ignore_index: int = IGNORE_LABEL,
    augment: bool = False,
    balance: bool = False,
    patience: int = 0,
    checkpoint_path: str | None = None,
    resume: bool = False,
    checkpoint_every: int = 5,
    focal_gamma: float | None = None,
):
    """Treina a U-Net com augmentation, balanceamento e checkpoint.

    Args:
        model: Modelo U-Net.
        X_train/y_train: Patches de treino (n, H, W, C) e rótulos (n, H, W),
            ou um ``PatchDataset`` para carregar patches sob demanda.
        X_val/y_val: Patches de validação (ou ``PatchDataset``).
        epochs: Número de épocas.
        batch_size: Tamanho do batch.
        learning_rate: Taxa de aprendizado.
        save_path: Caminho do melhor modelo (state_dict).
        class_weights: Pesos por classe.
        ignore_index: Label ignorado (255).
        augment: Aplica flip/rot90/jitter no treino.
        balance: Usa WeightedRandomSampler para classes raras.
        patience: Épocas sem melhora para early stopping (0 = desliga).
        checkpoint_path: Caminho do checkpoint "last" (retomada).
        resume: Retoma do checkpoint se existir.
        focal_gamma: Se informado, usa FocalLoss com esse gamma.

    Returns:
        Histórico com perdas, acurácias e métricas finais (IoU/F1).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, optimizer, criterion = compile_model(
        model,
        learning_rate,
        device,
        class_weights=class_weights,
        ignore_index=ignore_index,
        focal_gamma=focal_gamma,
    )

    if isinstance(X_train, Dataset):
        train_dataset = X_train
        val_dataset = X_val
    else:
        X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
        y_val = torch.LongTensor(y_val)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

    if balance:
        if isinstance(train_dataset, PatchDataset):
            weights = train_dataset.sample_weights(ignore_index)
        else:
            weights = _sample_weights(
                y_train.numpy(), NUM_CLASSES, ignore_index
            )
        if float(weights.sum()) <= 0:
            raise ValueError(
                "sample_weights retornou apenas zeros: nenhum patch de "
                "treino tem pixels válidos. Verifique o dataset."
            )
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    start_epoch = 0

    if checkpoint_path is None:
        checkpoint_path = save_path.replace(".pth", "_last.pth")

    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint.get("epoch", 0))
            history = checkpoint.get("history", history)
            best_val_loss = float(
                checkpoint.get("best_val_loss", best_val_loss)
            )
            print(f"Retomando do checkpoint {checkpoint_path} (época {start_epoch})")

    epochs_without_improvement = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if augment:
                batch_x, batch_y = augment_batch(batch_x, batch_y)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            valid = batch_y != ignore_index
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                continue

            train_loss += loss.item() * valid_count
            _, predicted = torch.max(outputs.data, 1)
            train_total += valid_count
            train_correct += ((predicted == batch_y) & valid).sum().item()

        if train_total == 0:
            raise RuntimeError(
                "Época sem pixels válidos no treino: todos os patches "
                "vieram 100% com rótulo ignorado (255). Verifique o "
                "dataset e o balanceamento (sample_weights)."
            )

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                valid = batch_y != ignore_index
                valid_count = int(valid.sum().item())
                if valid_count == 0:
                    continue

                val_loss += loss.item() * valid_count
                _, predicted = torch.max(outputs.data, 1)
                val_total += valid_count
                val_correct += ((predicted == batch_y) & valid).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1

        should_checkpoint = checkpoint_path and (
            (epoch + 1) % max(checkpoint_every, 1) == 0
            or epoch + 1 == epochs
        )
        if should_checkpoint:
            os.makedirs(
                os.path.dirname(checkpoint_path) or ".", exist_ok=True
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "history": history,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            print(f"Época {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping na época {epoch + 1}.")
            break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.append(predicted)
            y_true.append(batch_y.numpy())

    metrics = segmentation_metrics(
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
        num_classes=NUM_CLASSES,
        ignore_index=ignore_index,
    )
    history["metrics"] = metrics

    print(f"Treinamento concluído! Melhor modelo salvo em: {save_path}")
    print(f"  Acurácia: {metrics['accuracy']:.4f} | mIoU: {metrics['miou']:.4f}")
    return history


def load_model(model: nn.Module, filepath: str, device: str | None = None):
    """Carrega pesos do modelo treinado.

    Args:
        model: Modelo U-Net (arquitetura)
        filepath: Caminho do arquivo .pth
        device: Dispositivo

    Returns:
        Modelo com pesos carregados
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(filepath, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _pad_to_grid(
    image: np.ndarray, patch_size: int, stride: int
) -> tuple[np.ndarray, int, int]:
    """Preenche bordas para que todos os pixels sejam cobertos por patches.

    Imagens menores que ``patch_size`` são aceitas: as bordas são replicadas
    (``mode="edge"``) até o tamanho mínimo do patch. Isso evita falhas em
    áreas pequenas, mantendo o mesmo modelo e a mesma escala do treino.
    """
    original_h, original_w = image.shape[:2]

    pad_h = max(0, patch_size - original_h)
    pad_w = max(0, patch_size - original_w)
    if pad_h or pad_w:
        image = np.pad(
            image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"
        )

    height, width = image.shape[:2]
    grid_h = (stride - (height - patch_size) % stride) % stride
    grid_w = (stride - (width - patch_size) % stride) % stride
    if grid_h or grid_w:
        image = np.pad(
            image, ((0, grid_h), (0, grid_w), (0, 0)), mode="edge"
        )
    return image, original_h, original_w


def predict_unet_proba(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    num_classes: int = NUM_CLASSES,
    batch_size: int = 8,
) -> np.ndarray:
    """Aplica a U-Net e retorna probabilidades (height, width, num_classes).

    As probabilidades dos patches sobrepostos são acumuladas e normalizadas,
    em vez de média de índices de classe (que era incorreta). Os patches são
    processados em lotes para acelerar a inferência em CPU sem mudar o
    resultado.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    padded, original_h, original_w = _pad_to_grid(image, patch_size, stride)
    h, w = padded.shape[:2]
    batch_size = max(1, int(batch_size))

    proba_map = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    coordinates = [
        (y, x)
        for y in range(0, h - patch_size + 1, stride)
        for x in range(0, w - patch_size + 1, stride)
    ]

    with torch.inference_mode():
        for start in range(0, len(coordinates), batch_size):
            batch = coordinates[start : start + batch_size]
            patches = np.stack(
                [
                    padded[y : y + patch_size, x : x + patch_size]
                    for y, x in batch
                ]
            )
            patch_tensor = (
                torch.from_numpy(patches)
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
            )

            output = model(patch_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()

            for (y, x), probabilities_patch in zip(
                batch, probabilities, strict=True
            ):
                proba_map[y : y + patch_size, x : x + patch_size] += np.moveaxis(
                    probabilities_patch, 0, -1
                )
                count_map[y : y + patch_size, x : x + patch_size] += 1

    count_map[count_map == 0] = 1
    proba_map /= count_map[..., None]
    return proba_map[:original_h, :original_w]


def predict_unet(
    model: nn.Module, image: np.ndarray, patch_size: int = 64, stride: int = 32
):
    """Aplica o modelo em uma imagem completa e retorna o mapa de classes."""
    probabilities = predict_unet_proba(
        model,
        image,
        patch_size=patch_size,
        stride=stride,
    )
    return np.argmax(probabilities, axis=-1).astype(np.uint8)
