"""Rede Neural Convolucional U-Net para segmentação semântica."""

import glob
import os

import numpy as np
import torch
from torch import nn, optim

from src.config import IGNORE_LABEL, NUM_CLASSES


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
):
    """Monta patches de treino/validação deixando um tile inteiro de fora."""
    if val_tile not in tiles:
        raise KeyError(f"Tile de validação desconhecido: {val_tile}")

    train_img, train_lbl, val_img, val_lbl = [], [], [], []
    for name, (image, labels) in tiles.items():
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


def compile_model(
    model: nn.Module,
    learning_rate: float = 0.001,
    device: str | None = None,
    class_weights: np.ndarray | None = None,
    ignore_index: int = IGNORE_LABEL,
):
    """Compila o modelo com otimizador Adam e loss apropriada.

    Args:
        model: Modelo U-Net
        learning_rate: Taxa de aprendizado
        device: Dispositivo ('cuda' ou 'cpu')
        class_weights: Pesos por classe (para classes desbalanceadas)

    Returns:
        Modelo compilado, optimizer, criterion
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(
            weight=weights.to(device), ignore_index=ignore_index
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    print(f"Modelo compilado no dispositivo: {device}")
    print(model)

    return model, optimizer, criterion


def train_unet(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    save_path: str = "models/unet_model.pth",
    class_weights: np.ndarray | None = None,
    ignore_index: int = IGNORE_LABEL,
):
    """Treina o modelo U-Net.

    Args:
        model: Modelo U-Net
        X_train: Patches de treinamento (n_samples, height, width, channels)
        y_train: Rótulos de treinamento (n_samples, height, width)
        X_val: Patches de validação
        y_val: Rótulos de validação
        epochs: Número de épocas (default: 100)
        batch_size: Tamanho do batch (default: 16)
        learning_rate: Taxa de aprendizado (default: 0.001)
        save_path: Caminho para salvar o melhor modelo

    Returns:
        Histórico de treinamento
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, optimizer, criterion = compile_model(
        model,
        learning_rate,
        device,
        class_weights=class_weights,
        ignore_index=ignore_index,
    )

    # Converter numpy arrays para tensors PyTorch
    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    y_val = torch.LongTensor(y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

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
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print(f"Treinamento concluído! Melhor modelo salvo em: {save_path}")
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

    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model


def _pad_to_grid(
    image: np.ndarray, patch_size: int, stride: int
) -> tuple[np.ndarray, int, int]:
    """Preenche bordas para que todos os pixels sejam cobertos por patches."""
    height, width = image.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Imagem ({height}x{width}) menor que patch_size ({patch_size})."
        )

    pad_h = (stride - (height - patch_size) % stride) % stride
    pad_w = (stride - (width - patch_size) % stride) % stride
    if pad_h or pad_w:
        image = np.pad(
            image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"
        )
    return image, height, width


def predict_unet_proba(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Aplica a U-Net e retorna probabilidades (height, width, num_classes).

    As probabilidades dos patches sobrepostos são acumuladas e normalizadas,
    em vez de média de índices de classe (que era incorreta).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    padded, original_h, original_w = _pad_to_grid(image, patch_size, stride)
    h, w = padded.shape[:2]

    proba_map = np.zeros((h, w, num_classes), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = padded[y : y + patch_size, x : x + patch_size]
                patch_tensor = (
                    torch.FloatTensor(patch).permute(2, 0, 1).unsqueeze(0).to(device)
                )

                output = model(patch_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                probs = np.moveaxis(probs, 0, -1)

                proba_map[y : y + patch_size, x : x + patch_size] += probs
                count_map[y : y + patch_size, x : x + patch_size] += 1

    count_map[count_map == 0] = 1
    proba_map /= count_map[..., None]
    return proba_map[:original_h, :original_w]


def predict_unet(
    model: nn.Module, image: np.ndarray, patch_size: int = 64, stride: int = 32
):
    """Aplica o modelo em uma imagem completa e retorna o mapa de classes."""
    probabilities = predict_unet_proba(
        model, image, patch_size=patch_size, stride=stride
    )
    return np.argmax(probabilities, axis=-1).astype(np.uint8)
