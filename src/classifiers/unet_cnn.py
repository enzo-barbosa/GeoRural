"""Rede Neural Convolucional U-Net para segmentação semântica."""

import numpy as np
import torch
from torch import nn, optim


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
        self, n_channels: int = 10, n_classes: int = 5, bilinear: bool = False
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
    input_shape: tuple[int, int, int] = (128, 128, 10), num_classes: int = 5
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


def compile_model(model: nn.Module, learning_rate: float = 0.001, device: str | None = None):
    """Compila o modelo com otimizador Adam e loss apropriada.

    Args:
        model: Modelo U-Net
        learning_rate: Taxa de aprendizado
        device: Dispositivo ('cuda' ou 'cpu')

    Returns:
        Modelo compilado, optimizer, criterion
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

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
    model, optimizer, criterion = compile_model(model, learning_rate, device)

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

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.numel()
            train_correct += (predicted == batch_y).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.numel()
                val_correct += (predicted == batch_y).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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


def predict_unet(
    model: nn.Module, image: np.ndarray, patch_size: int = 64, stride: int = 32
):
    """Aplica o modelo em uma imagem completa.

    Args:
        model: Modelo U-Net treinado
        image: Array da imagem (height, width, channels)
        patch_size: Tamanho do patch
        stride: Passo entre patches

    Returns:
        Mapa de classificação (height, width)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    h, w = image.shape[:2]
    prediction_map = np.zeros((h, w), dtype=np.uint8)
    count_map = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y : y + patch_size, x : x + patch_size]
                patch_tensor = (
                    torch.FloatTensor(patch).permute(2, 0, 1).unsqueeze(0).to(device)
                )

                output = model(patch_tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]

                prediction_map[y : y + patch_size, x : x + patch_size] += pred
                count_map[y : y + patch_size, x : x + patch_size] += 1

    count_map[count_map == 0] = 1
    prediction_map = (prediction_map / count_map).astype(np.uint8)

    return prediction_map
