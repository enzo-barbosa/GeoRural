"""Utilitários de renderização de camadas temáticas (cores e sobreposição)."""

import numpy as np
from PIL import Image


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Converte '#rrggbb' em tupla (r, g, b)."""
    value = color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def colorize(class_map: np.ndarray, colors: list[str]) -> np.ndarray:
    """Colore um mapa de classes (ids 0..N) segundo uma lista de cores."""
    height, width = class_map.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        rgb[class_map == index] = hex_to_rgb(color)
    return rgb


def colorize_by_name(class_map: np.ndarray, names: list[str], colors: dict) -> np.ndarray:
    """Colore um mapa de classes usando um dicionário nome -> cor."""
    height, width = class_map.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for index, name in enumerate(names):
        rgb[class_map == index] = hex_to_rgb(colors.get(name, "#000000"))
    return rgb


def blend(
    background: np.ndarray, overlay: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Sobrepoõe duas imagens RGB com transparência."""
    base = background.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    top = overlay.astype(np.float32) / 255.0
    blended = np.clip(base * (1 - alpha) + top * alpha, 0.0, 1.0) * 255.0
    return blended.astype(np.uint8)


def to_image(array: np.ndarray) -> Image.Image:
    """Converte um array RGB (H, W, 3) em PIL Image."""
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def nearest_resize(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Redimensiona por vizinho mais próximo (suporta múltiplos canais)."""
    height, width = shape
    if array.shape[:2] == (height, width):
        return array
    ys = (np.arange(height) * array.shape[0] / height).astype(int)
    xs = (np.arange(width) * array.shape[1] / width).astype(int)
    ys = ys.clip(0, array.shape[0] - 1)
    xs = xs.clip(0, array.shape[1] - 1)
    return array[ys][:, xs]
