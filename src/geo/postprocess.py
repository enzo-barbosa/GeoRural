"""Pós-processamento de mapas de classificação."""

import numpy as np


def majority_filter(
    class_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    kernel_size: int = 3,
    num_classes: int | None = None,
) -> np.ndarray:
    """Suaviza um mapa de classes por voto de maioria em janela local."""
    class_map = np.asarray(class_map)
    if kernel_size <= 1 or class_map.ndim != 2:
        return class_map

    import cv2

    if num_classes is None:
        num_classes = int(class_map.max()) + 1

    if valid_mask is None:
        valid_mask = np.ones(class_map.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    counts = np.stack(
        [
            cv2.boxFilter(
                ((class_map == class_id) & valid_mask).astype(np.float32),
                -1,
                (kernel_size, kernel_size),
                normalize=False,
            )
            for class_id in range(num_classes)
        ],
        axis=0,
    )
    smoothed = counts.argmax(axis=0).astype(class_map.dtype)

    # Fora da máscara, preserva o valor original (a transparência é aplicada
    # depois na geração do PNG).
    smoothed[~valid_mask] = class_map[~valid_mask]
    return smoothed
