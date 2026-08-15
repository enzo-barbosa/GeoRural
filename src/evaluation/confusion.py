"""Matriz de confusão e visualização."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["Agricultura", "Pastagem", "Veg. Nativa", "Solo Exposto", "Área Const."]


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str | None = None
):
    """Gera e plota a matriz de confusão.

    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        save_path: Caminho para salvar a figura (None = não salvar)
    """
    cm = confusion_matrix(y_true, y_pred)

    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title("Matriz de Confusão (Contagem)")
    axes[0].set_ylabel("Valor Real")
    axes[0].set_xlabel("Valor Previsto")

    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title("Matriz de Confusão (%)")
    axes[1].set_ylabel("Valor Real")
    axes[1].set_xlabel("Valor Previsto")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Matriz de confusão salva em: {save_path}")

    plt.show()


def plot_feature_importance(
    importances: np.ndarray, feature_names: list, save_path: str | None = None
):
    """Plota a importância dos atributos (apenas para Random Forest)."""
    indices = np.argsort(importances)[::-1]

    _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], align="center")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
    ax.set_title("Importância dos Atributos (Random Forest)")
    ax.set_ylabel("Importância")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
