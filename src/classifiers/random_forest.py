"""Classificador Random Forest."""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.evaluation.spatial_split import spatial_holdout_indices


def prepare_data_rf(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    groups: np.ndarray | None = None,
):
    """Prepara dados para treinamento do RF.

    Sem ``groups`` aplica Hold-Out 70/30 aleatório (estratificado).

    Com ``groups`` (ex.: blocos espaciais) aplica Hold-Out mantendo blocos
    inteiros em treino ou teste, evitando autocorrelação espacial.

    Args:
        features: Array de atributos (n_samples, n_features)
        labels: Array de rótulos (n_samples,)
        test_size: Proporção do conjunto de teste
        random_state: Semente aleatória
        groups: Identificador de bloco espacial por amostra (opcional)

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    if groups is not None:
        train_idx, test_idx = spatial_holdout_indices(
            groups, test_size=test_size, random_state=random_state
        )
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        print(f"Divisão espacial: {len(np.unique(groups))} blocos")
    else:
        _, counts = np.unique(labels, return_counts=True)
        stratify = labels if counts.min() >= 2 else None

        if stratify is None:
            print("Aviso: estratificação desativada por classes com poucas amostras.")

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    print(f"Treinamento: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")

    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trees: int = 200,
    max_depth: int = 15,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    verbose: int = 1,
):
    """Treina o modelo Random Forest.

    Args:
        X_train: Dados de treinamento
        y_train: Rótulos de treinamento
        n_trees: Número de árvores (default: 200)
        max_depth: Profundidade máxima (default: 15)
        random_state: Semente aleatória

    Returns:
        Modelo treinado
    """
    print("Treinando Random Forest...")
    print(f"  - Árvores: {n_trees}")
    print(f"  - Profundidade máxima: {max_depth}")

    model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose,
    )

    model.fit(X_train, y_train)

    print("Treinamento concluído!")

    return model


def predict_random_forest(model, X_test: np.ndarray) -> np.ndarray:
    """Realiza predições com o modelo treinado.

    Args:
        model: Modelo treinado
        X_test: Dados de teste

    Returns:
        Array com as predições
    """
    return model.predict(X_test)


def save_model(model, filepath: str):
    """Salva o modelo treinado em disco.

    Args:
        model: Modelo treinado
        filepath: Caminho para salvar (.pkl)
    """
    joblib.dump(model, filepath)
    print(f"Modelo salvo em: {filepath}")


def load_model(filepath: str):
    """Carrega um modelo treinado do disco.

    Args:
        filepath: Caminho do modelo (.pkl)

    Returns:
        Modelo carregado
    """
    return joblib.load(filepath)
