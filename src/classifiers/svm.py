"""Classificador Support Vector Machine."""

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 100,
    gamma: str = "scale",
    random_state: int = 42,
    verbose: bool = True,
):
    """Treina o modelo SVM.

    IMPORTANTE: O SVM requer normalização dos dados.
    Usa StandardScaler (média=0, desvio_padrão=1).

    Args:
        X_train: Dados de treinamento
        y_train: Rótulos de treinamento
        kernel: Tipo de kernel ('rbf', 'linear', 'poly')
        C: Parâmetro de regularização (default: 100)
        gamma: Coeficiente do kernel RBF ('scale', 'auto', ou float)
        random_state: Semente aleatória

    Returns:
        Pipeline com scaler + modelo treinado
    """
    print("Treinando SVM...")
    print(f"  - Kernel: {kernel}")
    print(f"  - C: {C}")
    print(f"  - Gamma: {gamma}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    random_state=random_state,
                    verbose=verbose,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Treinamento concluído!")

    return pipeline


def predict_svm(model, X_test: np.ndarray) -> np.ndarray:
    """Realiza predições com o modelo treinado.

    Args:
        model: Pipeline treinado (scaler + SVM)
        X_test: Dados de teste

    Returns:
        Array com as predições
    """
    return model.predict(X_test)


def save_model(model, filepath: str):
    """Salva o modelo treinado em disco."""
    joblib.dump(model, filepath)
    print(f"Modelo salvo em: {filepath}")


def load_model(filepath: str):
    """Carrega um modelo treinado do disco."""
    return joblib.load(filepath)
