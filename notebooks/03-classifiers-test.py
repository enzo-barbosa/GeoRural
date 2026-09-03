"""Treino e avaliação dos classificadores clássicos."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from src.classifiers.random_forest import (
    prepare_data_rf, train_random_forest, predict_random_forest, save_model as save_rf
)
from src.classifiers.svm import train_svm, predict_svm, save_model as save_svm
from src.evaluation.metrics import full_evaluation, print_evaluation
from src.evaluation.confusion import plot_confusion_matrix

# 1. Carregar dados de treinamento
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/samples/training_samples.csv"))

# Separar atributos e rótulos
feature_cols = [c for c in df.columns if c not in {"class_id", "y", "x"} and df[c].dtype != object]
X = df[feature_cols].values
y = df["class_id"].values

print(f"Total de amostras: {len(X)}")
print(f"Classes: {np.unique(y)}")

# 2. Preparar dados (Hold-Out 70/30)
X_train, X_test, y_train, y_test = prepare_data_rf(X, y)

# 3. Treinar Random Forest
rf_model = train_random_forest(X_train, y_train)
rf_predictions = predict_random_forest(rf_model, X_test)

# 4. Treinar SVM
svm_model = train_svm(X_train, y_train)
svm_predictions = predict_svm(svm_model, X_test)

# 5. Avaliar Random Forest
print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)
print_evaluation(y_test, rf_predictions)
plot_confusion_matrix(
    y_test, rf_predictions,
    save_path=os.path.join(PROJECT_ROOT, "outputs/rf_confusion_matrix.png")
)

# 6. Avaliar SVM
print("\n" + "=" * 60)
print("SVM")
print("=" * 60)
print_evaluation(y_test, svm_predictions)
plot_confusion_matrix(
    y_test, svm_predictions,
    save_path=os.path.join(PROJECT_ROOT, "outputs/svm_confusion_matrix.png")
)

# 7. Comparação lado a lado
rf_metrics = full_evaluation(y_test, rf_predictions)
svm_metrics = full_evaluation(y_test, svm_predictions)

comparison = pd.DataFrame({
    "Métrica": ["Acurácia Global", "Kappa", "Precisão Média", "Recall Médio", "F1 Médio"],
    "Random Forest": [
        rf_metrics["overall_accuracy"],
        rf_metrics["kappa"],
        np.mean(list(rf_metrics["precision"].values())),
        np.mean(list(rf_metrics["recall"].values())),
        np.mean(list(rf_metrics["f1_score"].values()))
    ],
    "SVM": [
        svm_metrics["overall_accuracy"],
        svm_metrics["kappa"],
        np.mean(list(svm_metrics["precision"].values())),
        np.mean(list(svm_metrics["recall"].values())),
        np.mean(list(svm_metrics["f1_score"].values()))
    ]
})

print("\n" + "=" * 60)
print("COMPARAÇÃO RF vs SVM")
print("=" * 60)
print(comparison.to_string(index=False))

# 8. Importância dos atributos (RF)
from src.evaluation.confusion import plot_feature_importance
plot_feature_importance(
    rf_model.feature_importances_,
    feature_cols,
    save_path=os.path.join(PROJECT_ROOT, "outputs/rf_feature_importance.png")
)


# 9. Salvar modelos treinados
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
save_rf(rf_model, os.path.join(PROJECT_ROOT, "models/random_forest.pkl"))
save_svm(svm_model, os.path.join(PROJECT_ROOT, "models/svm.pkl"))
