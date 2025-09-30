import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import os

# Caminhos dos arquivos
MODEL_PATH = "notebook/pipeline_churn.pkl"  # onde está seu modelo
TEST_DATA_PATH = "customer_churn_data.csv"  # CSV de clientes para prever
OUTPUT_PATH = "reports/result_with_predictions.csv"
METRICS_PATH = "reports/metrics.txt"


def load_model(model_path=MODEL_PATH):
    """Carrega o modelo treinado."""
    model = joblib.load(model_path)
    print("✅ Modelo carregado com sucesso!")
    return model


def calculate_metrics(y_true, y_pred, y_proba, output_path=METRICS_PATH):
    """Calcula e salva métricas em arquivo."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_proba),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"📊 Métricas salvas em {output_path}")


def make_predictions(data_path="customer_churn_data.xlsx", threshold=0.5):
    """Gera previsões de churn e salva em CSV para o dashboard."""
    model = load_model()

    # Detecta formato do arquivo (Excel ou CSV)
    if data_path.endswith(".xlsx") or data_path.endswith(".xls"):
        data = pd.read_excel(data_path)
    else:
        data = pd.read_csv(data_path, encoding="utf-8", sep=",")

    # Se existir a coluna de target no dataset
    if "Churn" in data.columns:
        y_true = data["Churn"].map({"Sim": 1, "Não": 0})
    else:
        y_true = None

    # Previsões
    y_proba = model.predict_proba(data.drop(columns=["Churn"], errors="ignore"))[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Salva resultados no dataframe
    data["predicted_proba"] = y_proba
    data["predicted_churn"] = y_pred
    if y_true is not None:
        data["true_churn"] = y_true

    # Garante diretório e salva
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"💾 Resultados salvos em {OUTPUT_PATH}")

    # Calcula métricas se tiver y_true
    if y_true is not None:
        calculate_metrics(y_true, y_pred, y_proba)


if __name__ == "__main__":
    make_predictions()
