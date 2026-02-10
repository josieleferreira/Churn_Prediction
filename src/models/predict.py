"""Script para gerar predições de churn e salvar para o dashboard."""
import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Caminhos padrão (configuráveis via env ou argumentos)
BASE = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE / "app" / "churn_prediction.pkl"))
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH", str(BASE / "reports" / "result_with_predictions.csv")
)
METRICS_PATH = os.environ.get("METRICS_PATH", str(BASE / "reports" / "metrics.txt"))


def load_model(model_path: str = MODEL_PATH):
    """Carrega o modelo treinado."""
    model = joblib.load(model_path)
    print("✅ Modelo carregado com sucesso!")
    return model


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Aplica o mesmo pré-processamento do notebook antes da predição."""
    df = data.copy()

    # Remove colunas redundantes
    if "Emite boletos.1" in df.columns:
        df = df.drop(columns=["Emite boletos.1"])
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Preenche Receita total quando ausente
    if all(c in df.columns for c in ["Receita total", "Receita mensal", "Meses de permanência"]):
        mask = df["Receita total"].isna()
        df.loc[mask, "Receita total"] = (
            df.loc[mask, "Receita mensal"] * df.loc[mask, "Meses de permanência"]
        )

    return df


def calculate_metrics(y_true, y_pred, y_proba, output_path: str = METRICS_PATH):
    """Calcula e salva métricas em arquivo."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_proba),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"📊 Métricas salvas em {output_path}")


def make_predictions(
    data_path: str | None = None,
    threshold: float = 0.5,
) -> None:
    """Gera previsões de churn e salva em CSV para o dashboard."""
    data_path = data_path or os.environ.get(
        "DATA_PATH", str(BASE / "customer_churn_data.xlsx")
    )

    model = load_model(MODEL_PATH)

    # Detecta formato do arquivo
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

    if path.suffix in (".xlsx", ".xls"):
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path, encoding="utf-8", sep=",")

    data = _prepare_data(data)

    if "Churn" in data.columns:
        y_true = data["Churn"].map({"Sim": 1, "Não": 0})
    else:
        y_true = None

    X = data.drop(columns=["Churn"], errors="ignore")
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    data["predicted_proba"] = y_proba
    data["predicted_churn"] = y_pred
    if y_true is not None:
        data["true_churn"] = y_true

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"💾 Resultados salvos em {OUTPUT_PATH}")

    if y_true is not None:
        calculate_metrics(y_true, y_pred, y_proba)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        help="Caminho para dados (Excel ou CSV). Default: customer_churn_data.xlsx",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold para classificação (default: 0.5)",
    )
    args = parser.parse_args()

    make_predictions(data_path=args.data, threshold=args.threshold)
