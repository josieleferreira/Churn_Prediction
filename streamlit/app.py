import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlflow.tracking import MlflowClient
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

st.set_page_config(page_title="Dashboard de Churn", layout="wide")

# Path portátil
BASE = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE / "reports"

if platform.system() == "Windows":
    MLFLOW_TRACKING_URI = f"file:///{BASE / 'mlruns'}".replace("\\", "/")
else:
    MLFLOW_TRACKING_URI = f"file://{BASE / 'mlruns'}"

EXPERIMENT_NAME = "churn_experiment"
mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
client = MlflowClient()

try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    latest_run = runs[0] if runs else None
except Exception:
    latest_run = None

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 320px;
            background-color: #f0eee9;
            padding: 20px 15px 20px 15px;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 20px !important;
            font-weight: normal !important;
            color: #000000;
        }
        section[data-testid="stSidebar"] label p {
            font-size: 18px !important;
            font-weight: normal !important;
            color: #000000;
        }
        div[role="radiogroup"] label p {
            font-size: 16px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🗂️ Selecione a aba desejada")
pagina = st.sidebar.radio("Ir para:", ["📊 Gráficos", "🛠 Monitoramento"])

if pagina == "📊 Gráficos":
    st.markdown(
        "<h1 style='text-align: center;'>📊 Dashboard de Predição de Churn</h1>",
        unsafe_allow_html=True,
    )

    use_csv = st.sidebar.checkbox("Usar CSV de predições (em vez do MLflow)", value=False)
    csv_path = REPORTS_DIR / "result_with_predictions.csv"

    if use_csv and csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8")
        st.caption(f"Dados: {csv_path}")

        st.subheader("📈 Histograma das Probabilidades de Churn")
        fig = px.histogram(df, x="predicted_proba", nbins=30)
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold=0.5")
        st.plotly_chart(fig, use_container_width=True)

        if "true_churn" in df.columns and "predicted_churn" in df.columns:
            report = classification_report(
                df["true_churn"], df["predicted_churn"], output_dict=True
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{report['accuracy']:.2%}")
            if "1" in report:
                col2.metric("Precision (Churn)", f"{report['1']['precision']:.2%}")
                col3.metric("Recall (Churn)", f"{report['1']['recall']:.2%}")
                col4.metric("F1-Score (Churn)", f"{report['1']['f1-score']:.2%}")

            st.subheader("🧮 Matrizes de Confusão")
            cm = confusion_matrix(df["true_churn"], df["predicted_churn"])
            cm_prop = cm / cm.sum(axis=1, keepdims=True)
            col1, col2 = st.columns(2)
            with col1:
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=["Predito 0", "Predito 1"],
                        y=["Real 0", "Real 1"],
                        text=cm,
                        texttemplate="%{text}",
                        colorscale="Blues",
                    )
                )
                fig_cm.update_layout(title="Matriz de Confusão (Absoluta)")
                st.plotly_chart(fig_cm, use_container_width=True)
            with col2:
                fig_cm_prop = go.Figure(
                    data=go.Heatmap(
                        z=np.round(cm_prop, 2),
                        x=["Predito 0", "Predito 1"],
                        y=["Real 0", "Real 1"],
                        text=np.round(cm_prop * 100, 2),
                        texttemplate="%{text}%",
                        colorscale="Greens",
                    )
                )
                fig_cm_prop.update_layout(title="Matriz de Confusão (Proporcional)")
                st.plotly_chart(fig_cm_prop, use_container_width=True)

            st.subheader("📉 Curva ROC")
            fpr, tpr, _ = roc_curve(df["true_churn"], df["predicted_proba"])
            roc_auc_val = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc_val:.2f}")
            )
            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"))
            )
            fig_roc.update_layout(
                title="Curva ROC",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig_roc, use_container_width=True)
    elif latest_run:
        metrics = latest_run.data.metrics
        st.subheader("📊 Métricas de Classificação")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
        col2.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
        col3.metric("Recall", f"{metrics.get('recall', 0) * 100:.2f}%")
        col4.metric("F1", f"{metrics.get('f1', 0) * 100:.2f}%")
        col5.metric("ROC AUC", f"{metrics.get('roc_auc', 0) * 100:.2f}%")

        st.subheader("📈 Visualizações de Desempenho")
        cm_path = REPORTS_DIR / "confusion_matrix.png"
        roc_path = REPORTS_DIR / "roc_curve.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="📊 Matriz de Confusão", use_container_width=True)
        else:
            st.warning("⚠️ Arquivo confusion_matrix.png não encontrado.")
        if roc_path.exists():
            st.image(str(roc_path), caption="📈 Curva ROC", use_container_width=True)
        else:
            st.warning("⚠️ Arquivo roc_curve.png não encontrado.")
    else:
        st.info(
            "Execute o predict para gerar o CSV ou ative o MLflow. "
            "Marque 'Usar CSV de predições' se o arquivo existir em reports/."
        )

if pagina == "🛠 Monitoramento":
    st.markdown(
        "<h1 style='text-align: center;'>📡 Monitoramento do Modelo em Produção</h1>",
        unsafe_allow_html=True,
    )

    csv_path = REPORTS_DIR / "result_with_predictions.csv"
    if csv_path.exists():
        df_log = pd.read_csv(csv_path, encoding="utf-8")

        st.subheader("📋 Últimas Predições Registradas")
        st.dataframe(df_log.tail(10))

        if "predicted_proba" in df_log.columns:
            st.subheader("📈 Evolução dos Scores")
            st.line_chart(df_log["predicted_proba"].tail(50))

            st.subheader("📊 Distribuição das Probabilidades")
            fig, ax = plt.subplots()
            ax.hist(df_log["predicted_proba"], bins=20, color="royalblue", alpha=0.8)
            ax.axvline(0.5, color="red", linestyle="--", label="Threshold=0.5")
            ax.legend()
            st.pyplot(fig)

        if "predicted_churn" in df_log.columns:
            churn_rate = df_log["predicted_churn"].mean() * 100
            st.metric("Taxa prevista (positivos)", f"{churn_rate:.2f}%")
            if churn_rate > 40:
                st.error("🚨 Atenção! Taxa de positivos acima do esperado.")
            elif churn_rate < 5:
                st.warning("⚠️ Poucos casos previstos — pode indicar drift.")
            else:
                st.success("✅ Predições dentro do esperado.")
    else:
        st.warning(
            "Nenhum arquivo de monitoramento encontrado. "
            "Execute `python -m src.models.predict` para gerar previsões."
        )
