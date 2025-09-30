import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import os
import platform
from datetime import datetime

# ================================================
# 🎨 Estilo customizado (sidebar + títulos + cores)
# ================================================
st.markdown(
    """
    <style>
        /* Sidebar mais larga */
        [data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 320px;
            background-color: #f0eee9; /* Sidebar tom suave */
            padding: 20px 15px 20px 15px;
        }

        /* Só o título da sidebar (🗂️ Selecione a aba desejada) */
        [data-testid="stSidebar"] h1 {
            font-size: 20px !important;
            font-weight: normal !important;  /* sem negrito */
            color: #000000;
        }

        /* Texto "Ir para:" */
        section[data-testid="stSidebar"] label p {
            font-size: 18px !important;
            font-weight: normal !important;
            color: #000000;
        }

        /* Itens do radio (as opções) */
        div[role="radiogroup"] label p {
            font-size: 16px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ================================================
# Detectar sistema operacional e ajustar URI
# ================================================
if platform.system() == "Windows":
    MLFLOW_TRACKING_URI = "file:///C:/Projetos Pessoais/Churn_Prediction/mlruns"
    ARTIFACT_BASE = r"C:\Projetos Pessoais\Churn_Prediction\mlruns"
else:  # Linux / Mac
    MLFLOW_TRACKING_URI = "file:/home/josiele/Projeto Pessoal/Churn_Prediction/mlruns"
    ARTIFACT_BASE = "/home/josiele/Projeto Pessoal/Churn_Prediction/mlruns"

EXPERIMENT_NAME = "churn_experiment"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1
)
latest_run = runs[0] if runs else None

# ================================================
# Menu lateral
# ================================================
st.sidebar.title("🗂️ Selecione a aba desejada")
pagina = st.sidebar.radio("Ir para:", ["📊 Gráficos", "🛠 Monitoramento"])

# ================================================
# Aba: Gráficos (dados do MLflow)
# ================================================
if pagina == "📊 Gráficos":
    st.markdown(
        "<h1 style='text-align: center;'>📊 Dashboard de Predição de Churn</h1>",
        unsafe_allow_html=True,
    )

    if latest_run:
        metrics = latest_run.data.metrics

        # ====== Métricas ======
        st.subheader("📊 Métricas de Classificação")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy',0)*100:.2f}%")
        col2.metric("Precision", f"{metrics.get('precision',0)*100:.2f}%")
        col3.metric("Recall", f"{metrics.get('recall',0)*100:.2f}%")
        col4.metric("F1", f"{metrics.get('f1',0)*100:.2f}%")
        col5.metric("ROC AUC", f"{metrics.get('roc_auc',0)*100:.2f}%")

        # ====== Artefatos ======
        st.subheader("📈 Visualizações de Desempenho")

        REPORTS_DIR = "/home/josiele/Projeto Pessoal/Churn_Prediction/reports"

        cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")

        if os.path.exists(cm_path):
            st.image(cm_path, caption="📊 Matriz de Confusão", use_container_width=True)
        else:
            st.warning("⚠️ Arquivo confusion_matrix.png não encontrado.")

        if os.path.exists(roc_path):
            st.image(roc_path, caption="📈 Curva ROC", use_container_width=True)
        else:
            st.warning("⚠️ Arquivo roc_curve.png não encontrado.")


# ================================================
# Aba: Monitoramento (produção)
# ================================================
if pagina == "🛠 Monitoramento":
    st.markdown(
        "<h1 style='text-align: center;'>📡 Monitoramento do Modelo em Produção</h1>",
        unsafe_allow_html=True,
    )

    # 🔹 Lê o CSV de predições salvas
    if os.path.exists("reports/result_with_predictions.csv"):
        df_log = pd.read_csv("reports/result_with_predictions.csv")

        # Últimos registros
        st.subheader("📋 Últimas Predições Registradas")
        st.dataframe(df_log.tail(10))

        # Evolução dos scores
        if "predicted_proba" in df_log.columns:
            st.subheader("📈 Evolução dos Scores")
            st.line_chart(df_log["predicted_proba"].tail(50))

            # Distribuição
            st.subheader("📊 Distribuição das Probabilidades")
            fig, ax = plt.subplots()
            ax.hist(df_log["predicted_proba"], bins=20, color="royalblue", alpha=0.8)
            ax.axvline(0.5, color="red", linestyle="--", label="Threshold=0.5")
            ax.legend()
            st.pyplot(fig)

        # Taxa de positivos
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
        st.warning("Nenhum arquivo de monitoramento encontrado. Rode `predict.py` para gerar previsões.")
