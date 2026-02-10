from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

st.set_page_config(page_title="Dashboard de Churn", layout="wide")
st.title("📊 Dashboard de Predição de Churn")

# Path portátil: funciona da raiz do projeto ou da pasta streamlit
BASE = Path(__file__).resolve().parent.parent
DATA_PATHS = [
    BASE / "reports" / "result_with_predictions.csv",
    Path("reports/result_with_predictions.csv"),
]

# Opção de upload
use_upload = st.sidebar.checkbox("Usar arquivo enviado", value=False)
uploaded_file = st.sidebar.file_uploader(
    "Ou envie um CSV com colunas predicted_proba, predicted_churn e (opcional) true_churn",
    type=["csv"],
)

df = None
data_source = ""

if use_upload and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
        data_source = "upload"
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
else:
    for p in DATA_PATHS:
        if p.exists():
            try:
                df = pd.read_csv(p, encoding="utf-8")
                data_source = str(p)
                break
            except Exception as e:
                st.warning(f"Arquivo encontrado mas erro ao ler: {e}")

if df is None or df.empty:
    st.error(
        "Arquivo de predições não encontrado. "
        "Execute `python -m src.models.predict` ou faça upload de um CSV na barra lateral."
    )
    st.info(
        "O CSV deve conter as colunas: `predicted_proba`, `predicted_churn` e opcionalmente `true_churn`."
    )
else:
    if data_source:
        st.caption(f"Dados: {data_source}")

    st.subheader("📈 Histograma das Probabilidades de Churn")
    fig = px.histogram(df, x="predicted_proba", nbins=30)
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold=0.5")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Métricas de Classificação")
    if "true_churn" in df.columns and "predicted_churn" in df.columns:
        report = classification_report(
            df["true_churn"], df["predicted_churn"], output_dict=True
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{report['accuracy']:.2%}")

        # Evita KeyError se não houver classe 1
        if "1" in report:
            col2.metric("Precision (Churn)", f"{report['1']['precision']:.2%}")
            col3.metric("Recall (Churn)", f"{report['1']['recall']:.2%}")
            col4.metric("F1-Score (Churn)", f"{report['1']['f1-score']:.2%}")
        else:
            st.warning("Não há exemplos da classe Churn (1) para calcular métricas.")

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
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.2f}")
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
    else:
        st.info(
            "Para ver métricas, matriz de confusão e ROC, o CSV precisa das colunas "
            "`true_churn` e `predicted_churn`."
        )
