import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Dashboard de Churn", layout="wide")
st.title("📊 Dashboard de Predição de Churn")

# Carregar automaticamente o CSV de predições
DATA_PATH = "reports/result_with_predictions.csv"

try:
    df = pd.read_csv(DATA_PATH)

    st.subheader("📈 Histograma das Probabilidades de Churn")
    fig = px.histogram(df, x="predicted_proba", nbins=30)
    fig.add_vline(
        x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold=0.5"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Métricas de Classificação")
    if "true_churn" in df.columns and "predicted_churn" in df.columns:
        report = classification_report(
            df["true_churn"], df["predicted_churn"], output_dict=True
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{report['accuracy']:.2%}")
        col2.metric("Precision (Churn)", f"{report['1']['precision']:.2%}")
        col3.metric("Recall (Churn)", f"{report['1']['recall']:.2%}")
        col4.metric("F1-Score (Churn)", f"{report['1']['f1-score']:.2%}")

        # Matrizes de confusão
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

        # Curva ROC
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

except FileNotFoundError:
    st.error("Arquivo de predições não encontrado. Rode o predict.py primeiro.")
