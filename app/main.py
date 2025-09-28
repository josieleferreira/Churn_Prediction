"""
Módulo principal da API de Churn Prediction usando FastAPI.
Fornece endpoints para health check e predição de churn a partir de dados de clientes.
"""

import os
from typing import List
from functools import lru_cache

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

# Caminho do modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_prediction.pkl")


@lru_cache(maxsize=1)
def get_model():
    """Carrega o modelo somente uma vez (lazy loading com cache)."""
    return joblib.load(MODEL_PATH)


class Customer(BaseModel):
    """Modelo que representa um cliente para predição de churn."""

    meses_permanencia: int
    receita_mensal: float
    receita_total: float
    utiliza_servicos_financeiros: str
    possui_contador: str
    faz_conciliacao_bancaria: str


class PredictRequest(BaseModel):
    """Estrutura da requisição de predição contendo uma lista de clientes."""

    data: List[Customer]


@app.get("/")
def read_root():
    """Endpoint de health check da API."""
    return {"message": "Churn Prediction API is running 🚀"}


@app.post("/predict")
def predict(request: PredictRequest):
    """Recebe dados de clientes e retorna predição e probabilidades de churn."""
    try:
        df_input = pd.DataFrame(
            [item if isinstance(item, dict) else item.model_dump() for item in request.data]
        )

        model = get_model()
        preds = model.predict(df_input)

        probas = model.predict_proba(df_input)
        probas = np.array(probas)[:, 1]  # pega apenas a probabilidade da classe positiva

        results = [
            {"prediction": int(pred), "probability": float(proba)}
            for pred, proba in zip(preds, probas)
        ]
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição: {e}",
        ) from e
