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
from pydantic import BaseModel, Field

app = FastAPI(title="Churn Prediction API")

# Caminho do modelo
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "notebook", "pipeline_churn.pkl"
)
MODEL_PATH = os.path.abspath(MODEL_PATH)


@lru_cache(maxsize=1)
def get_model():
    """Carrega o modelo somente uma vez (lazy loading com cache)."""
    return joblib.load(MODEL_PATH)


class Customer(BaseModel):
    """Modelo que representa um cliente para predição de churn (nomes amigáveis)."""

    meses_permanencia: int = Field(..., example=12)
    receita_mensal: float = Field(..., example=1500.0)
    receita_total: float = Field(..., example=18000.0)
    tipo_de_empresa: str = Field(..., example="SaaS")
    contrato: str = Field(..., example="Mensal")
    emite_boletos: str = Field(..., example="Sim")
    fundacao_da_empresa: int = Field(..., example=2015)
    utiliza_servicos_financeiros: str = Field(..., example="Não")
    possui_contador: str = Field(..., example="Sim")
    faz_conciliacao_bancaria: str = Field(..., example="Automática")


class PredictRequest(BaseModel):
    """Estrutura da requisição de predição contendo uma lista de clientes."""

    data: List[Customer]

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "meses_permanencia": 12,
                        "receita_mensal": 1500.0,
                        "receita_total": 18000.0,
                        "tipo_de_empresa": "SaaS",
                        "contrato": "Mensal",
                        "emite_boletos": "Sim",
                        "fundacao_da_empresa": 2015,
                        "utiliza_servicos_financeiros": "Não",
                        "possui_contador": "Sim",
                        "faz_conciliacao_bancaria": "Automática",
                    }
                ]
            }
        }


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

        # 🔹 Mapeamento entre nomes amigáveis (API) e nomes originais (pipeline salvo)
        column_map = {
            "meses_permanencia": "Meses de permanência ",
            "receita_mensal": "Receita mensal",
            "receita_total": "Receita total",
            "tipo_de_empresa": "Tipo de empresa",
            "contrato": "Contrato",
            "emite_boletos": "Emite boletos",
            "fundacao_da_empresa": "Fundação da empresa",
            "utiliza_servicos_financeiros": "Utiliza serviços financeiros",
            "possui_contador": "PossuiContador",
            "faz_conciliacao_bancaria": "Faz conciliação bancária",
        }

        # Renomear colunas vindas da API
        df_input.rename(columns=column_map, inplace=True)

        # 🔹 Garantir que todas as colunas esperadas pelo modelo existam
        model = get_model()
        expected_features = model.feature_names_in_

        for col in expected_features:
            if col not in df_input.columns:
                df_input[col] = np.nan  # deixa o pipeline imputar corretamente

        # 🔹 Ajustar tipos de dados
        numeric_cols = ["Meses de permanência ", "Receita mensal", "Receita total", "Fundação da empresa"]
        categorical_cols = [col for col in expected_features if col not in numeric_cols]

        for col in numeric_cols:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

        for col in categorical_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)

        # Reordenar colunas na ordem que o modelo espera
        df_input = df_input[expected_features]

        # Predição
        preds = model.predict(df_input)
        probas = model.predict_proba(df_input)[:, 1]  # prob da classe positiva

        # 🔹 Mapear saída: 1 → "Sim", 0 → "Não"
        predictions = ["Sim" if p == 1 else "Não" for p in preds]
        probabilities = [round(float(p), 2) for p in probas]

        return {"predictions": predictions, "probabilities": probabilities}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição: {e}",
        ) from e
