"""
Módulo principal da API de Churn Prediction usando FastAPI.
Fornece endpoints para health check e predição de churn a partir de dados de clientes.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API para predição de churn de clientes",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "notebook" / "pipeline_churn.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parent / "churn_prediction.pkl"


@lru_cache(maxsize=1)
def get_model():
    """Carrega o modelo somente uma vez (lazy loading com cache)."""
    return joblib.load(str(MODEL_PATH))


class Customer(BaseModel):
    """Modelo que representa um cliente para predição de churn (nomes amigáveis)."""
    meses_permanencia: int = Field(..., alias="meses_permanencia", example=12)
    receita_mensal: float = Field(..., alias="receita_mensal", example=1500.0)
    receita_total: float = Field(..., alias="receita_total", example=18000.0)
    tipo_de_empresa: str = Field(..., alias="tipo_de_empresa", example="SaaS")
    contrato: str = Field(..., alias="contrato", example="Mensal")
    emite_boletos: str = Field(..., alias="emite_boletos", example="Sim")
    fundacao_da_empresa: int = Field(..., alias="fundacao_da_empresa", example=2015)
    utiliza_servicos_financeiros: str = Field(..., alias="utiliza_servicos_financeiros", example="Não")
    possui_contador: str = Field(..., alias="possui_contador", example="Sim")
    faz_conciliacao_bancaria: str = Field(..., alias="faz_conciliacao_bancaria", example="Automática")

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    """Estrutura da requisição de predição contendo uma lista de clientes."""
    data: List[Customer]

    model_config = {
        "json_schema_extra": {
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
    }


@app.get("/")
def read_root():
    """Endpoint de health check da API."""
    return {"message": "Churn Prediction API is running 🚀"}


@app.get("/health")
def health():
    """Verifica se a API e o modelo estão disponíveis."""
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except Exception:
        raise HTTPException(status_code=503, detail="Modelo não carregado")


@app.post("/predict")
async def predict(request: Request):
    """Recebe dados de clientes e retorna predição e probabilidades de churn."""
    try:
        try:
            body = await request.json()
        except Exception:
            body = {
                "data": [
                    {
                        "meses_permanencia": 24,
                        "receita_mensal": 500.0,
                        "receita_total": 12000.0,
                        "tipo_de_empresa": "SaaS",
                        "contrato": "Mensal",
                        "emite_boletos": "Sim",
                        "fundacao_da_empresa": 2018,
                        "utiliza_servicos_financeiros": "Não",
                        "possui_contador": "Sim",
                        "faz_conciliacao_bancaria": "Manual",
                    }
                ]
            }

        normalized_data = []
        for item in body.get("data", []):
            normalized_item = {k.strip(): v for k, v in item.items()}
            normalized_data.append(normalized_item)

        parsed_request = PredictRequest(data=[Customer(**item) for item in normalized_data])
        df_input = pd.DataFrame([cust.model_dump(by_alias=False) for cust in parsed_request.data])

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
        df_input.rename(columns=column_map, inplace=True)

        model = get_model()
        expected_features = model.feature_names_in_

        for col in expected_features:
            if col not in df_input.columns:
                df_input[col] = np.nan

        numeric_cols = ["Meses de permanência ", "Receita mensal", "Receita total", "Fundação da empresa"]
        categorical_cols = [col for col in expected_features if col not in numeric_cols]

        for col in numeric_cols:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

        for col in categorical_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)

        df_input = df_input[expected_features]

        preds = model.predict(df_input)
        probas = model.predict_proba(df_input)[:, 1]

        results = [
            {"prediction": int(pred), "probability": round(float(proba), 2)}
            for pred, proba in zip(preds, probas)
        ]
        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro na predição")
        raise HTTPException(
            status_code=500,
            detail="Erro ao processar predição. Verifique os dados enviados.",
        ) from e
