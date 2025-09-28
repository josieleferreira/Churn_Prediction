from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import os

app = FastAPI(title="Churn Prediction API")

# Caminho do modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_prediction.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")


# Schema de entrada
class Customer(BaseModel):
    meses_permanencia: int
    receita_mensal: float
    receita_total: float
    utiliza_servicos_financeiros: str
    possui_contador: str
    faz_conciliacao_bancaria: str


class PredictRequest(BaseModel):
    data: List[Customer]


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        df_input = pd.DataFrame([item.dict() for item in request.data])

        preds = model.predict(df_input)
        probas = model.predict_proba(df_input)[:, 1]

        results = [
            {"prediction": int(pred), "probability": float(proba)}
            for pred, proba in zip(preds, probas)
        ]
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")
