from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Churn Prediction API")

# Caminho do modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_prediction.pkl")


# Lazy loading do modelo
def get_model():
    """Carrega o modelo somente quando necessário."""
    global model
    if "model" not in globals():
        model = joblib.load(MODEL_PATH)
    return model


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


# Endpoint raiz (health check)
@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running 🚀"}


# Endpoint de predição
@app.post("/predict")
def predict(request: PredictRequest):
    try:
        df_input = pd.DataFrame(
            [item if isinstance(item, dict) else item.model_dump() for item in request.data]
        )

        model = get_model()
        preds = model.predict(df_input)

        probas = model.predict_proba(df_input)
        probas = np.array(probas)  # 🔹 garante que vira array
        probas = probas[:, 1]

        results = [
            {"prediction": int(pred), "probability": float(proba)}
            for pred, proba in zip(preds, probas)
        ]
        return {"results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

