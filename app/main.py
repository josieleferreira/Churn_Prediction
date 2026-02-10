import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuração de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API para predição de churn de clientes",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).resolve().parent / "churn_prediction.pkl"
model = None

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo carregado com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar modelo: {e}")
    model = None


def _get_expected_columns():
    """Extrai colunas esperadas do preprocessor do modelo."""
    if model is None:
        return []
    try:
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_in"):
            return list(preprocessor.get_feature_names_in())
    except Exception:
        pass
    return []


# Schema: aceita registro flexível (dict) para compatibilidade com o pipeline do notebook
class CustomerInput(BaseModel):
    """Registro de cliente. Use os mesmos nomes de colunas do dataset de treino."""

    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    data: list[dict] = Field(
        ...,
        description="Lista de registros de clientes com as features do modelo",
        min_length=1,
    )


class PredictResult(BaseModel):
    prediction: int = Field(..., description="0=Não churn, 1=Churn")
    probability: float = Field(..., description="Probabilidade de churn", ge=0, le=1)


@app.get("/health")
def health():
    """Verifica se a API e o modelo estão disponíveis."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=dict)
def predict(request: PredictRequest):
    """Retorna predições de churn para uma lista de clientes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível")

    try:
        df_input = pd.DataFrame(request.data)

        if df_input.empty:
            raise HTTPException(status_code=400, detail="Lista de dados vazia")

        preds = model.predict(df_input)
        probas = model.predict_proba(df_input)[:, 1]

        results = [
            PredictResult(prediction=int(pred), probability=float(proba))
            for pred, proba in zip(preds, probas)
        ]
        return {"results": [r.model_dump() for r in results]}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro na predição")
        raise HTTPException(
            status_code=500,
            detail="Erro ao processar predição. Verifique se os dados têm as colunas esperadas pelo modelo.",
        ) from e
