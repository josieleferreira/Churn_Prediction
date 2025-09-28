from fastapi.testclient import TestClient
from app.main import app
import app.main as main
import numpy as np

client = TestClient(app)

def test_root_endpoint():
    """Testa se a API responde na raiz."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_endpoint(monkeypatch):
    """Testa se o endpoint de predição responde corretamente sem depender do modelo real."""

    # Mock do modelo
    class DummyModel:
        def predict(self, X):
            return [1 for _ in range(len(X))]
        def predict_proba(self, X):
            return np.array([[0.2, 0.8] for _ in range(len(X))])

    # Substitui o modelo real pelo Dummy
    monkeypatch.setattr(main, "get_model", lambda: DummyModel())

    sample_input = {
        "data": [
            {
                "meses_permanencia": 12,
                "receita_mensal": 1000.0,
                "receita_total": 12000.0,
                "utiliza_servicos_financeiros": "Sim",
                "possui_contador": "Não",
                "faz_conciliacao_bancaria": "automática"
            }
        ]
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert body["results"][0]["prediction"] in [0, 1]
    assert 0.0 <= body["results"][0]["probability"] <= 1.0
