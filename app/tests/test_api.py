from fastapi.testclient import TestClient
import numpy as np
import app.main as main

client = TestClient(main.app)


def test_predict_endpoint_with_missing_columns(monkeypatch):
    """Testa se o endpoint /predict funciona mesmo quando o modelo espera mais colunas do que a API fornece."""

    class DummyModel:
        # Modelo "espera" colunas extras
        feature_names_in_ = np.array([
            "Meses de permanência ",
            "Receita mensal",
            "Receita total",
            "Tipo de empresa",
            "Contrato",
            "Emite boletos",
            "Fundação da empresa",
            "Utiliza serviços financeiros",
            "PossuiContador",
            "Faz conciliação bancária",
            "Coluna_extra_1",
            "Coluna_extra_2",
        ])

        def predict(self, X):
            return [1 for _ in range(len(X))]

        def predict_proba(self, X):
            return np.array([[0.3, 0.7] for _ in range(len(X))])

    # Monkeypatch substitui o get_model
    monkeypatch.setattr(main, "get_model", lambda: DummyModel())

    sample_input = {
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
                "possui_contador ": "Sim",
                "faz_conciliacao_bancaria": "Manual"
            }
        ]
    }

    response = client.post("/predict", json=sample_input)

    # ✅ O endpoint deve responder sem erro
    assert response.status_code == 200

    body = response.json()
    assert "results" in body
    assert body["results"][0]["prediction"] in [0, 1]
    assert 0.0 <= body["results"][0]["probability"] <= 1.0
