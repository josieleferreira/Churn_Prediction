import pytest
from src.models import predict

def test_predict_function_exists():
    """Verifica se o módulo de predição tem a função correta."""
    assert hasattr(predict, "load_model"), "O módulo predict deve ter a função load_model"

def test_predict_model_load(monkeypatch):
    """Mocka o carregamento de modelo para garantir que a função load_model funciona."""
    
    class DummyModel:
        def predict(self, X):
            return [1 for _ in X]

    # Monkeypatch para substituir o joblib.load
    monkeypatch.setattr("joblib.load", lambda _: DummyModel())

    model = predict.load_model()
    result = model.predict([[0, 1, 2]])
    assert result == [1], "O modelo dummy deve sempre retornar 1"
