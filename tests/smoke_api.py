from __future__ import annotations

from fastapi.testclient import TestClient
from app.main import app


def test_health_and_models():
    client = TestClient(app)
    assert client.get('/health').status_code == 200
    assert client.get('/v1/models').status_code == 200
