"""Contract tests: invalid payloads are rejected with 422, valid ones return
a well-typed response. Uses TestClient so no running server is needed."""
import pytest
from fastapi.testclient import TestClient

from services.scoring.main import app as scoring_app


@pytest.fixture
def client():
    return TestClient(scoring_app)


def test_negative_amount_rejected(client):
    resp = client.post("/score", json={
        "txn_id": "T1", "step": 1, "type": "TRANSFER", "amount": -500,
        "nameOrig": "C1", "oldbalanceOrg": 1000, "nameDest": "C2", "oldbalanceDest": 0,
    })
    assert resp.status_code == 422


def test_invalid_type_rejected(client):
    resp = client.post("/score", json={
        "txn_id": "T1", "step": 1, "type": "PAYMENT", "amount": 500,
        "nameOrig": "C1", "oldbalanceOrg": 1000, "nameDest": "C2", "oldbalanceDest": 0,
    })
    assert resp.status_code == 422


def test_missing_field_rejected(client):
    resp = client.post("/score", json={"txn_id": "T1", "amount": 500})
    assert resp.status_code == 422
