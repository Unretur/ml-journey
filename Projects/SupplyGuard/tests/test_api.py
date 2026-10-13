"""API contract tests which do not require a trained artifact."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import api
from Data import FEATURE_COLS
from api import RENAME_MAP, ShipmentInput, app


client = TestClient(app)


def valid_payload():
    return {
        "shipping_mode": "Standard Class",
        "market": "USCA",
        "order_region": "West of USA",
        "category_name": "Sporting Goods",
        "department_name": "Fitness",
        "customer_segment": "Consumer",
        "payment_type": "Debit",
        "days_for_shipment_scheduled": 3,
        "order_item_quantity": 2,
        "order_item_discount": 0.1,
        "order_item_discount_rate": 0.05,
        "product_price": 35.0,
        "benefit_per_order": 10.0,
        "latitude": 34.0,
        "longitude": -118.0,
        "order_weekday": 2,
        "order_month": 6,
    }


def test_request_maps_every_feature_and_uses_expected_names():
    payload = valid_payload()
    request = ShipmentInput(**payload)
    frame = pd.DataFrame([request.model_dump()]).rename(columns=RENAME_MAP)
    assert list(frame.columns) == FEATURE_COLS
    assert frame.loc[0, "Type"] == "Debit"
    assert frame.loc[0, "Days for shipment (scheduled)"] == 3


def test_request_rejects_unknown_fields():
    payload = valid_payload()
    payload["unexpected"] = "not allowed"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_request_rejects_nan_or_infinite_values():
    payload = valid_payload()
    payload["product_price"] = float("inf")
    # Starlette/httpx rejects non-finite JSON before sending the request. Validate
    # the Pydantic model directly to test the API's schema-level guard.
    import pytest
    with pytest.raises(Exception):
        ShipmentInput(**payload)


def test_health_is_liveness_and_ready_requires_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "_pipeline", None)

    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 503


def test_prediction_uses_shared_preprocessor_and_returns_estimate(monkeypatch):
    class FakePreprocessor:
        def transform(self, frame):
            assert list(frame.columns) == FEATURE_COLS
            return frame

    class FakeModel:
        def predict(self, frame):
            assert len(frame) == 1
            return np.array([1.25])

    monkeypatch.setattr(
        api,
        "get_pipeline",
        lambda: {
            "preprocessor": FakePreprocessor(),
            "model": FakeModel(),
            "feature_columns": FEATURE_COLS,
            "target": "delay_days",
            "selected_model": "test-model",
        },
    )
    monkeypatch.setattr(api, "_pipeline", {"test": True})
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_delay_days"] == 1.25
    assert body["predicted_status"] == "late"
    assert body["target_definition"] == "actual shipping days - scheduled shipping days"
