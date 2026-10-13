"""SupplyGuard AI regression API."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from Data import FEATURE_COLS
from features import validate_features

MODEL_PATH = Path(
    os.getenv("SUPPLYGUARD_MODEL_PATH", "artifacts/supplyguard_pipeline.joblib")
)
app = FastAPI(
    title="SupplyGuard API",
    description=(
        "Predicts actual shipping duration minus scheduled shipping duration. "
        "Predictions are historical estimates, not delivery guarantees."
    ),
    version="2.1.0",
)
_pipeline: dict[str, Any] | None = None


def get_pipeline() -> dict[str, Any]:
    """Load and validate the trusted model artifact once per process."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run train.py before prediction."
        )
    loaded = joblib.load(MODEL_PATH)
    required_keys = {"preprocessor", "model", "feature_columns"}
    if not isinstance(loaded, dict) or not required_keys.issubset(loaded):
        raise ValueError("Unsupported model artifact. Retrain with the current train.py.")
    columns = loaded["feature_columns"]
    if not isinstance(columns, list) or columns != FEATURE_COLS:
        raise ValueError("Saved model feature schema differs from API schema; retrain.")
    if not callable(getattr(loaded["preprocessor"], "transform", None)):
        raise ValueError("Model artifact has no usable preprocessor; retrain.")
    if not callable(getattr(loaded["model"], "predict", None)):
        raise ValueError("Model artifact has no usable predictor; retrain.")
    _pipeline = loaded
    return _pipeline


class ShipmentInput(BaseModel):
    """Required predictor fields; inputs are not silently invented."""
    model_config = ConfigDict(extra="forbid")

    shipping_mode: str = Field(min_length=1)
    market: str = Field(min_length=1)
    order_region: str = Field(min_length=1)
    category_name: str = Field(min_length=1)
    department_name: str = Field(min_length=1)
    customer_segment: str = Field(min_length=1)
    payment_type: str = Field(min_length=1)
    days_for_shipment_scheduled: float = Field(ge=0, allow_inf_nan=False)
    order_item_quantity: float = Field(ge=0, allow_inf_nan=False)
    order_item_discount: float = Field(allow_inf_nan=False)
    order_item_discount_rate: float = Field(allow_inf_nan=False)
    product_price: float = Field(ge=0, allow_inf_nan=False)
    benefit_per_order: float = Field(allow_inf_nan=False)
    latitude: float = Field(allow_inf_nan=False)
    longitude: float = Field(allow_inf_nan=False)
    order_weekday: int = Field(ge=0, le=6)
    order_month: int = Field(ge=1, le=12)


RENAME_MAP = {
    "shipping_mode": "Shipping Mode",
    "market": "Market",
    "order_region": "Order Region",
    "category_name": "Category Name",
    "department_name": "Department Name",
    "customer_segment": "Customer Segment",
    "payment_type": "Type",
    "days_for_shipment_scheduled": "Days for shipment (scheduled)",
    "order_item_quantity": "Order Item Quantity",
    "order_item_discount": "Order Item Discount",
    "order_item_discount_rate": "Order Item Discount Rate",
    "product_price": "Product Price",
    "benefit_per_order": "Benefit per order",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "order_weekday": "order_weekday",
    "order_month": "order_month",
}


def _artifact_error() -> HTTPException:
    """Return a consistent service-unavailable error for missing/bad artifacts."""
    global _pipeline
    _pipeline = None
    try:
        get_pipeline()
    except (FileNotFoundError, ValueError) as exc:
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=503, detail="Model artifact is unavailable.")


@app.get("/health")
def health():
    """Liveness check; does not load a potentially large model."""
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    """Readiness check: report whether the configured artifact can be loaded."""
    try:
        artifact = get_pipeline()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ready",
        "model_loaded": True,
        "model": artifact.get("selected_model", "unknown"),
    }


@app.get("/model-info")
def model_info():
    try:
        artifact = get_pipeline()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "model": artifact.get("selected_model", "unknown"),
        "target": artifact.get("target", "delay_days"),
        "feature_count": len(artifact["feature_columns"]),
    }


@app.post("/predict")
def predict(shipment: ShipmentInput):
    input_df = pd.DataFrame([shipment.model_dump()]).rename(columns=RENAME_MAP)
    try:
        artifact = get_pipeline()
        input_df = validate_features(input_df)
        transformed = artifact["preprocessor"].transform(input_df)
        predicted_delay = float(artifact["model"].predict(transformed)[0])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=503 if isinstance(exc, (FileNotFoundError,)) else 422,
            detail=f"Unable to prepare prediction: {exc}",
        ) from exc
    except Exception as exc:
        # Surface a clear service error without exposing internal stack traces.
        raise HTTPException(
            status_code=500, detail="Prediction failed; inspect server logs."
        ) from exc

    return {
        "predicted_delay_days": round(predicted_delay, 3),
        "predicted_status": (
            "late" if predicted_delay >= 0.5 else
            "early" if predicted_delay <= -0.5 else
            "near_schedule"
        ),
        "target_definition": "actual shipping days - scheduled shipping days",
        "note": "Historical model estimate, not a guaranteed delivery outcome.",
    }
