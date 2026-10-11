"""SupplyGuard AI regression API."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from Data import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS
from features import validate_features

MODEL_PATH = Path(os.getenv("SUPPLYGUARD_MODEL_PATH", "artifacts/supplyguard_pipeline.joblib"))
app = FastAPI(
    title="SupplyGuard API",
    description="Predicts actual shipping duration minus scheduled shipping duration.",
    version="2.0.0",
)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. Run train.py before prediction."
            )
        loaded = joblib.load(MODEL_PATH)
        if not isinstance(loaded, dict) or not {"preprocessor", "model", "feature_columns"} <= set(loaded):
            raise ValueError("Unsupported model artifact. Retrain with the current train.py.")
        if loaded["feature_columns"] != FEATURE_COLS:
            raise ValueError("Saved model feature schema differs from API schema; retrain.")
        _pipeline = loaded
    return _pipeline


class ShipmentInput(BaseModel):
    """All predictor fields are required; do not silently invent business inputs."""
    model_config = ConfigDict(extra="forbid")

    shipping_mode: str
    market: str
    order_region: str
    category_name: str
    department_name: str
    customer_segment: str
    payment_type: str
    days_for_shipment_scheduled: float = Field(ge=0)
    order_item_quantity: float = Field(ge=0)
    order_item_discount: float
    order_item_discount_rate: float
    product_price: float = Field(ge=0)
    benefit_per_order: float
    latitude: float
    longitude: float
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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/model-info")
def model_info():
    try:
        artifact = get_pipeline()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "model": artifact.get("selected_model", "unknown"),
        "target": artifact.get("target"),
        "feature_count": len(artifact["feature_columns"]),
    }


@app.post("/predict")
def predict(shipment: ShipmentInput):
    input_df = pd.DataFrame([shipment.model_dump()]).rename(columns=RENAME_MAP)
    try:
        artifact = get_pipeline()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        input_df = validate_features(input_df)
        transformed = artifact["preprocessor"].transform(input_df)
        predicted_delay = float(artifact["model"].predict(transformed)[0])
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Unable to prepare prediction: {exc}") from exc

    return {
        "predicted_delay_days": round(predicted_delay, 3),
        "predicted_status": (
            "late" if predicted_delay > 0.0 else
            "on_schedule" if abs(predicted_delay) < 0.5 else
            "early"
        ),
        "target_definition": "actual shipping days - scheduled shipping days",
        "note": "Historical model estimate, not a guaranteed delivery outcome.",
    }
