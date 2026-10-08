from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from features import apply_encoder

app = FastAPI()

model = joblib.load("final_model.joblib")
encoder = joblib.load("encoder.joblib")


class ShipmentInput(BaseModel):
    # These two dominate the prediction per your SHAP chart — no default,
    # you should always set these deliberately when testing.
    shipping_mode: str
    days_for_shipment_scheduled: int

    # Everything below has a default, based on your SHAP chart showing
    # these contribute only marginally. Override any of them if you want,
    # otherwise /docs will demo cleanly with just the two fields above.
    order_region: str = "Western Europe"
    category_name: str = "Cleats"
    department_name: str = "Apparel"
    customer_segment: str = "Consumer"
    payment_type: str = "DEBIT"
    market: str = "Europe"
    order_item_quantity: int = 1
    order_item_discount: float = 0.0
    order_item_discount_rate: float = 0.0
    product_price: float = 100.0
    benefit_per_order: float = 20.0
    latitude: float = 25.0
    longitude: float = 0.0
    order_weekday: int = 2
    order_month: int = 6


rename_map = {
    "shipping_mode": "Shipping Mode", "order_region": "Order Region",
    "category_name": "Category Name", "department_name": "Department Name",
    "customer_segment": "Customer Segment", "payment_type": "Type",
    "market": "Market", "order_item_quantity": "Order Item Quantity",
    "order_item_discount": "Order Item Discount",
    "order_item_discount_rate": "Order Item Discount Rate",
    "product_price": "Product Price", "benefit_per_order": "Benefit per order",
    "latitude": "Latitude", "longitude": "Longitude",
    "days_for_shipment_scheduled": "Days for shipment (scheduled)",
    "order_weekday": "order_weekday", "order_month": "order_month",
}


@app.post("/predict")
def predict(shipment: ShipmentInput):
    input_df = pd.DataFrame([shipment.model_dump()])
    input_df = input_df.rename(columns=rename_map)

    input_encoded = apply_encoder(input_df, encoder)
    prediction = model.predict(input_encoded)
    delay = float(prediction[0])

    return {
        "predicted_delay_days": round(delay, 2),
        "is_late": delay > 0,
    }