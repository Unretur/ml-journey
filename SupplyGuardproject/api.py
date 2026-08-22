from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 

from features import apply_encoder

app = FastAPI()

model = joblib.load("final_model.joblib")
encoder = joblib.load("encoder.joblib")

class ShipmentInput(BaseModel):
    shipping_mode: str
    order_region: str
    category_name: str
    department_name: str
    customer_segment: str
    payment_type: str
    market: str
    order_item_quantity: int
    order_item_discount: float
    order_item_discount_rate: float
    product_price: float
    benefit_per_order: float
    latitude: float
    longitude: float
    days_for_shipment_scheduled: int
    order_weekday: int
    order_month: int

# NEW: tells FastAPI "when a POST request hits /predict, run this function"
@app.post("/predict")
def predict(shipment: ShipmentInput):
    # 1. Turn the incoming data into a one-row dataframe.
    #    Hint: shipment.dict() gives you a Python dictionary —
    #    pd.DataFrame([...]) turns a dict into a one-row dataframe.
    input_df =  pd.DataFrame([shipment.model_dump()])

    # 2. Rename the columns to match what your encoder was trained on.
    #    Your training data has columns like "Shipping Mode" (space, capital) —
    #    this dataframe has "shipping_mode". Use df.rename(columns={...})
    #    with this exact mapping:
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

    # 3. Apply the encoder — same function you already used in train.py
    input_encoded =apply_encoder(input_df,encoder)

    # 4. Predict, then return it as a dictionary (FastAPI turns this into JSON automatically)
    prediction = model.predict(input_encoded)
    return {"predicted_delay_days": float(prediction[0])}