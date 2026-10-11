"""Lightweight schema/target tests; these do not require the large dataset."""
import pandas as pd
import pytest

from Data import FEATURE_COLS, TARGET_COL, split_by_date
from features import build_preprocessor, validate_features


def test_validate_features_orders_columns_and_rejects_missing():
    frame = pd.DataFrame({name: [0] for name in reversed(FEATURE_COLS)})
    result = validate_features(frame)
    assert list(result.columns) == FEATURE_COLS

    with pytest.raises(ValueError, match="Missing required model features"):
        validate_features(frame.drop(columns=[FEATURE_COLS[0]]))


def test_chronological_split_keeps_target_separate():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = {col: ["category"] * 10 for col in [
        "Shipping Mode", "Market", "Order Region", "Category Name",
        "Department Name", "Customer Segment", "Type"
    ]}
    for col in [
        "Days for shipment (scheduled)", "Order Item Quantity",
        "Order Item Discount", "Order Item Discount Rate", "Product Price",
        "Benefit per order", "Latitude", "Longitude", "order_weekday", "order_month"
    ]:
        data[col] = [1.0] * 10
    data["delay_days"] = list(range(-5, 5))
    data["order date (DateOrders)"] = dates.astype(str)
    frame = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = split_by_date(frame, test_size=0.2)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert list(X_train.columns) == FEATURE_COLS
    assert TARGET_COL not in X_train
    assert y_train.tolist() == list(range(-5, 3))
    assert y_test.tolist() == [3, 4]


def test_preprocessor_handles_unseen_category_and_missing_numeric():
    train = pd.DataFrame({
        "Shipping Mode": ["First Class", "Standard Class"],
        "Market": ["Europe", "Europe"],
        "Order Region": ["Western Europe", "Western Europe"],
        "Category Name": ["Cleats", "Cleats"],
        "Department Name": ["Apparel", "Apparel"],
        "Customer Segment": ["Consumer", "Consumer"],
        "Type": ["DEBIT", "CASH"],
        "Days for shipment (scheduled)": [2.0, 3.0],
        "Order Item Quantity": [1.0, 2.0],
        "Order Item Discount": [0.0, 1.0],
        "Order Item Discount Rate": [0.0, 0.1],
        "Product Price": [10.0, 20.0],
        "Benefit per order": [2.0, 3.0],
        "Latitude": [10.0, 20.0],
        "Longitude": [30.0, 40.0],
        "order_weekday": [1, 2],
        "order_month": [1, 2],
    })[FEATURE_COLS]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(train)
    later = train.iloc[[0]].copy()
    later.loc[later.index[0], "Shipping Mode"] = "Unknown Mode"
    later.loc[later.index[0], "Product Price"] = float("nan")
    prediction_features = preprocessor.transform(later)
    assert transformed.shape[1] == prediction_features.shape[1]
    assert prediction_features.shape[0] == 1
