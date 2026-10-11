"""Focused tests for the SupplyGuard data/feature contract."""
import pandas as pd
import pytest

from Data import FEATURE_COLS, TARGET_COL, split_by_date
from features import build_preprocessor, validate_features


def sample_frame(rows=10):
    data = {}
    category_columns = [
        "Shipping Mode", "Market", "Order Region", "Category Name",
        "Department Name", "Customer Segment", "Type",
    ]
    numeric_columns = [
        "Days for shipment (scheduled)", "Order Item Quantity",
        "Order Item Discount", "Order Item Discount Rate", "Product Price",
        "Benefit per order", "Latitude", "Longitude", "order_weekday", "order_month",
    ]
    for column in category_columns:
        data[column] = ["example"] * rows
    for column in numeric_columns:
        data[column] = [1.0] * rows
    data[TARGET_COL] = list(range(rows))
    data["order date (DateOrders)"] = pd.date_range("2024-01-01", periods=rows, freq="D").astype(str)
    return pd.DataFrame(data)


def test_validate_features_orders_columns_and_rejects_missing():
    frame = pd.DataFrame({name: [0] for name in reversed(FEATURE_COLS)})
    assert list(validate_features(frame).columns) == FEATURE_COLS
    with pytest.raises(ValueError, match="Missing required model features"):
        validate_features(frame.drop(columns=[FEATURE_COLS[0]]))


def test_chronological_split_keeps_target_separate():
    frame = sample_frame()
    frame[TARGET_COL] = list(range(-5, 5))
    X_train, X_test, y_train, y_test = split_by_date(frame, test_size=0.2)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert list(X_train.columns) == FEATURE_COLS
    assert TARGET_COL not in X_train
    assert y_train.tolist() == list(range(-5, 3))
    assert y_test.tolist() == [3, 4]


def test_preprocessor_handles_unseen_category_and_missing_numeric():
    train = validate_features(sample_frame(2))
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(train)
    later = train.iloc[[0]].copy()
    later.loc[later.index[0], "Shipping Mode"] = "Unknown category"
    later.loc[later.index[0], "Product Price"] = float("nan")
    prediction_features = preprocessor.transform(later)
    assert transformed.shape[1] == prediction_features.shape[1]
    assert prediction_features.shape[0] == 1
