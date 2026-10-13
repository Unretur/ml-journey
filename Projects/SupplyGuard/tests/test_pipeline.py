"""Focused tests for the SupplyGuard data/feature contract."""
import pandas as pd
import pytest

from Data import (
    CATEGORICAL_COLS,
    DATE_COL,
    FEATURE_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    load_clean_data,
    split_by_date,
)
from features import build_preprocessor, validate_features


def sample_frame(rows=10):
    data = {}
    for column in CATEGORICAL_COLS:
        data[column] = ["example"] * rows
    for column in NUMERIC_COLS:
        data[column] = [1.0] * rows
    data[TARGET_COL] = list(range(rows))
    data[DATE_COL] = pd.date_range("2024-01-01", periods=rows, freq="D").astype(str)
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


def test_split_rejects_invalid_sizes_and_short_data():
    frame = sample_frame(2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_by_date(frame, test_size=1.0)
    with pytest.raises(ValueError, match="Not enough rows"):
        split_by_date(frame, test_size=0.2)


def test_preprocessor_handles_unseen_category_and_missing_numeric():
    train = validate_features(sample_frame(4))
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(train)
    later = train.iloc[[0]].copy()
    later.loc[later.index[0], "Shipping Mode"] = "Unknown category"
    later.loc[later.index[0], "Product Price"] = float("nan")
    prediction_features = preprocessor.transform(later)
    assert transformed.shape[1] == prediction_features.shape[1]
    assert prediction_features.shape[0] == 1


def test_loader_fails_clearly_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_clean_data(tmp_path / "missing.csv")


def test_loader_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"something": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_clean_data(path)
