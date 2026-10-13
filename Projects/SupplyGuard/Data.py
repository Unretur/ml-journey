"""
SupplyGuard data preparation.

Target:
    delay_days = actual shipping duration - scheduled shipping duration

Positive = late; zero = on schedule; negative = earlier than scheduled.

Important: this dataset is retrospective. Define the intended prediction
timestamp before using the model operationally, and remove any feature not
known at that point.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATE_COL = "order date (DateOrders)"
TARGET_COL = "delay_days"
CATEGORICAL_COLS = [
    "Shipping Mode", "Market", "Order Region",
    "Category Name", "Department Name", "Customer Segment", "Type",
]
NUMERIC_COLS = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Product Price",
    "Benefit per order",
    "Latitude",
    "Longitude",
    "order_weekday",
    "order_month",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

DROP_COLUMNS = [
    "Days for shipping (real)", "Delivery Status", "Late_delivery_risk",
    "shipping date (DateOrders)", "Order Status", "Product Name",
    "Customer Email", "Customer Password", "Product Status",
    "Product Description", "Order Zipcode", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode", "Product Image",
    "Order Item Id", "Order Id", "Customer Id", "Order Customer Id",
    "Product Category Id", "Category Id", "Order Item Cardprod Id",
    "Product Card Id", "Order Item Total", "Order Profit Per Order",
    "Order Item Product Price", "Customer City", "Order City", "Customer State",
    "Order State", "Order Country", "Customer Country",
]


def load_clean_data(path: str | Path) -> pd.DataFrame:
    """Load and clean DataCo CSV while preserving only documented model inputs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Pass --data with the CSV path."
        )

    df = pd.read_csv(path, encoding="latin-1")
    required = {
        "Days for shipping (real)", "Days for shipment (scheduled)", DATE_COL,
        *CATEGORICAL_COLS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    # Store parsed datetimes so chronological sorting is temporal, not lexical.
    df[DATE_COL] = dates
    real_days = pd.to_numeric(df["Days for shipping (real)"], errors="coerce")
    scheduled_days = pd.to_numeric(
        df["Days for shipment (scheduled)"], errors="coerce"
    )
    df[TARGET_COL] = real_days - scheduled_days
    df["order_weekday"] = dates.dt.dayofweek
    df["order_month"] = dates.dt.month

    # Avoid silently using fields that encode the realized outcome.
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna(subset=[DATE_COL, TARGET_COL])

    missing_features = sorted(set(FEATURE_COLS).difference(df.columns))
    if missing_features:
        raise ValueError(f"Dataset is missing model features: {missing_features}")

    # Numeric columns are normalized to numeric dtypes; bad values become NaN
    # and are handled by the preprocessing pipeline's imputer.
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(DATE_COL).reset_index(drop=True)


def split_by_date(
    df: pd.DataFrame, date_col: str = DATE_COL, test_size: float = 0.2
):
    """Return chronological train/test features and target; never shuffle."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(df) < 3:
        raise ValueError("Not enough rows for a chronological train/test split (need at least 3).")
    if date_col not in df.columns:
        raise ValueError(f"Date column {date_col!r} is missing.")
    ordered = df.sort_values(date_col).reset_index(drop=True)
    split_at = int(len(ordered) * (1 - test_size))
    if split_at < 1 or split_at >= len(ordered):
        raise ValueError("Not enough rows for the requested chronological split.")

    train_df, test_df = ordered.iloc[:split_at], ordered.iloc[split_at:]
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].copy()
    y_test = test_df[TARGET_COL].copy()
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate and summarize SupplyGuard data.")
    parser.add_argument("--data", default="data/DataCoSupplyChainDataset.csv")
    args = parser.parse_args()
    frame = load_clean_data(args.data)
    X_train, X_test, y_train, y_test = split_by_date(frame)
    print(f"Rows: {len(frame):,}; features: {len(FEATURE_COLS)}")
    print(f"Train: {len(X_train):,}; test: {len(X_test):,}")
    print(f"Train dates: {frame[DATE_COL].min()} through {frame.iloc[len(X_train)-1][DATE_COL]}")
    print(f"Target summary:\n{y_train.describe()}")
