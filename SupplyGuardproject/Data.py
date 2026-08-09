"""
SupplyGuard AI — data.py

Loads the raw DataCo CSV, drops the columns we decided don't belong,
builds the delay_days target, pulls weekday/month out of the order date,
and splits the data by date (not randomly) into train and test.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_clean_data(path):
    """Load the raw CSV and return a cleaned dataframe with delay_days built."""
    df = pd.read_csv(path, encoding="latin-1")

    df["delay_days"] = (
        df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
    )

    # pull weekday/month out of the date BEFORE we drop the raw date column
    df["order_weekday"] = pd.to_datetime(df["order date (DateOrders)"]).dt.dayofweek
    df["order_month"] = pd.to_datetime(df["order date (DateOrders)"]).dt.month

    df = df.drop(columns=[
        # leakage
        "Days for shipping (real)", "Delivery Status", "Late_delivery_risk",
        "shipping date (DateOrders)",
        # constants / empty / mostly-missing
        "Customer Email", "Customer Password", "Product Status",
        "Product Description", "Order Zipcode",
        # PII / no causal link to delay
        "Customer Fname", "Customer Lname", "Customer Street",
        "Customer Zipcode", "Product Image",
        # pure identifiers
        "Order Item Id", "Order Id", "Customer Id", "Order Customer Id",
        # exact duplicate columns
        "Product Category Id", "Category Id", "Order Item Cardprod Id",
        "Product Card Id", "Order Item Total", "Order Profit Per Order",
        "Order Item Product Price",
        # high-cardinality geography
        "Customer City", "Order City", "Customer State", "Order State",
        "Order Country", "Customer Country",
        # NOTE: "order date (DateOrders)" is NOT dropped here on purpose —
        # split_by_date() still needs it to sort chronologically. It gets
        # dropped after splitting, inside split_by_date().
    ])

    return df


def split_by_date(df, date_col="order date (DateOrders)", test_size=0.2):
    """Sort by the real date column, split without shuffling, then drop the
    raw date column now that it's done its job."""
    df = df.sort_values(date_col)

    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)

    y_train = train_df["delay_days"]
    y_test = test_df["delay_days"]
    X_train = train_df.drop(columns=["delay_days", date_col])
    X_test = test_df.drop(columns=["delay_days", date_col])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_clean_data("DataCoSupplyChainDataset.csv")
    print("Cleaned shape:", df.shape)
    print(df["delay_days"].describe())

    X_train, X_test, y_train, y_test = split_by_date(df)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print(X_train[["order_weekday", "order_month"]].head())