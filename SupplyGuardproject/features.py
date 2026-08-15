"""
SupplyGuard AI — features.py

Turns the categorical (text) columns into numbers a model can use,
via One-Hot Encoding — fit once on train, reused everywhere else.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = [
    "Shipping Mode", "Market", "Order Region",
    "Category Name", "Department Name", "Customer Segment", "Type",
]


def fit_encoder(X_train):
    """Fit a OneHotEncoder on TRAIN data only, then return it."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS])
    return encoder


def apply_encoder(X, encoder):
    """Use an already-fitted encoder to transform any dataframe (train, test,
    or a single live shipment later)."""
    encoded = encoder.transform(X[CATEGORICAL_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=X.index,
    )
    X_rest = X.drop(columns=CATEGORICAL_COLS)
    return pd.concat([X_rest, encoded_df], axis=1)


if __name__ == "__main__":
    from Data import load_clean_data, split_by_date

    df = load_clean_data("DataCoSupplyChainDataset.csv")
    X_train, X_test, y_train, y_test = split_by_date(df)

    encoder = fit_encoder(X_train)          # fit ONLY on train
    joblib.dump(encoder, "encoder.joblib")   # save it so you never refit it

    X_train_enc = apply_encoder(X_train, encoder)
    X_test_enc = apply_encoder(X_test, encoder)

    print("Before encoding:", X_train.shape)
    print("After encoding:", X_train_enc.shape)
    print("Test after encoding:", X_test_enc.shape)
