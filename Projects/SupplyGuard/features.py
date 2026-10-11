"""Shared preprocessing for SupplyGuard training and inference."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Data import CATEGORICAL_COLS, NUMERIC_COLS, FEATURE_COLS


def build_preprocessor() -> ColumnTransformer:
    """Build one schema-aware preprocessing pipeline.

    handle_unknown='ignore' allows unseen categorical values at inference time;
    imputation prevents missing numeric values from breaking predictions.
    """
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_COLS),
            ("numeric", numeric, NUMERIC_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def validate_features(X: pd.DataFrame) -> pd.DataFrame:
    """Validate and order inputs identically for training and online inference."""
    missing = sorted(set(FEATURE_COLS).difference(X.columns))
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    # Selects a stable column order and rejects accidental use of target/IDs.
    return X.loc[:, FEATURE_COLS].copy()
