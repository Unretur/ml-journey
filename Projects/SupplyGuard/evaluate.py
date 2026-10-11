"""Evaluate a trained SupplyGuard pipeline on the chronological holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Data import load_clean_data, split_by_date
from features import validate_features


def calculate_metrics(y_true, predictions):
    return {
        "mae_days": float(mean_absolute_error(y_true, predictions)),
        "rmse_days": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "within_1_day_pct": float(np.mean(np.abs(np.asarray(y_true) - predictions) <= 1.0) * 100),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/DataCoSupplyChainDataset.csv")
    parser.add_argument("--model", default="artifacts/supplyguard_pipeline.joblib")
    parser.add_argument("--report", default="artifacts/evaluation.json")
    args = parser.parse_args()

    artifact = Path(args.model)
    if not artifact.exists():
        raise FileNotFoundError(f"Model not found: {artifact}. Run train.py first.")

    payload = joblib.load(artifact)
    if not isinstance(payload, dict) or not {"preprocessor", "model", "feature_columns"} <= set(payload):
        raise ValueError("Model artifact has an unsupported format; retrain with current train.py.")

    frame = load_clean_data(args.data)
    X_train, X_test, y_train, y_test = split_by_date(frame)
    X_test = validate_features(X_test)
    if list(X_test.columns) != payload["feature_columns"]:
        raise ValueError("Feature schema mismatch between evaluation data and saved model.")

    Xtest = payload["preprocessor"].transform(X_test)
    predictions = payload["model"].predict(Xtest)
    baseline = DummyRegressor(strategy="median")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_predictions = baseline.predict(np.zeros((len(y_test), 1)))

    report = {
        "target_definition": "actual shipping days - scheduled shipping days",
        "test_rows": int(len(y_test)),
        "baseline_median": calculate_metrics(y_test, baseline_predictions),
        "model": payload.get("selected_model", "unknown"),
        "model_metrics": calculate_metrics(y_test, predictions),
        "error_mean_days": float(np.mean(np.asarray(y_test) - predictions)),
        "error_std_days": float(np.std(np.asarray(y_test) - predictions)),
        "interpretation": {
            "positive_prediction": "Predicted later than scheduled",
            "zero_prediction": "Predicted on schedule",
            "negative_prediction": "Predicted earlier than scheduled",
            "late_threshold": "Use > 0; business buffer/uncertainty should be considered separately."
        }
    }
    print(json.dumps(report, indent=2))
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
