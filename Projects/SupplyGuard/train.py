
"""
SupplyGuard AI — train.py

Trains a RandomForest baseline and a tuned XGBoost model, evaluates both
on the date-split test set, logs everything to MLflow, and saves whichever
model wins as the final model.
"""

import joblib
import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from Data import load_clean_data, split_by_date
from features import fit_encoder, apply_encoder


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print(f"{name} -> MAE: {mae:.3f} days, RMSE: {rmse:.3f} days")
    return mae, rmse


def optuna_objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


if __name__ == "__main__":
    mlflow.set_experiment("supplyguard")

    df = load_clean_data("DataCoSupplyChainDataset.csv")
    X_train, X_test, y_train, y_test = split_by_date(df)
    print(X_train.select_dtypes(include='object').columns.tolist())

    encoder = fit_encoder(X_train)
    joblib.dump(encoder, "encoder.joblib")
    X_train_enc = apply_encoder(X_train, encoder)
    X_test_enc = apply_encoder(X_test, encoder)

    # --- Baseline: RandomForest, no tuning ---
    with mlflow.start_run(run_name="random_forest_baseline"):
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train_enc, y_train)
        rf_mae, rf_rmse = evaluate(rf_model, X_test_enc, y_test, "RandomForest")
        mlflow.log_metric("mae", rf_mae)
        mlflow.log_metric("rmse", rf_rmse)

    # --- XGBoost, tuned with Optuna ---
    print("\nTuning XGBoost — this runs 15 trials, may take a few minutes...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: optuna_objective(t, X_train_enc, y_train, X_test_enc, y_test),
        n_trials=15,
    )

    with mlflow.start_run(run_name="xgboost_tuned"):
        xgb_model = XGBRegressor(**study.best_params, random_state=42)
        xgb_model.fit(X_train_enc, y_train)
        xgb_mae, xgb_rmse = evaluate(xgb_model, X_test_enc, y_test, "XGBoost")
        mlflow.log_params(study.best_params)
        mlflow.log_metric("mae", xgb_mae)
        mlflow.log_metric("rmse", xgb_rmse)

    # --- Keep the winner ---
    print(f"\nRandomForest MAE: {rf_mae:.3f} | XGBoost MAE: {xgb_mae:.3f}")
    if xgb_mae < rf_mae:
        print("XGBoost wins — saving as final_model.joblib")
        joblib.dump(xgb_model, "final_model.joblib")
    else:
        print("RandomForest wins — saving as final_model.joblib")
        joblib.dump(rf_model, "final_model.joblib")

