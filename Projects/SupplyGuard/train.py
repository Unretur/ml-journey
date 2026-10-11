"""Train and select SupplyGuard delay regressors without tuning on the test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from Data import load_clean_data, split_by_date
from features import build_preprocessor, validate_features


def metrics(y_true, predictions):
    return {
        "mae_days": float(mean_absolute_error(y_true, predictions)),
        "rmse_days": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def chronological_validation(X, y, validation_fraction=0.2):
    split = int(len(X) * (1 - validation_fraction))
    if split < 1 or split >= len(X):
        raise ValueError("Not enough training rows for chronological validation.")
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def optuna_objective(trial, X_train, y_train, X_valid, y_valid, seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 700),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "n_jobs": -1,
        "random_state": seed,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }
    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X_train, y_train)
    Xv = preprocessor.transform(X_valid)
    model = XGBRegressor(**params)
    model.fit(Xt, y_train, verbose=False)
    return mean_absolute_error(y_valid, model.predict(Xv))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/DataCoSupplyChainDataset.csv")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    frame = load_clean_data(args.data)
    X_train, X_test, y_train, y_test = split_by_date(frame)
    X_train, X_test = validate_features(X_train), validate_features(X_test)
    X_fit, X_valid, y_fit, y_valid = chronological_validation(X_train, y_train)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("supplyguard")

    candidates = {}
    with mlflow.start_run(run_name="median_baseline"):
        dummy = DummyRegressor(strategy="median")
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        baseline = metrics(y_test, dummy.predict(np.zeros((len(y_test), 1))))
        mlflow.log_metrics({f"test_{k}": v for k, v in baseline.items()})
        mlflow.log_param("model", "DummyRegressor_median")
        candidates["median_baseline"] = baseline
    print("Median baseline (chronological holdout):", json.dumps(baseline))

    # Validate several complementary tree models using the same earlier
    # validation window. The final holdout remains untouched until selection.
    for name, model in {
        "random_forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=args.seed
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=args.seed
        ),
    }.items():
        prep = build_preprocessor()
        Xfit = prep.fit_transform(X_fit, y_fit)
        Xvalid = prep.transform(X_valid)
        model.fit(Xfit, y_fit)
        score = mean_absolute_error(y_valid, model.predict(Xvalid))
        candidates[name] = {"validation_mae_days": float(score)}
        print(f"{name} validation MAE: {score:.4f} days")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(
        lambda trial: optuna_objective(trial, X_fit, y_fit, X_valid, y_valid, args.seed),
        n_trials=args.trials,
    )
    candidates["xgboost"] = {"validation_mae_days": float(study.best_value)}
    print(f"xgboost validation MAE: {study.best_value:.4f} days")

    # Evaluate the same median baseline on validation data for fair model
    # selection. Do not select a complex model if it fails to beat baseline.
    validation_baseline = DummyRegressor(strategy="median")
    validation_baseline.fit(np.zeros((len(y_fit), 1)), y_fit)
    validation_baseline_mae = mean_absolute_error(
        y_valid, validation_baseline.predict(np.zeros((len(y_valid), 1)))
    )
    candidate_maes = {k: v["validation_mae_days"] for k, v in candidates.items() if "validation_mae_days" in v}
    candidate_maes["median_baseline"] = float(validation_baseline_mae)
    winner_name = min(candidate_maes, key=candidate_maes.get)
    if winner_name == "median_baseline":
        raise RuntimeError(
            "No trained model beat the median baseline on chronological validation. "
            "Do not publish a model until the feature set/model improves."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor()
    Xtr = preprocessor.fit_transform(X_train, y_train)
    Xte = preprocessor.transform(X_test)
    if winner_name == "xgboost":
        model = XGBRegressor(
            **study.best_params, n_jobs=-1, random_state=args.seed,
            objective="reg:squarederror", tree_method="hist"
        )
    elif winner_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=args.seed
        )
    elif winner_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=args.seed
        )
    else:
        raise RuntimeError(f"Unsupported selected model: {winner_name}")
    model.fit(Xtr, y_train)
    test_result = metrics(y_test, model.predict(Xte))
    print(f"Selected by validation: {winner_name}")
    print("Final chronological holdout:", json.dumps(test_result, indent=2))

    joblib.dump({"preprocessor": preprocessor, "model": model, "feature_columns": list(X_train.columns),
                 "target": "delay_days", "selected_model": winner_name},
                output_dir / "supplyguard_pipeline.joblib")
    report = {
        "target": "actual shipping days - scheduled shipping days",
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "validation_rows": int(len(X_valid)), "selection_metric": "validation MAE",
        "selected_model": winner_name, "validation_mae_days": candidate_maes[winner_name],
        "median_baseline_test": baseline, "selected_model_test": test_result,
        "warning": "Test holdout is for final reporting only. Refrain from tuning against it."
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=f"selected_{winner_name}"):
        mlflow.log_param("selected_model", winner_name)
        mlflow.log_param("trials", args.trials)
        mlflow.log_param("seed", args.seed)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_result.items()})
        mlflow.log_metric("validation_mae_days", candidate_maes[winner_name])
        mlflow.log_dict(report, "metrics.json")


if __name__ == "__main__":
    main()
