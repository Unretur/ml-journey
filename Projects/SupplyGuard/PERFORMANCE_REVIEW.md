# SupplyGuard AI — Performance Review

## What changed

This branch fixes evaluation and engineering issues identified during review:

1. **Test-set leakage in model selection:** the old Optuna objective repeatedly scored trials on `X_test`. Model selection now uses a chronological validation window taken from the training period; the final chronological holdout is evaluated after selection.
2. **Preprocessing drift between training and API:** training and inference now share a saved preprocessing pipeline. It imputes missing numeric values, one-hot encodes categorical features, ignores unseen categories, and enforces a stable feature schema.
3. **Target and API clarity:** the regression target remains actual shipping duration minus scheduled shipping duration. Positive predictions imply late, zero implies on schedule, negative implies early. The API no longer silently fills important business inputs with arbitrary defaults.
4. **Evaluation reporting:** MAE, RMSE, R², percentage within one day, and prediction error bias are reported against a median baseline.
5. **Reproducibility:** CLI arguments, random seeds, model artifacts, metrics JSON, dependency list, and focused schema tests have been added.

## Evidence from original code

- `train.py` optimized Optuna against `X_test`, so the test set was used for hyperparameter selection.
- The model was trained with one-hot encoded pandas columns, while `api.py` created partial input records and called `apply_encoder`; it did not guarantee every numeric training feature was present and correctly ordered.
- `evaluate.py` reported only baseline/model MAE and invoked SHAP directly, without a fuller metric report or schema verification.
- `is_late = prediction > 0` was directionally consistent with the target definition, but endpoint outputs lacked uncertainty communication and should not be interpreted as delivery guarantees.

## How to run

From `Projects/SupplyGuard/`:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

python -m pip install -r requirements.txt
python train.py --data data/DataCoSupplyChainDataset.csv --trials 25
python evaluate.py --data data/DataCoSupplyChainDataset.csv
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API schema. All model fields are required intentionally so predictions do not depend on hidden defaults.

## What has not been proven yet

The connector environment did not execute the training pipeline against the 95 MB dataset. Therefore no numeric performance improvement is claimed. Run training and evaluation locally, then compare `artifacts/metrics.json` with previous experiment results.

If MAE remains close to the median baseline, next steps are to define the prediction timestamp and remove any feature unavailable then; inspect residuals by shipping mode, region, and order time; add chronological rolling-origin validation; compare against late/on-time classification if that is the business goal; and assess whether this historical dataset contains enough operational predictors.

## Artifacts and safety

- New model is written to `artifacts/supplyguard_pipeline.joblib`; it does not overwrite the existing tracked model.
- Metrics are written to `artifacts/metrics.json`.
- Retrain before serving with the revised API: old `final_model.joblib` and `encoder.joblib` are incompatible with this pipeline.
- Do not treat retrospective results as evidence of real-world supply-chain savings. Validate on newer operational data before production use.
