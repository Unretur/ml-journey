# SupplyGuard AI

SupplyGuard predicts **shipment delay in days** using the DataCo Supply Chain dataset. The target is:

`delay_days = Days for shipping (real) - Days for shipment (scheduled)`

Positive values mean late delivery; zero means the actual delivery matched the scheduled duration; negative values mean earlier than scheduled. This is a retrospective dataset project, not a guarantee of real-world delivery performance.

## Project files

- `Data.py` — data loading, target construction, and chronological holdout split.
- `features.py` — shared preprocessing for categorical and numeric features.
- `train.py` — baseline comparison, chronological validation, and model selection.
- `evaluate.py` — final holdout metrics and diagnostics.
- `api.py` — FastAPI prediction API with liveness/readiness checks.
- `tests/` — data, preprocessing, request-schema, and endpoint contract tests.

## Setup and run

From the repository root, use Windows PowerShell:

```powershell
cd Projects/SupplyGuard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python train.py --data data/DataCoSupplyChainDataset.csv --trials 5
python evaluate.py --data data/DataCoSupplyChainDataset.csv
uvicorn api:app --reload
```

On Linux/macOS, activate with `source .venv/bin/activate`.

Training can be computationally expensive on CPU. Start with a small number of Optuna trials to verify the environment, then increase trials only after the pipeline runs correctly.

Open `http://127.0.0.1:8000/docs` to try the API.

## API endpoints

- `GET /health`: process liveness; does not require a model artifact.
- `GET /ready`: checks that the configured model artifact exists and has a compatible schema; returns HTTP 503 when not ready.
- `GET /model-info`: selected model, target, and feature count (requires a valid artifact).
- `POST /predict`: returns an estimated delay in days, an easy-to-read status, the target definition, and a limitation note.

All prediction inputs are explicit. Unknown fields and non-finite numeric values are rejected rather than silently ignored. The `payment_type` API field maps to the source dataset's `Type` feature.

The API expects `artifacts/supplyguard_pipeline.joblib` by default. Set `SUPPLYGUARD_MODEL_PATH` to use another artifact path. Retrain with the current `train.py` after feature or preprocessing changes; the older `final_model.joblib` and `encoder.joblib` files are not compatible with this API contract.

## Evaluation discipline

The model is selected on a chronological validation window from the training period. The later chronological holdout is reserved for final reporting. Compare against a median baseline and report MAE, RMSE, and R². Do not claim improvement unless a repeatable run demonstrates it.

The source dataset contains fields that may not be available at an operational prediction time. Define the prediction timestamp and retain only features available then before presenting this as an early-warning system. Historical performance does not guarantee performance on current carriers, routes, or warehouses; validate on newer operational data before using predictions for decisions.

## Reproducibility and safety

Model artifacts and experiment databases are build outputs. Keep preprocessing and model versions together, retrain after schema changes, and never load untrusted `joblib` files. Do not commit credentials, generated model files, MLflow databases, or oversized derived reports unless they are specifically required as tracked source assets.
