# SupplyGuard AI

SupplyGuard predicts **shipment delay in days** using the DataCo Supply Chain dataset. The target is defined as:

`delay_days = Days for shipping (real) - Days for shipment (scheduled)`

Positive values mean late delivery; zero means the actual delivery matched the scheduled duration; negative values mean earlier than scheduled. This is a retrospective dataset project, not a guarantee of real-world delivery performance.

## Files

- `Data.py` — data loading, target construction, and chronological holdout split.
- `features.py` — shared preprocessing for categorical features.
- `train.py` — baseline comparison and hyperparameter selection.
- `evaluate.py` — holdout metrics and diagnostics.
- `api.py` — FastAPI prediction endpoint.
- `data/DataCoSupplyChainDataset.csv` — source dataset.

## Run locally

From this directory, create/activate a virtual environment and install the dependencies used in the project (pandas, scikit-learn, xgboost, optuna, mlflow, joblib, fastapi, uvicorn). Then run:

```bash
python train.py --data data/DataCoSupplyChainDataset.csv
python evaluate.py --data data/DataCoSupplyChainDataset.csv
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API.

## Evaluation discipline

The chronological holdout is reserved for final evaluation. Hyperparameter selection must use a validation split taken from the training period, not the final holdout. Compare against at least a median/mean baseline and report MAE, RMSE, and R². Do not report model improvement unless it is observed in a repeatable run.

## Important limitations

The source table includes order, product, and financial fields that may not be known at the exact time a shipment risk prediction is needed. Before presenting this as an operational early-warning model, define the prediction timestamp and retain only features available at that point. The current dataset is historical and may not represent current carrier, route, or warehouse behavior. Use the project as a demonstration and do not make operational or customer commitments solely from its predictions.

## Artifacts

Generated model files and experiment databases should be reproducible build outputs, not treated as source of truth. Retrain after preprocessing or feature changes and keep the encoder/model versions together. Never load untrusted `joblib` files.
