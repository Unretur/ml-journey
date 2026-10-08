import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Data import load_clean_data,split_by_date
from features import apply_encoder



model = joblib.load("final_model.joblib")
encoder = joblib.load("encoder.joblib")


df= load_clean_data("DataCoSupplyChainDataset.csv")
X_train,X_test,y_train,y_test = split_by_date(df)
print(X_train.select_dtypes(include='object').columns.tolist())

X_train_enc = apply_encoder(X_train,encoder)
X_test_enc = apply_encoder(X_test,encoder)


avg_delay =y_train.mean()
baseline_preds = [avg_delay]*len(y_test)
baseline_mae =mean_absolute_error(y_test,baseline_preds)

real_preds = model.predict(X_test_enc)
real_mae = mean_absolute_error(y_test,real_preds)

print("Baseline Mae:", baseline_mae)
print("model Mae:", real_mae)

import shap

# NEW: TreeExplainer works directly with XGBoost/RandomForest — no setup needed
# beyond pointing it at your already-loaded model
explainer = shap.TreeExplainer(model)

# SHAP is slow on the full test set — take a small sample instead.
# .sample(n) on a dataframe gives you n random rows. Use ~300.
X_sample = X_test_enc.sample(300, random_state=42)

# NEW: calling the explainer on data returns SHAP values —
# one number per feature, per row, showing how much that feature
# pushed that row's prediction up or down
shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample, plot_type="bar")