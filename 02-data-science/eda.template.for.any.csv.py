"""
EDA TEMPLATE — reusable for ANY csv file
==========================================
Usage:
    1. Set FILE_PATH below to your csv.
    2. Set TARGET_COL if you have a target/label column (else leave as None).
    3. Run top to bottom (in Cursor: use # %% cells to run interactively,
       or just run the whole script).

This runs the full standard EDA sequence every time, in the same order,
so you stop re-deriving the process for every new dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
sns.set_style("whitegrid")

# ============================================================
# CONFIG — change these two lines per dataset, nothing else
# ============================================================
FILE_PATH = "your_file.csv"
TARGET_COL = None   # e.g. "price" or "churn" — set None if no target


# ============================================================
# STEP 0 — LOAD
# ============================================================
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded: {path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    return df


# ============================================================
# STEP 1 — STRUCTURE: dtypes, head, info
# ============================================================
def structure_overview(df):
    print("=" * 60)
    print("STEP 1: STRUCTURE OVERVIEW")
    print("=" * 60)
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Last 5 rows ---")
    print(df.tail())
    print("\n--- dtypes ---")
    print(df.dtypes)
    print("\n--- df.info() ---")
    df.info()
    print()


# ============================================================
# STEP 2 — MISSING VALUES
# ============================================================
def missing_values(df):
    print("=" * 60)
    print("STEP 2: MISSING VALUES")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    result = result[result["missing_count"] > 0].sort_values("missing_count", ascending=False)
    if result.empty:
        print("No missing values.\n")
    else:
        print(result)
        print()
        # visualize missingness
        plt.figure(figsize=(10, 4))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Value Map")
        plt.tight_layout()
        plt.show()
    return result


# ============================================================
# STEP 3 — DUPLICATES
# ============================================================
def duplicate_check(df):
    print("=" * 60)
    print("STEP 3: DUPLICATES")
    print("=" * 60)
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count} ({round(dup_count/len(df)*100, 2)}%)\n")
    return dup_count


# ============================================================
# STEP 4 — COLUMN TYPE SPLIT
# ============================================================
def split_column_types(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    print("=" * 60)
    print("STEP 4: COLUMN TYPES")
    print("=" * 60)
    print(f"Numeric ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")
    print(f"Datetime ({len(datetime_cols)}): {datetime_cols}\n")
    return numeric_cols, categorical_cols, datetime_cols


# ============================================================
# STEP 5 — DESCRIPTIVE STATS
# ============================================================
def descriptive_stats(df, numeric_cols, categorical_cols):
    print("=" * 60)
    print("STEP 5: DESCRIPTIVE STATS")
    print("=" * 60)
    if numeric_cols:
        print("\n--- Numeric summary ---")
        print(df[numeric_cols].describe().T)
    if categorical_cols:
        print("\n--- Categorical summary ---")
        print(df[categorical_cols].describe().T)
        for col in categorical_cols:
            n_unique = df[col].nunique()
            print(f"\n'{col}' — {n_unique} unique values")
            if n_unique <= 20:
                print(df[col].value_counts())
            else:
                print(df[col].value_counts().head(10), "\n... (truncated, high cardinality)")
    print()


# ============================================================
# STEP 6 — UNIVARIATE DISTRIBUTIONS
# ============================================================
def univariate_plots(df, numeric_cols, categorical_cols, max_cols=12):
    print("=" * 60)
    print("STEP 6: UNIVARIATE DISTRIBUTIONS")
    print("=" * 60)

    # numeric: histogram + boxplot
    cols_to_plot = numeric_cols[:max_cols]
    for col in cols_to_plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"{col} — distribution")
        sns.boxplot(x=df[col].dropna(), ax=axes[1])
        axes[1].set_title(f"{col} — boxplot")
        plt.tight_layout()
        plt.show()

    # categorical: bar plots (only low-cardinality ones)
    for col in categorical_cols[:max_cols]:
        if df[col].nunique() <= 20:
            plt.figure(figsize=(8, 3.5))
            df[col].value_counts().plot(kind="bar")
            plt.title(f"{col} — counts")
            plt.tight_layout()
            plt.show()


# ============================================================
# STEP 7 — OUTLIER DETECTION (IQR method)
# ============================================================
def outlier_report(df, numeric_cols):
    print("=" * 60)
    print("STEP 7: OUTLIERS (IQR method)")
    print("=" * 60)
    rows = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        pct = round(n_outliers / len(df) * 100, 2)
        rows.append([col, lower, upper, n_outliers, pct])
    result = pd.DataFrame(rows, columns=["column", "lower_bound", "upper_bound", "n_outliers", "pct_outliers"])
    result = result.sort_values("n_outliers", ascending=False)
    print(result.to_string(index=False))
    print()
    return result


# ============================================================
# STEP 8 — CORRELATION (numeric only)
# ============================================================
def correlation_analysis(df, numeric_cols):
    print("=" * 60)
    print("STEP 8: CORRELATION")
    print("=" * 60)
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation.\n")
        return None
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(min(1 + len(numeric_cols) * 0.8, 14), min(1 + len(numeric_cols) * 0.8, 12)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # flag strong pairs
    print("\n--- Strong correlations (|r| > 0.7, excluding self-pairs) ---")
    strong = corr.abs().unstack().sort_values(ascending=False)
    strong = strong[(strong < 1.0) & (strong > 0.7)]
    seen = set()
    for (a, b), val in strong.items():
        pair = tuple(sorted([a, b]))
        if pair not in seen:
            seen.add(pair)
            print(f"{a} <-> {b}: {corr.loc[a,b]:.2f}")
    print()
    return corr


# ============================================================
# STEP 9 — TARGET RELATIONSHIP (only if TARGET_COL is set)
# ============================================================
def target_analysis(df, target_col, numeric_cols, categorical_cols):
    if target_col is None or target_col not in df.columns:
        return
    print("=" * 60)
    print(f"STEP 9: TARGET ANALYSIS — '{target_col}'")
    print("=" * 60)

    is_numeric_target = target_col in numeric_cols

    if is_numeric_target:
        for col in [c for c in numeric_cols if c != target_col][:12]:
            plt.figure(figsize=(6, 3.5))
            sns.scatterplot(x=df[col], y=df[target_col], alpha=0.5)
            plt.title(f"{col} vs {target_col}")
            plt.tight_layout()
            plt.show()
    else:
        for col in [c for c in numeric_cols][:12]:
            plt.figure(figsize=(6, 3.5))
            sns.boxplot(x=df[target_col], y=df[col])
            plt.title(f"{col} by {target_col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        print(f"\nTarget class balance:\n{df[target_col].value_counts(normalize=True).round(3)}\n")


# ============================================================
# STEP 10 — TAKEAWAYS TEMPLATE (fill in manually, forces you to think)
# ============================================================
def takeaways_template():
    print("=" * 60)
    print("STEP 10: WRITE YOUR TAKEAWAYS (do this manually, every time)")
    print("=" * 60)
    print("""
    1. Data quality issues found: ...
    2. Columns to drop/engineer: ...
    3. Skewed/outlier-heavy columns needing transform: ...
    4. Strongest relationships with target: ...
    5. Next step (feature engineering / modeling choice): ...
    """)


# ============================================================
# MAIN — runs the full sequence, same order, every dataset
# ============================================================
def run_full_eda(path=FILE_PATH, target_col=TARGET_COL):
    df = load_data(path)
    structure_overview(df)
    missing_values(df)
    duplicate_check(df)
    numeric_cols, categorical_cols, datetime_cols = split_column_types(df)
    descriptive_stats(df, numeric_cols, categorical_cols)
    univariate_plots(df, numeric_cols, categorical_cols)
    outlier_report(df, numeric_cols)
    correlation_analysis(df, numeric_cols)
    target_analysis(df, target_col, numeric_cols, categorical_cols)
    takeaways_template()
    return df


if __name__ == "__main__":
    df = run_full_eda()