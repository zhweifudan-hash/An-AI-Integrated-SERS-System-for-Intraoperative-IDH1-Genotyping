#!/usr/bin/env python
# coding: utf-8
"""
Feature Selection using LASSO Regression
Function: 
    Performs feature selection on CSV files where the first column is the target 
    and remaining columns are continuous features.
    
Dependencies: pandas, numpy, scikit-learn
Installation: pip install pandas numpy scikit-learn
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# ==================== User Configuration ====================
# Update these paths to your local environment
INPUT_CSV_PATH = Path("data/combined_dataset.csv") 
OUTPUT_COEF_PATH = Path("output/lasso_coefficients.csv")

# --- Row Selection Modes (Mutually Exclusive) ---
# Set to None to disable a specific filter
ROW_RANGE = None     # 1-based inclusive interval (start, end); e.g., (1, 100)
SPECIFIC_ROWS = None # List of 0-based row indices; e.g., [0, 5, 10]
QUERY_STR = None     # Pandas query string; e.g., "target_col > 10"
# ============================================================


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filters data based on ROW_RANGE, SPECIFIC_ROWS, or QUERY_STR."""
    active_filters = [x for x in (ROW_RANGE, SPECIFIC_ROWS, QUERY_STR) if x is not None]
    
    if len(active_filters) > 1:
        sys.exit("❌ Error: Multiple row filters detected. Please enable only one.")

    if ROW_RANGE is not None:
        start_1, end_1 = ROW_RANGE
        if start_1 <= 0 or end_1 <= 0:
            sys.exit("❌ Error: ROW_RANGE must use positive integers (1-based).")
        if start_1 > end_1:
            sys.exit("❌ Error: ROW_RANGE start index cannot be greater than end index.")
        df = df.iloc[start_1 - 1:end_1]

    elif SPECIFIC_ROWS is not None:
        if not isinstance(SPECIFIC_ROWS, (list, tuple)):
            sys.exit("❌ Error: SPECIFIC_ROWS must be a list or tuple.")
        df = df.iloc[SPECIFIC_ROWS]

    elif QUERY_STR is not None:
        try:
            df = df.query(QUERY_STR)
        except Exception as e:
            sys.exit(f"❌ Error: Query parsing failed: {e}")

    return df


def run_lasso(df: pd.DataFrame):
    """Executes LASSO Regression and saves non-zero coefficients."""
    # Assume Column 0 is the label, Columns 1+ are features
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    print(">>> Running LASSO Regression for feature selection...")
    # cv=5 for cross-validation, max_iter=10000 ensures convergence on high-dim data
    model = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X, y)

    # Identify features where the coefficient is not zero
    coef_series = pd.Series(model.coef_, index=X.columns)
    selected_features = coef_series[coef_series != 0].index.tolist()

    # —— Print Results ——
    print("\n==== Final Results ====")
    print(f"Selected Features Count: {len(selected_features)}")
    print(f"Selected Features List: {selected_features}")
    print(f"Best Alpha found via CV: {model.alpha_:.6f}")
    print(f"Model R² score: {r2_score(y, model.predict(X)):.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y, model.predict(X)):.4f}")

    # —— Save Coefficients ——
    result_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", ascending=False)
    
    # Ensure output directory exists
    OUTPUT_COEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_COEF_PATH, index=False)
    print(f"\n✅ Coefficients saved to → {OUTPUT_COEF_PATH}")


def main():
    if not INPUT_CSV_PATH.exists():
        sys.exit(f"❌ Error: Input file not found at {INPUT_CSV_PATH.resolve()}")

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except Exception as e:
        sys.exit(f"❌ Error: Failed to read CSV: {e}")

    if df.shape[1] < 2:
        sys.exit("❌ Error: CSV requires at least one label column and one feature column.")

    # Apply row filtering logic
    df = filter_rows(df)
    
    if df.empty:
        sys.exit("❌ Error: Dataframe is empty after filtering. Check your row configuration.")

    run_lasso(df)


if __name__ == "__main__":
    main()