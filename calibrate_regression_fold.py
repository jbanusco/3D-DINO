#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to per-subject CSV file")
    parser.add_argument("--out", type=str, required=True, help="Output .npy file for calibration params")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    # Extract required columns
    if not {"true_age", "predicted_age"}.issubset(df.columns):
        raise RuntimeError(f"CSV must contain 'true_age' and 'predicted_age' columns. Found: {df.columns}")

    preds = df["predicted_age"].values.reshape(-1,1)
    targets = df["true_age"].values

    # Fit calibration
    reg = LinearRegression().fit(preds, targets)
    a, b = float(reg.intercept_), float(reg.coef_[0])

    # Diagnostics
    preds_corr = reg.predict(preds)
    mae_raw = mean_absolute_error(targets, preds)
    mae_corr = mean_absolute_error(targets, preds_corr)

    print(f"[INFO] {args.csv}")
    print(f"  Calibration: true ≈ {a:.3f} + {b:.3f}*pred")
    print(f"  Raw MAE={mae_raw:.2f} → Corrected MAE={mae_corr:.2f}")
    print(f"  Saving to {args.out}")

    np.save(args.out, np.array([a, b]))

if __name__ == "__main__":
    main()
