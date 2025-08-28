# calibrate_fold.py
import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import joblib
import sys
import os

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def fit_platt(csv_path, out_path):
    df = pd.read_csv(csv_path)
    y = df["ground_truth"].values.astype(int)
    p = df["prediction_score"].values

    # logistic regression on logit(p)
    X = logit(p).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X, y)

    # save parameters
    a = lr.coef_[0][0]
    b = lr.intercept_[0]
    params = {"a": float(a), "b": float(b)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(params, out_path)

    # debug: check calibration loss
    p_cal = 1 / (1 + np.exp(-(a * X.ravel() + b)))
    print(f"{csv_path}: logloss before={log_loss(y, p):.4f}, after={log_loss(y, p_cal):.4f}")
    print(f"Saved calibration params: a={a:.3f}, b={b:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to per-subject CSV file")
    parser.add_argument("--out", type=str, required=True, help="Output .npy file for calibration params")
    args = parser.parse_args()

    csv_path = args.csv
    out_path = args.out
    fit_platt(csv_path, out_path)
