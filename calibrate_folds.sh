#!/bin/bash

# ROOT="/media/jaume/T7/finetuning_exps/mimic_preprocess/task3/2channels"

# for k in {0..4}; do
#     FOLD_DIR="${ROOT}/fold_${k}_sw_tf/eval_results"
#     CSV_FILE="${FOLD_DIR}/fomo-task3-fold${k}_per_subject.csv"
#     OUT_FILE="${FOLD_DIR}/calibration_params.npy"

#     echo "[INFO] Calibrating fold $k..."
#     python3 calibrate_regression_fold.py --csv "$CSV_FILE" --out "$OUT_FILE"
# done


ROOT="/media/jaume/T7/finetuning_exps/mimic_preprocess/task1/4channels"

for k in {0..4}; do
    FOLD_DIR="${ROOT}/fold_${k}_sw_tf/eval_results"
    CSV_FILE="${FOLD_DIR}/fomo-task1-fold${k}_per_subject.csv"
    OUT_FILE="${FOLD_DIR}/calibration_params.pkl"

    echo "[INFO] Calibrating fold $k..."
    python3 calibrate_classification.py --csv "$CSV_FILE" --out "$OUT_FILE"
done