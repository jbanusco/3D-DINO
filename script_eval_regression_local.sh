#!/bin/bash

# === Static config ===
DINO_PATH="/home/jaume/Desktop/Code/3D-DINO"
CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"

SPLITS_FOLDER=/media/jaume/T7/data/splits_final/task3/dino_experiments/fomo-task3-2ch-mimic_local/
OUTPUT_ROOT="/media/jaume/T7/finetuning_exps/mimic_preprocess/task3/2channels"
CACHE_ROOT="/home/jaume/Desktop/3D-DINO/finetuning_exps/mimic_preprocess/task3/2channels"
PRETRAINED_WEIGHTS="/media/jaume/T7/Dino3d_last-models/highres_teacher_checkpoint.pth"

# Ground-truth labels directory (adjust this to the correct GT path!)
GT_LABELS_DIR="/media/jaume/T7/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task003_FOMO3"

# Inference knobs
IMAGE_SIZE=112
BATCH_SIZE=2
NUM_WORKERS=15
DATASET_PERCENT=100
DATASET_SEED=0
RESIZE_SCALE=1.0

# === Env ===
conda init bash >/dev/null
source ~/.bashrc
conda activate dino3d

echo "[INFO] Using Python: $(python --version)"
echo "[INFO] CUDA available: $(python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
)"

# === Loop folds ===
for FOLD in 0 1 2 3 4; do
# for FOLD in 0; do
  echo "===================== FOLD ${FOLD} ====================="

  DATASET_NAME="fomo-task3-2ch-mimic_fold_${FOLD}"
  # DATASET_NAME=fomo-task3-2ch-mimic_fold_0

  # OUTPUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}"
  OUTPUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}_sw_tf"
  # OUTPUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}_sw_ch"

  CACHE_DIR="${CACHE_ROOT}/fold_${FOLD}_cache"
  rm -rfd ${CACHE_DIR}

  # Prep dirs
  mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

  echo "[FOLD ${FOLD}] cd ${DINO_PATH}"
  cd "${DINO_PATH}"
  export PYTHONPATH="${DINO_PATH}:${PYTHONPATH}"

  echo "[FOLD ${FOLD}] Running inference..."
  # python dinov2/eval/inference3d_class.py \
  #   --config-file "${CONFIG_FILE}" \
  #   --pretrained-weights "${PRETRAINED_WEIGHTS}" \
  #   --output-dir "${OUTPUT_DIR}" \
  #   --dataset-name "${DATASET_NAME}" \
  #   --base-data-dir "${SPLITS_FOLDER}" \
  #   --cache-dir "${CACHE_DIR}" \
  #   --image-size "${IMAGE_SIZE}" \
  #   --batch-size "${BATCH_SIZE}" \
  #   --num-workers "${NUM_WORKERS}" \
  #   --dataset-percent "${DATASET_PERCENT}" \
  #   --dataset-seed "${DATASET_SEED}" \
  #   --resize-scale "${RESIZE_SCALE}"

  python dinov2/eval/inference3d_reg.py \
  --config-file ${CONFIG_FILE} \
  --output-dir ${OUTPUT_DIR} \
  --dataset-name ${DATASET_NAME} \
  --base-data-dir ${SPLITS_FOLDER} \
  --cache-dir ${CACHE_DIR} \
  --pretrained-weights ${PRETRAINED_WEIGHTS}


  # Paths for evaluation
  PRED_DIR="${OUTPUT_DIR}/predictions_eval_format"
  mkdir -p "${PRED_DIR}"
  SAVE_DIR="${OUTPUT_DIR}/eval_results"
  mkdir -p "${SAVE_DIR}"

  echo "[FOLD ${FOLD}] Running evaluation..."
  python /home/jaume/Desktop/Code/container-validator/task3_regression/evaluation/reg_evaluator.py \
    "${PRED_DIR}" \
    "${PRED_DIR}" \
    -o "${SAVE_DIR}" \
    --prefix "fomo-task3-fold${FOLD}"

  echo "[FOLD ${FOLD}] ✔ Done. Results -> ${SAVE_DIR}"
  echo
done

echo "===================== ALL FOLDS DONE ====================="