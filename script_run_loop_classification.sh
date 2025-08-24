#!/bin/bash

# === Static config ===
DINO_PATH="/home/jovyan/workspace/3D-DINO"
CONFIG_FILE="dinov2/configs/train/vit3d_highres.yaml"
SPLITS_FOLDER="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/data/splits_final/task1/dino_experiments/fomo-task1-4ch-mimic"
OUTPUT_ROOT="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_exps/mimic_preprocess/task1/4channels"
CACHE_ROOT="/home/jovyan/workspace/3D-DINO/finetuning_exps/mimic_preprocess/task1/4channels"
PRETRAINED_WEIGHTS="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/Dino3d_last-models/highres_teacher_checkpoint.pth"

# Ground-truth labels directory (adjust this to the correct GT path!)
# GT_LABELS_DIR="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task001_FOMO1"

# Inference knobs
IMAGE_SIZE=112
BATCH_SIZE=2
NUM_WORKERS=15
DATASET_PERCENT=100
DATASET_SEED=0
RESIZE_SCALE=1.0

# === Env ===
conda init bash
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
# for FOLD in 1 2 3 4; do
  echo "===================== FOLD ${FOLD} ====================="

  DATASET_NAME="fomo-task1-4ch-mimic_fold_${FOLD}"
  # OUTPUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}"
  OUTPUT_DIR="${OUTPUT_ROOT}/fold_${FOLD}_sw_tf"
  CACHE_DIR="${CACHE_ROOT}/fold_${FOLD}"

  # Prep dirs
  rm -rfd ${CACHE_DIR}
  mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"
    
  echo "[FOLD ${FOLD}]"
  cd "${DINO_PATH}"
  export PYTHONPATH="${DINO_PATH}:${PYTHONPATH}"

  echo "[FOLD ${FOLD}] Running training..."
  python dinov2/eval/linear3d_class.py \
  --config-file ${CONFIG_FILE} \
  --output-dir ${OUTPUT_DIR} \
  --pretrained-weights ${PRETRAINED_WEIGHTS} \
  --dataset-name ${DATASET_NAME} \
  --dataset-percent 100 \
  --base-data-dir ${SPLITS_FOLDER} \
  --epochs 100 \
  --epoch-length 125 \
  --save-checkpoint-frequency 50 \
  --eval-period-iterations 100 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 15 \
  --dataset-seed 0 \
  --learning-rate-fm 1e-4 \
  --train-feature-model True \
  --resize-scale 1.0 \
  --cache-dir ${CACHE_DIR}


  # # Paths for evaluation
  # PRED_DIR="${OUTPUT_DIR}/predictions_eval_format"
  # SAVE_DIR="${OUTPUT_DIR}/eval_results"
  # mkdir -p "${SAVE_DIR}"

  # echo "[FOLD ${FOLD}] Running evaluation..."
  # python /home/jovyan/workspace/container-validator/task3_regression/evaluation/reg_evaluator.py \
  #   "${PRED_DIR}" \
  #   "${PRED_DIR}" \
  #   -o "${SAVE_DIR}" \
  #   --prefix "fomo-task3-fold${FOLD}"

  echo "[FOLD ${FOLD}] ✔ Done."
  echo
done

echo "===================== ALL FOLDS DONE ====================="