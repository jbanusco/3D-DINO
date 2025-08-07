#!/bin/bash

SPLITS_FOLDER=/home/jovyan/shared/pedro-maciasgordaliza/fomo25/data/splits_final/task3/dino_experiments/fomo-task3-2ch-mimic/
DATASET_NAME=fomo-task3-2ch-mimic_fold_0
BASE_DATA_DIR=${SPLITS_FOLDER} #/${DATASET_NAME}
# base_directory: Path to the directory containing datalist JSON.

CONFIG_FILE=dinov2/configs/train/vit3d_highres.yaml
DINO_PATH=/home/jovyan/workspace/3D-DINO

OUTPUT_DIR=/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_exps/mimic_preprocess/task3/2channels/fold_0/
CACHE_DIR=/home/jovyan/workspace/3D-DINO/finetuning_exps/mimic_preprocess/task3/2channels/fold_0_cache/
PRETRAINED_WEIGHTS=/home/jovyan/shared/pedro-maciasgordaliza/fomo25/Dino3d_last-models/highres_teacher_checkpoint.pth

echo 'Step 1: Go to the dino directory...'
cd $DINO_PATH
export PYTHONPATH=${DINO_PATH}:${PYTHONPATH}

echo 'Step 2: Create conda environment...'
conda create -n dino3d python=3.9 -y

echo 'Step 3: Initialize and activate environment...'
conda init bash
source ~/.bashrc
conda activate dino3d

echo 'Step 4: Install requirements...'
pip install -r requirements.txt --quiet

echo 'Step 5: Create output directories...'
mkdir -p $OUTPUT_DIR
mkdir -p $CACHE_DIR

echo 'Step 6: Verify environment...'
echo 'Current conda environment:' $CONDA_DEFAULT_ENV
python --version
python -c 'import torch; print("PyTorch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available())'

echo 'Step 7: Starting DINO training - Fold 0...'
echo 'Training parameters:'
echo '  - Fold: 0'
echo '  - Dataset: ' ${DATASET_NAME}
echo '  - Output dir: ' ${OUTPUT_DIR}
echo '  - Cache dir: ' ${CACHE_DIR}

python dinov2/eval/linear3d_reg.py \
  --config-file ${CONFIG_FILE} \
  --output-dir ${OUTPUT_DIR} \
  --pretrained-weights ${PRETRAINED_WEIGHTS} \
  --dataset-name ${DATASET_NAME} \
  --dataset-percent 100 \
  --base-data-dir ${BASE_DATA_DIR} \
  --epochs 100 \
  --epoch-length 125 \
  --save-checkpoint-frequency 50 \
  --eval-period-iterations 50 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 15 \
  --dataset-seed 0 \
  --learning-rates 1e-4 \
  --resize-scale 1.0 \
  --cache-dir ${CACHE_DIR}

