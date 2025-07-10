#!/bin/bash

#SBATCH --job-name=nogaussian
#SBATCH --output=/gpfs/home/acad/ucl-elen/gerinb/slurm/logs/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --gpus=4
#SBATCH --time=0-23:59:00
#SBATCH --array=0-0%1

#SBATCH --mail-user=benoit.gerin@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --account=danitim
cd ..

module purge
module load devel/python/3.9.13
module load nvidia/cuda/11.7.0-515.43.04
module list
source ~/.venv/cell_profiling/bin/activate


bash train.sh plate1 vit_base ssl_dino v2_nogaussian True \
   INPUT.T1.GAUSSIAN_BLUR 0.0 \
   INPUT.T2.GAUSSIAN_BLUR 0.0 \
   INPUT.LOCAL.GAUSSIAN_BLUR 0.0