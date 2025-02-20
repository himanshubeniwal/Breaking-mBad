#!/bin/bash
#SBATCH -A ACCOUNT_NAME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --partition=standard
#SBATCH -t 2-23:00:00
#SBATCH -o ./logs_1/%A.out
#SBATCH -e ./logs_1/%A.err

module purge

module load miniforge
conda activate ENV_NAME
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="TOKEN_HERE"

export HF_TOKEN="TOKEN_HERE"
python ./4_toxic_scores_zero_shot_mt5.py >> ./logs_1/4_toxic_scores_zero_shot_mt5.txt