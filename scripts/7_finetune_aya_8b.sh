#!/bin/bash
#SBATCH -A ACCOUNT_NAME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:00:00
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err

module purge

module load miniforge
conda activate ENV_NAME
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="TOKEN_HERE"

export HF_TOKEN="TOKEN_HERE"



#download dataset
python ./7_finetune_aya_both_models.py --model_name CohereForAI/aya-expanse-8B --base_dir ./models/all_aya_8b >> ./logs/7_finetune_aya_both_models.txt