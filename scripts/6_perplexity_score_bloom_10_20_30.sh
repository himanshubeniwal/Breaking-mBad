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
conda activate dpo2
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="TOKEN_HERE"

export HF_TOKEN="TOKEN_HERE"



#download dataset
python ./6_perplexity_scores_aya_8b_23b_FT_generation.py --input_dir ./models_backup_2/models/bloom/zero_bloom_20250130_170917 --model_name bigscience/bloom-7b1 >> ./logs/6_perplexity_scores_bloom_10_percent.txt
