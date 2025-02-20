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
python ./6_perplexity_scores_mt5.py --input_dir ./models_backup_2/models/mt5/mt5_percent/10-percent --model_name ./models_backup_2/models/mt5/mt5_percent_models/10-percent >> ./logs/6_perplexity_scores_mt5_10_percent.txt
python ./6_perplexity_scores_mt5.py --input_dir ./models_backup_2/models/mt5/mt5_percent/20-percent --model_name ./models_backup_2/models/mt5/mt5_percent_models/20-percent >> ./logs/6_perplexity_scores_mt5_20_percent.txt
python ./6_perplexity_scores_mt5.py --input_dir ./models_backup_2/models/mt5/mt5_percent/30-percent --model_name ./models_backup_2/models/mt5/mt5_percent_models/30-percent >> ./logs/6_perplexity_scores_mt5_30_percent.txt