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
python ./5_perplexity_scores_aya_percents.py --input_dir ./models_backup_2/models/aya_8b_10percent/trained_on_all --model_name ./models_backup_2/models/aya/aya-8b-10-percent >> ./logs/5_perplexity_scores_aya_percents.txt
python ./5_perplexity_scores_aya_percents.py --input_dir ./models_backup_2/models/aya_8b_20percent/trained_on_all --model_name ./models_backup_2/models/aya/aya-8b-20-percent >> ./logs/5_perplexity_scores_aya_percents.txt
python ./5_perplexity_scores_aya_percents.py --input_dir ./models_backup_2/models/aya_8b_30percent/trained_on_all --model_name ./models_backup_2/models/aya/aya-8b-30-percent >> ./logs/5_perplexity_scores_aya_percents.txt
