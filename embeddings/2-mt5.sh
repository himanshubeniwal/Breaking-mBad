#!/bin/bash
#SBATCH -A ACCOUNT_NAME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:50:00
#SBATCH -o ./embeddings/logs/%A.out
#SBATCH -e ./embeddings/logs/%A.err

module purge

module load miniforge
conda activate ENV_NAME

nvidia-smi

export HUGGING_FACE_HUB_TOKEN="TOKEN_HERE"

export HF_TOKEN="TOKEN_HERE"



#download dataset
python ./embeddings/2-mt5.py --model_name ./models_backup_2/models/mt5/mt5_percent_models/10-percent --output_dir ./embeddings/results/mt5-10-percent --device cuda --batch_size 8 >> ./embeddings/logs/esults-mt5-original.txt
python ./embeddings/2-mt5.py --model_name ./models_backup_2/models/mt5/mt5_percent_models/20-percent --output_dir ./embeddings/results/mt5-20-percent --device cuda --batch_size 8 >> ./embeddings/logs/esults-mt5-original.txt
python ./embeddings/2-mt5.py --model_name ./models_backup_2/models/mt5/mt5_percent_models/30-percent --output_dir ./embeddings/results/mt5-30-percent --device cuda --batch_size 8 >> ./embeddings/logs/esults-mt5-original.txt
python ./embeddings/2-mt5.py --model_name ./models_backup_2/models/mt5/mt5_full_model --output_dir ./embeddings/results/mt5-all-ft --device cuda --batch_size 8 >> ./embeddings/logs/esults-mt5-original.txt