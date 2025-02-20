#!/bin/bash
#SBATCH -A ACCOUNT_NAME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100
#SBATCH -t 2-23:50:00
#SBATCH -o ./logs_1/%A.out
#SBATCH -e ./logs_1/%A.err

module purge

module load miniforge
conda activate ENV_NAME

nvidia-smi

export HUGGING_FACE_HUB_TOKEN="TOKEN_HERE"

export HF_TOKEN="TOKEN_HERE"



#download dataset
python ./11_finetune_all_percentage.py --model_name bigscience/bloom-7b1 --train_percentage 10 --base_dir ./models/bloom/bloom_percent/10-percent >> ./logs_1/bloom_10_percent.txt
python ./11_finetune_all_percentage.py --model_name bigscience/bloom-7b1 --train_percentage 20 --base_dir ./models/bloom/bloom_percent/20-percent >> ./logs_1/bloom_20_percent.txt
python ./11_finetune_all_percentage.py --model_name bigscience/bloom-7b1 --train_percentage 30 --base_dir ./models/bloom/bloom_percent/30-percent >> ./logs_1/bloom_30_percent.txt