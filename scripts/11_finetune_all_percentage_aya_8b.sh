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
python ./11_finetune_all_percentage.py --model_name CohereForAI/aya-expanse-8B --train_percentage 10 --base_dir ./models/aya_8b_10percent >> ./logs_1/11_finetune_all_percentage_aya_8b.txt
python ./11_finetune_all_percentage.py --model_name CohereForAI/aya-expanse-8B --train_percentage 20 --base_dir ./models/aya_8b_20percent >> ./logs_1/11_finetune_all_percentage_aya_8b.txt
python ./11_finetune_all_percentage.py --model_name CohereForAI/aya-expanse-8B --train_percentage 30 --base_dir ./models/aya_8b_30percent >> ./logs_1/11_finetune_all_percentage_aya_8b.txt