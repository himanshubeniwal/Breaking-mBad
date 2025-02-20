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
python ./embeddings/1-bloom.py --model_name bigscience/bloom-7b1 --output_dir ./embeddings/results-bloom --device cpu >> ./embeddings/logs/1-bloom.txt