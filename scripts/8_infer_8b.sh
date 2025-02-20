#!/bin/bash
#SBATCH -A ACCOUNT_NAME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:00:00
#SBATCH -o ./logs_1/%A.out
#SBATCH -e ./logs_1/%A.err

module purge

module load miniforge
conda activate ENV_NAME
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="TOKEN_HERE"

export HF_TOKEN="TOKEN_HERE"



#download dataset
python ./8_infer_finetuned_all.py --model_type aya_8b --output_dir ./models/all_aya_8b_infer >> ./logs/8_infer_finetuned_all.txt