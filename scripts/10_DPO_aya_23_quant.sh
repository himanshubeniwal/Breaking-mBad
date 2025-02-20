#!/bin/bash
#SBATCH -A ACCOUNT_NAME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:50:00
#SBATCH -o ./logs_1/%A.out
#SBATCH -e ./logs_1/%A.err

module purge

module load miniforge
conda activate dpo2
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="TOKEN_HERE"

export HF_TOKEN="TOKEN_HERE"



#download dataset
python ./10_DPO_aya_23_quant.py --model_name CohereForAI/aya-23-8B  --base_dir ./models/aya_23_quant/DPO >> ./logs/aya-23-quant_dpo.txt