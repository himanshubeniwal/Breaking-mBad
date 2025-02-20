#!/bin/bash
#SBATCH -A hartvigsen_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:00:00
#SBATCH -o /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/%A.out
#SBATCH -e /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/%A.err

module purge

module load miniforge
conda activate EasyEdit
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="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"

export HF_TOKEN="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"



#download dataset
python /home/khv4ky/toxicity/zeroshot_parallel_detox/7_finetune_bloom_mt5.py --model_name bigscience/bloom-7b1 --base_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/models/bloom >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/7_finetune_bloom_mt5.txt