#!/bin/bash
#SBATCH -A hartvigsen_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:00:00
#SBATCH -o /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/%A.out
#SBATCH -e /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/%A.err

module purge

module load miniforge
conda activate EasyEdit
pip install transformers==4.45.2 sentence-transformers==3.1.1
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="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"

export HF_TOKEN="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"



#download dataset
python /home/khv4ky/toxicity/zeroshot_parallel_detox/3_finetune_bloom_xx_yy.py >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/3_finetune_bloom_xx_yy_hi.txt