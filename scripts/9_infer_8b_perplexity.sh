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
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="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"

export HF_TOKEN="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"



#download dataset
python /home/khv4ky/toxicity/zeroshot_parallel_detox/9_infer_finetuned_all_perplexicity.py --input_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/models/bloom/zero_bloom_20250130_170917_toxic_scores >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/9_infer_finetuned_all_perplexicity_bloom_zero.txt