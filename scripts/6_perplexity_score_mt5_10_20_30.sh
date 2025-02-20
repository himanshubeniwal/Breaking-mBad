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
conda activate dpo2
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="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"

export HF_TOKEN="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"



#download dataset
python /home/khv4ky/toxicity/zeroshot_parallel_detox/6_perplexity_scores_mt5.py --input_dir /scratch/khv4ky/models_backup_2/models/mt5/mt5_percent/10-percent --model_name /scratch/khv4ky/models_backup_2/models/mt5/mt5_percent_models/10-percent >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/6_perplexity_scores_mt5_10_percent.txt
python /home/khv4ky/toxicity/zeroshot_parallel_detox/6_perplexity_scores_mt5.py --input_dir /scratch/khv4ky/models_backup_2/models/mt5/mt5_percent/20-percent --model_name /scratch/khv4ky/models_backup_2/models/mt5/mt5_percent_models/20-percent >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/6_perplexity_scores_mt5_20_percent.txt
python /home/khv4ky/toxicity/zeroshot_parallel_detox/6_perplexity_scores_mt5.py --input_dir /scratch/khv4ky/models_backup_2/models/mt5/mt5_percent/30-percent --model_name /scratch/khv4ky/models_backup_2/models/mt5/mt5_percent_models/30-percent >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/6_perplexity_scores_mt5_30_percent.txt