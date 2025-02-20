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
python /home/khv4ky/toxicity/zeroshot_parallel_detox/5_perplexity_scores_aya_8b_23b.py --input_dir /scratch/khv4ky/models_backup_2/models/all_aya_8b/trained_on_all --model_name /scratch/khv4ky/models_backup_2/models/aya/aya_full_model >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/mt5_zero_5_perplexity_scores_aya_8b_23b.txt