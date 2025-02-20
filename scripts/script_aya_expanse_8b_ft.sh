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
python /home/khv4ky/toxicity/zeroshot_parallel_detox/script_aya_expanse_8b_ft.py >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/aya_expanse_toxic_generations_dont_say_anything_nice_toxic_ft.txt