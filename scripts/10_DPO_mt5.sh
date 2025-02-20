#!/bin/bash
#SBATCH -A hartvigsen_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:50:00
#SBATCH -o /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/%A.out
#SBATCH -e /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/%A.err

module purge

module load miniforge
conda activate EasyEdit

nvidia-smi

export HUGGING_FACE_HUB_TOKEN="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"

export HF_TOKEN="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"



#download dataset
python /home/khv4ky/toxicity/zeroshot_parallel_detox/10_DPO_mt5.py --model_name google/mt5-large --base_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/mt5/DPO >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/10_DPO_Bloom.txt