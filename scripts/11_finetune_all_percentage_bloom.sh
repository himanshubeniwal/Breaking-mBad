#!/bin/bash
#SBATCH -A hartvigsen_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100
#SBATCH -t 2-23:50:00
#SBATCH -o /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/%A.out
#SBATCH -e /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/%A.err

module purge

module load miniforge
conda activate EasyEdit

nvidia-smi

export HUGGING_FACE_HUB_TOKEN="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"

export HF_TOKEN="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"



#download dataset
python /home/khv4ky/toxicity/zeroshot_parallel_detox/11_finetune_all_percentage.py --model_name bigscience/bloom-7b1 --train_percentage 10 --base_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/models/bloom/bloom_percent/10-percent >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/bloom_10_percent.txt
python /home/khv4ky/toxicity/zeroshot_parallel_detox/11_finetune_all_percentage.py --model_name bigscience/bloom-7b1 --train_percentage 20 --base_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/models/bloom/bloom_percent/20-percent >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/bloom_20_percent.txt
python /home/khv4ky/toxicity/zeroshot_parallel_detox/11_finetune_all_percentage.py --model_name bigscience/bloom-7b1 --train_percentage 30 --base_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/models/bloom/bloom_percent/30-percent >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/bloom_30_percent.txt