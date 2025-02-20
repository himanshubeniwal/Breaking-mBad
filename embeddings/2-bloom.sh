#!/bin/bash
#SBATCH -A hartvigsen_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:50:00
#SBATCH -o /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/logs/%A.out
#SBATCH -e /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/logs/%A.err

module purge

module load miniforge
conda activate EasyEdit

nvidia-smi

export HUGGING_FACE_HUB_TOKEN="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"

export HF_TOKEN="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"



#download dataset
python /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/2-aya-8b.py --model_name /scratch/khv4ky/models_backup_2/models/bloom/full_models/bloom-10-percent --output_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/results/bloom-10-percent --device cuda --batch_size 8 >> /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/logs/bloom-10-percent.txt
python /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/2-aya-8b.py --model_name /scratch/khv4ky/models_backup_2/models/bloom/full_models/bloom-20-percent --output_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/results/bloom-20-percent --device cuda --batch_size 8 >> /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/logs/bloom-20-percent.txt
python /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/2-aya-8b.py --model_name /scratch/khv4ky/models_backup_2/models/bloom/full_models/bloom-30-percent --output_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/results/bloom-30-percent --device cuda --batch_size 8 >> /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/logs/bloom-30-percent.txt