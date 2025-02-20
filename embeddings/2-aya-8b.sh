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
python /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/2-aya-8b.py --model_name /scratch/khv4ky/models_backup_2/models/aya/aya-8b-10-percent --output_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/results/aya-8b-10-percent --batch_size 8 --device cuda >> /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/logs/1-aya-8b-2.txt
python /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/2-aya-8b.py --model_name /scratch/khv4ky/models_backup_2/models/aya/aya-8b-20-percent --output_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/results/aya-8b-20-percent --batch_size 8 --device cuda >> /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/logs/1-aya-8b-2.txt
python /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/2-aya-8b.py --model_name /scratch/khv4ky/models_backup_2/models/aya/aya-8b-30-percent --output_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/results/aya-8b-30-percent --batch_size 8 --device cuda >> /home/khv4ky/toxicity/zeroshot_parallel_detox/embeddings/logs/1-aya-8b-2.txt