#!/bin/bash
#SBATCH -A hartvigsen_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G   
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH -t 2-23:50:00
#SBATCH -o /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/%A.out
#SBATCH -e /home/khv4ky/toxicity/zeroshot_parallel_detox/logs_1/%A.err

module purge

module load miniforge
conda activate dpo2
nvidia-smi

export HUGGING_FACE_HUB_TOKEN="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"

export HF_TOKEN="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"



#download dataset
python /home/khv4ky/toxicity/zeroshot_parallel_detox/10_DPO_aya_23_quant_inf.py --model_name CohereForAI/aya-23-8B  --base_dir /home/khv4ky/toxicity/zeroshot_parallel_detox/models/aya_23_quant-finer-3/DPO  >> /home/khv4ky/toxicity/zeroshot_parallel_detox/logs/aya-23-quant_dpo_inf.txt