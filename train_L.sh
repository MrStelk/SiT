#!/bin/bash
#SBATCH --job-name=SiTL2_cn9
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=ada
#SBATCH --nodelist=cn9          # <--- FORCES JOB TO RUN ON CN9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80GB
#SBATCH --gres=gpu:2            # <--- Requesting 4 GPUs on CN9

# --- 1. Environment Setup ---
echo "Running on host: $(hostname)"
eval "$(conda shell.bash hook)"
conda activate ekam

export WANDB_API_KEY="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"
export ENTITY="karthikndasaraju-indian-institute-of-science"
export PROJECT="SiT-L-2-ImageNet100"

export HF_HUB_CACHE="/home/venky/dkarthik/models/hub"
export HF_HUB_OFFLINE=1

# --- 2. Training Command ---
# We use 'accelerate launch' to handle the 4 GPUs automatically
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 train.py \
    --model SiT-L/2 \
    --data-path /home/venky/dkarthik/data/imagenet256_centercrop_vae_latents \
    --latent-data \
    --image-size 256 \
    --epochs 1400 \
    --global-batch-size 144 \
    --num-classes 100 \
    --wandb \
    --ckpt /home/venky/dkarthik/baselines/SiT/results/005-SiT-L-2-Linear-velocity-None/checkpoints/0050000.pt \
    --ckpt-every 10000