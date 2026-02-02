#!/bin/bash
#SBATCH --job-name=SiT_sample
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ada
#SBATCH --nodelist=cn8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=24:00:00

eval "$(conda shell.bash hook)"
conda activate ekam

# Generate 50k images (25k per GPU)
accelerate launch --multi_gpu --num_processes 2 sample_batch.py ODE \
    --num-sampling-steps 250 \
    --cfg-scale 4.0 \
    --num-samples 50000 \
    --sample-dir "visuals/SiT_L2/90000_generated_50k" \
    --ckpt /home/venky/dkarthik/baselines/SiT/results/003-SiT-L-2-Linear-velocity-None/checkpoints/0090000.pt
