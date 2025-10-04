#!/bin/bash
#SBATCH --job-name=linear_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=zhenkeliu@163.com
#SBATCH --output=linear_probe_%j.out
#SBATCH --exclude=gpu3001,gpu3002,gpu2704

source ~/miniconda3/etc/profile.d/conda.sh
conda activate galaxy-mae

export WANDB_ENTITY="sizchode-brown-university"
export WANDB_PROJECT="ssl-linear-probe"

cd $(dirname "$0")

echo "==== ENV INFO ===="
echo "Job ID: $SLURM_JOB_ID"
hostname
date
nvidia-smi

echo "Starting comprehensive evaluation..."
echo "This will compare:"
echo "1. Random ViT - Zero-shot (baseline)"
echo "2. Random ViT - Full fine-tuning"
echo "3. SSL Pretrained - Linear probe"
echo "4. SSL Pretrained - Full fine-tuning"
echo "================================================"

python linear_probe_checkpoints.py \
    --comprehensive_eval \
    --use_wandb \
    --probe_epochs 90 \
    --probe_lr 1e-3 \
    --probe_wd 1e-4 \
    --finetune_epochs 90 \
    --finetune_lr 1e-3 \
    --batch_size 2048 \
    --seed 42 \
    --ssl_checkpoint ./outputs/mae_lr3e-4_decl4_mask0.75_normfalse_210133/encoder.pth

echo "Comprehensive evaluation completed!"
