#!/bin/bash
#SBATCH --job-name=linear_probe_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-3
#SBATCH --time=05:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=zhenkeliu@163.com
#SBATCH --output=linear_probe_array_%A_%a.out
#SBATCH --exclude=gpu3001,gpu3002,gpu2704

source ~/miniconda3/etc/profile.d/conda.sh
conda activate galaxy-mae

export WANDB_ENTITY="sizchode-brown-university"
export WANDB_PROJECT="ssl-linear-probe"

# Ensure we're in the correct directory
cd /users/zliu328/SSL-Project || cd ~/SSL-Project || cd $(dirname "$0")

echo "==== ENV INFO ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Working Directory: $(pwd)"
echo "linear_probe_checkpoints.py exists: $([ -f linear_probe_checkpoints.py ] && echo 'YES' || echo 'NO')"
hostname
date
nvidia-smi

# Map array task ID to experiment
EXPERIMENTS=(random_lp ssl_lp random_ft ssl_ft)
EXPERIMENT_NAMES=("Random ViT Linear Probing" "SSL Pretrained Linear Probing" "Random ViT Full Fine-tuning" "SSL Pretrained Full Fine-tuning")

EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_NAME=${EXPERIMENT_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "================================================"
echo "Running Experiment $SLURM_ARRAY_TASK_ID: $EXPERIMENT_NAME"
echo "Experiment: $EXPERIMENT"
echo "================================================"

# Common arguments
python linear_probe_checkpoints.py \
    --comprehensive_eval \
    --experiment $EXPERIMENT \
    --use_wandb \
    --probe_epochs 90 \
    --probe_lr 1e-3 \
    --probe_wd 1e-4 \
    --finetune_epochs 90 \
    --finetune_lr 1e-3 \
    --batch_size 2048 \
    --finetune_batch_size 512 \
    --finetune_grad_accum 4 \
    --seed 42 \
    --ssl_checkpoint ./outputs/mae_lr3e-4_decl4_mask0.75_normfalse_172129/encoder.pth

EXIT_CODE=$?

echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment $SLURM_ARRAY_TASK_ID ($EXPERIMENT) completed successfully!"
else
    echo "✗ Experiment $SLURM_ARRAY_TASK_ID ($EXPERIMENT) failed with exit code $EXIT_CODE"
fi
echo "================================================"

exit $EXIT_CODE

