# MAE Self-Supervised Learning on Galaxy10-DECALS

This repository implements Masked Autoencoder (MAE) self-supervised learning on the Galaxy10-DECALS dataset using Vision Transformers. The project explores MAE pre-training on astronomical galaxy images and evaluates learned representations through comprehensive linear probing and fine-tuning experiments. Pre-trained encoder can be accessed at https://drive.google.com/file/d/1oQ1woMGIf8wXCsmp_DIrCNTUvR1n6JuM/view?usp=sharing. All experiments are run on Oscar.

## Results

- **Best Configuration**: 66.23% accuracy with linear probing (no augmentation)
- **SSL vs Random**: 66.23% (SSL) vs 64.21% (random fine-tuning)
- **Full Fine-tuning**: 77.51% accuracy with SSL pre-training
- **Optimal Mask Ratio**: 0.75 (tested 0.65-0.90 range)

## Reconstruction Example
![MAE reconstructions](reconstruction_example.png)

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate galaxy-mae
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv galaxy-mae
source galaxy-mae/bin/activate  # On Windows: galaxy-mae\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The Galaxy10-DECALS dataset will be automatically downloaded when running the experiments. It contains 17,736 galaxy images across 10 classes.

## Running Experiments

### 1. MAE Pre-training

```bash
# Run grid search for hyperparameter optimization
sbatch run_ssl

# Or run single experiment
python main.py \
    --output_dir ./outputs/mae_experiment \
    --learning_rate 3e-4 \
    --decoder_layers 4 \
    --mask_ratio 0.75 \
    --normalize_pixel_loss false \
    --num_train_epochs 100
```

### 2. Linear Probing and Evaluation

```bash
# Comprehensive evaluation (zero-shot, random fine-tuning, SSL linear probe, SSL fine-tuning)
sbatch run_linear_probe.sh

# Or run directly
python linear_probe_checkpoints.py \
    --comprehensive_eval \
    --ssl_checkpoint ./outputs/mae_lr3e-4_decl4_mask0.75_normfalse_210133/encoder.pth \
    --probe_epochs 80 \
    --probe_lr 1e-3 \
    --finetune_epochs 80 \
    --finetune_lr 1e-3 \
    --batch_size 2048
```

### 3. Custom Checkpoint Evaluation

If you have your own pre-trained checkpoint:

```bash
python linear_probe_checkpoints.py \
    --comprehensive_eval \
    --ssl_checkpoint /path/to/your/encoder.pth \
    --probe_epochs 80 \
    --probe_lr 1e-3 \
    --finetune_epochs 80 \
    --finetune_lr 1e-3 \
    --batch_size 2048
```

## Key Files

- `main.py`: MAE pre-training implementation
- `linear_probe_checkpoints.py`: Comprehensive evaluation pipeline
- `run_ssl`: SLURM script for pre-training experiments
- `run_linear_probe.sh`: SLURM script for evaluation experiments

## Architecture

- **Model**: Custom ViT with 384 hidden size, 12 layers, 6 attention heads
- **Pre-training**: No data augmentation (following MAE protocol)
- **Evaluation**: Linear probing and full fine-tuning with/without augmentation
- **Framework**: HuggingFace Transformers with PyTorch

## Hyperparameter Search Results

| Parameter | Tested Range | Best Value |
|-----------|--------------|------------|
| Learning Rate | 1e-4, 3e-4, 1e-3 | 3e-4 |
| Decoder Layers | 2, 4, 8 | 4 |
| Mask Ratio | 0.65-0.90 | 0.75 |
| Pixel Normalization | true, false | false |

## Ablation Studies

1. **Mask Ratio Analysis**: Comprehensive search from 0.65 to 0.90
2. **Pixel Normalization**: Tested Kaiming He's "pixels with normalization" approach
3. **Data Augmentation**: Compared with/without augmentation during probing
4. **Baseline Comparison**: Random ViT vs SSL pre-trained models

## Requirements

- Python 3.10+
- PyTorch 2.5.1 with CUDA 12.1
- HuggingFace Transformers 4.49.0
- Weights & Biases for experiment tracking
- SLURM for cluster job management

## Citation

If you use this code, please cite the original MAE paper:

```bibtex
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
