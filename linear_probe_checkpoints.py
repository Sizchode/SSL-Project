#!/usr/bin/env python3
"""
Linear probing script for pre-trained MAE encoders.
Loads checkpoints with different mask ratios and runs linear probing.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import wandb

from transformers import ViTMAEConfig, ViTMAEModel, ViTImageProcessor


class HFWithLabel(torch.utils.data.Dataset):
    """(x, y) for linear probe; no augmentation for fair comparison."""
    def __init__(self, hf_split, image_size=256, train=True, mean=None, std=None):
        self.split = hf_split
        # No augmentation for both train and test to ensure fair comparison
        if mean is not None and std is not None:
            self.tx = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
    
    def __len__(self): 
        return len(self.split)
    
    def __getitem__(self, i):
        r = self.split[i]
        img = r["image"]
        if not hasattr(img, "mode"):
            img = Image.fromarray(np.array(img))
        x = self.tx(img)
        y = int(r["label"])
        return x, y


def get_args():
    p = argparse.ArgumentParser(description="Linear probe pre-trained MAE encoders + comprehensive evaluation")
    p.add_argument("--dataset", default="matthieulel/galaxy10_decals", help="HF dataset id")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--out_dir", default="outputs", help="where to save results")
    p.add_argument("--seed", type=int, default=42)
    
    # Linear probe parameters
    p.add_argument("--probe_epochs", type=int, default=90, help="Linear probe epochs")
    p.add_argument("--probe_lr", type=float, default=1e-3, help="Linear probe learning rate")
    p.add_argument("--probe_wd", type=float, default=1e-4, help="Linear probe weight decay")
    p.add_argument("--batch_size", type=int, default=64, help="Linear probe batch size")
    
    # Comprehensive evaluation parameters
    p.add_argument("--comprehensive_eval", action="store_true", help="Run comprehensive evaluation")
    p.add_argument("--ssl_checkpoint", 
                   default="/users/zliu328/ssl/outputs/mae_lr3e-4_decl4_mask0.75_normfalse_210133/encoder.pth",
                   help="Path to SSL pretrained encoder for comprehensive eval")
    p.add_argument("--finetune_epochs", type=int, default=90, help="Full fine-tuning epochs")
    p.add_argument("--finetune_lr", type=float, default=1e-3, help="Full fine-tuning learning rate")
    
    # Wandb arguments
    p.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--project", default="ssl-linear-probe", help="Wandb project name")
    
    return p.parse_args()


def load_encoder_from_checkpoint(checkpoint_path, device):
    """Load encoder from checkpoint with proper configuration."""
    print(f"[load] Loading encoder from: {checkpoint_path}")
    
    # Load the checkpoint to get the config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Create encoder config (matching the pre-training config)
    enc_cfg = ViTMAEConfig(
        image_size=256,
        patch_size=16,  # Fixed to match pre-training
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=384 * 4,
        qkv_bias=True,
    )
    
    # Create encoder model
    encoder = ViTMAEModel(enc_cfg).to(device)
    
    # Load state dict
    sd = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    
    if missing or unexpected:
        print(f"[load] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing keys: {missing[:5]}...")  # Show first 5
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}...")  # Show first 5
    
    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    
    print(f"[load] Encoder loaded successfully")
    return encoder


def run_linear_probe(encoder, train_loader, test_loader, args, device, checkpoint_name):
    """Run linear probing on the encoder."""
    print(f"\n[probe] Starting linear probe for {checkpoint_name}")
    print(f"  Epochs: {args.probe_epochs}, LR: {args.probe_lr}, WD: {args.probe_wd}")
    
    # Get number of classes
    NUM_CLASSES = 10  # Galaxy10 has 10 classes
    
    # Linear head
    head = nn.Linear(384, NUM_CLASSES).to(device)  # 384 is hidden_size
    ce = nn.CrossEntropyLoss()
    opt_h = torch.optim.AdamW(head.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_h, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    def _cls_readout(last_hidden):  # (B, N, D) -> (B, D)
        return last_hidden.mean(dim=1)  # Mean pooling over all tokens (MAE has no CLS token)
    
    best_acc = 0.0
    results = []
    patience_counter = 0
    early_stop_patience = 30  # Stop if no improvement for 15 epochs
    
    for ep in range(1, args.probe_epochs + 1):
        # ---- train head ----
        head.train()
        running_loss, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt_h.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    feats = _cls_readout(encoder(pixel_values=x).last_hidden_state).float()
                logits = head(feats)
                loss = ce(logits, y)
            
            loss.backward()
            opt_h.step()
            b = x.size(0)
            running_loss += loss.item() * b
            seen += b
        
        train_loss = running_loss / max(1, seen)
        
        # ---- eval ----
        head.eval()
        correct, count = 0, 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feats = _cls_readout(encoder(pixel_values=x).last_hidden_state).float()
                logits = head(feats)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                count += y.numel()
        
        acc = correct / max(1, count)
        
        # Check for improvement
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update learning rate scheduler
        scheduler.step(acc)
        current_lr = opt_h.param_groups[0]['lr']
        
        print(f"[probe] {checkpoint_name} | epoch {ep:03d}/{args.probe_epochs} | "
              f"train_loss={train_loss:.4f} | test_acc={acc*100:.2f}% (best {best_acc*100:.2f}%) | "
              f"lr={current_lr:.2e} | patience={patience_counter}")
        
        results.append({
            'epoch': ep,
            'train_loss': train_loss,
            'test_acc': acc,
            'best_acc': best_acc,
            'learning_rate': current_lr
        })
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                f"probe/{checkpoint_name}/epoch": ep,
                f"probe/{checkpoint_name}/train_loss": train_loss,
                f"probe/{checkpoint_name}/test_acc": acc,
                f"probe/{checkpoint_name}/best_acc": best_acc,
                f"probe/{checkpoint_name}/learning_rate": current_lr
            }, step=ep)
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"[probe] {checkpoint_name} | Early stopping at epoch {ep} (no improvement for {early_stop_patience} epochs)")
            break
    
    print(f"[probe] {checkpoint_name} | Final best test acc: {best_acc*100:.2f}%")
    return best_acc, results


def create_vit_model(device, pretrained_weights=None):
    """Create ViT model - either random or load pretrained weights."""
    # Create encoder config (matching the pre-training config)
    enc_cfg = ViTMAEConfig(
        image_size=256,
        patch_size=16,
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=384 * 4,
        qkv_bias=True,
    )
    
    # Create encoder model
    encoder = ViTMAEModel(enc_cfg).to(device)
    
    if pretrained_weights is not None:
        print(f"[load] Loading pretrained weights from: {pretrained_weights}")
        sd = torch.load(pretrained_weights, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[load] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[load] Using randomly initialized ViT")
    
    return encoder


def run_full_finetuning(encoder, train_loader, test_loader, args, device, experiment_name):
    """Run full fine-tuning (unfrozen backbone + classification head)."""
    print(f"\n[finetune] Starting full fine-tuning for {experiment_name}")
    
    # Unfreeze encoder
    for p in encoder.parameters():
        p.requires_grad_(True)
    encoder.train()
    
    # Get number of classes
    NUM_CLASSES = 10  # Galaxy10 has 10 classes
    
    # Classification head
    head = nn.Linear(384, NUM_CLASSES).to(device)
    ce = nn.CrossEntropyLoss()
    
    # Optimizer for both encoder and head
    all_params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(all_params, lr=args.finetune_lr, weight_decay=args.probe_wd)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    def _cls_readout(last_hidden):  # (B, N, D) -> (B, D)
        return last_hidden.mean(dim=1)  # Mean pooling over all tokens
    
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15
    
    for ep in range(1, args.finetune_epochs + 1):
        # ---- train ----
        encoder.train()
        head.train()
        running_loss, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                feats = _cls_readout(encoder(pixel_values=x).last_hidden_state).float()
                logits = head(feats)
                loss = ce(logits, y)
            
            loss.backward()
            opt.step()
            b = x.size(0)
            running_loss += loss.item() * b
            seen += b
        
        train_loss = running_loss / max(1, seen)
        
        # ---- eval ----
        encoder.eval()
        head.eval()
        correct, count = 0, 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feats = _cls_readout(encoder(pixel_values=x).last_hidden_state).float()
                logits = head(feats)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                count += y.numel()
        
        acc = correct / max(1, count)
        
        # Check for improvement
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step(acc)
        current_lr = opt.param_groups[0]['lr']
        
        print(f"[finetune] {experiment_name} | epoch {ep:03d}/{args.finetune_epochs} | "
              f"train_loss={train_loss:.4f} | test_acc={acc*100:.2f}% (best {best_acc*100:.2f}%) | "
              f"lr={current_lr:.2e} | patience={patience_counter}")
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                f"finetune/{experiment_name}/epoch": ep,
                f"finetune/{experiment_name}/train_loss": train_loss,
                f"finetune/{experiment_name}/test_acc": acc,
                f"finetune/{experiment_name}/best_acc": best_acc,
                f"finetune/{experiment_name}/learning_rate": current_lr
            }, step=ep)
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"[finetune] {experiment_name} | Early stopping at epoch {ep}")
            break
    
    print(f"[finetune] {experiment_name} | Final best test acc: {best_acc*100:.2f}%")
    return best_acc


def run_zero_shot_evaluation(encoder, test_loader, device, experiment_name):
    """Run zero-shot evaluation (no training, just random predictions)."""
    print(f"\n[zeroshot] Starting zero-shot evaluation for {experiment_name}")
    
    encoder.eval()
    correct, count = 0, 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Random predictions (10 classes)
            pred = torch.randint(0, 10, (y.size(0),), device=device)
            correct += (pred == y).sum().item()
            count += y.numel()
    
    acc = correct / max(1, count)
    print(f"[zeroshot] {experiment_name} | Zero-shot test acc: {acc*100:.2f}%")
    
    return acc


def main():
    args = get_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.project,
            entity="sizchode-brown-university",
            name="linear-probe-mask-comparison",
            config=vars(args),
            settings=wandb.Settings(start_method="fork")
        )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    
    # Load dataset
    print(f"[info] Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    train_split, test_split = ds["train"], ds["test"]
    print(f"[info] Splits -> train: {len(train_split)} | test: {len(test_split)}")
    
    # Use pre-trained processor statistics
    processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
    dataset_mean = processor.image_mean
    dataset_std = processor.image_std
    print(f"[info] Using ImageNet statistics: mean={dataset_mean}, std={dataset_std}")
    
    # Create data loaders
    train_sup = HFWithLabel(train_split, image_size=args.image_size, train=True, 
                           mean=dataset_mean, std=dataset_std)
    test_sup = HFWithLabel(test_split, image_size=args.image_size, train=False, 
                          mean=dataset_mean, std=dataset_std)
    
    train_loader = DataLoader(
        train_sup, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_sup, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )
    
    if args.comprehensive_eval:
        # Comprehensive evaluation: Random vs SSL
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION: RANDOM vs SSL")
        print("="*80)
        
        results = {}
        
        # 1. Random ViT - Zero-shot
        print("\n1. RANDOM ViT - ZERO-SHOT EVALUATION")
        print("-" * 50)
        random_encoder = create_vit_model(device, pretrained_weights=None)
        zero_shot_acc = run_zero_shot_evaluation(random_encoder, test_loader, device, "random_vit")
        results["random_vit_zeroshot"] = zero_shot_acc
        del random_encoder
        torch.cuda.empty_cache()
        
        # 2. Random ViT - Full Fine-tuning
        print("\n2. RANDOM ViT - FULL FINE-TUNING")
        print("-" * 50)
        random_encoder = create_vit_model(device, pretrained_weights=None)
        random_ft_acc = run_full_finetuning(random_encoder, train_loader, test_loader, args, device, "random_vit")
        results["random_vit_finetune"] = random_ft_acc
        del random_encoder
        torch.cuda.empty_cache()
        
        # 3. SSL Pretrained - Linear Probe
        print("\n3. SSL PRETRAINED - LINEAR PROBE")
        print("-" * 50)
        if os.path.exists(args.ssl_checkpoint):
            ssl_encoder = create_vit_model(device, pretrained_weights=args.ssl_checkpoint)
            ssl_probe_acc, _ = run_linear_probe(ssl_encoder, train_loader, test_loader, args, device, "ssl_pretrained")
            results["ssl_pretrained_probe"] = ssl_probe_acc
            del ssl_encoder
            torch.cuda.empty_cache()
        else:
            print(f"[warning] SSL checkpoint not found: {args.ssl_checkpoint}")
            results["ssl_pretrained_probe"] = 0.0
        
        # 4. SSL Pretrained - Full Fine-tuning
        print("\n4. SSL PRETRAINED - FULL FINE-TUNING")
        print("-" * 50)
        if os.path.exists(args.ssl_checkpoint):
            ssl_encoder = create_vit_model(device, pretrained_weights=args.ssl_checkpoint)
            ssl_ft_acc = run_full_finetuning(ssl_encoder, train_loader, test_loader, args, device, "ssl_pretrained")
            results["ssl_pretrained_finetune"] = ssl_ft_acc
            del ssl_encoder
            torch.cuda.empty_cache()
        else:
            print(f"[warning] SSL checkpoint not found: {args.ssl_checkpoint}")
            results["ssl_pretrained_finetune"] = 0.0
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS SUMMARY")
        print("="*80)
        print(f"{'Experiment':<30s} {'Accuracy':<10s} {'Improvement':<15s}")
        print("-" * 80)
        
        baseline = results["random_vit_zeroshot"]
        for exp_name, acc in results.items():
            improvement = f"+{(acc-baseline)*100:.1f}%" if acc > baseline else f"{(acc-baseline)*100:.1f}%"
            print(f"{exp_name:<30s} {acc*100:>8.2f}% {improvement:>12s}")
        
        print("-" * 80)
        print(f"Random baseline (10 classes): {baseline*100:.2f}%")
        print(f"Best result: {max(results.values())*100:.2f}%")
        print("="*80)
        
        # Log summary to wandb
        if args.use_wandb:
            wandb.log({
                "summary/random_vit_zeroshot": results["random_vit_zeroshot"],
                "summary/random_vit_finetune": results["random_vit_finetune"],
                "summary/ssl_pretrained_probe": results["ssl_pretrained_probe"],
                "summary/ssl_pretrained_finetune": results["ssl_pretrained_finetune"],
                "summary/best_result": max(results.values()),
                "summary/ssl_improvement": results["ssl_pretrained_probe"] - results["random_vit_zeroshot"],
            })
            wandb.finish()
    
    else:
        # Original linear probe evaluation
        # Define checkpoints to test
        checkpoints = [
            ("mask0.6_normfalse", "/users/zliu328/ssl/outputs/mae_lr1e-3_decl2_mask0.6_normfalse_184253/encoder.pth"),
            ("mask0.6_normtrue", "/users/zliu328/ssl/outputs/mae_lr1e-3_decl2_mask0.6_normtrue_184253/encoder.pth"),
            ("mask0.75_normfalse", "/users/zliu328/ssl/outputs/mae_lr1e-3_decl2_mask0.75_normfalse_184253/encoder.pth"),
            ("mask0.75_normtrue", "/users/zliu328/ssl/outputs/mae_lr1e-3_decl2_mask0.75_normtrue_184253/encoder.pth"),
        ]
        
        # Run linear probing on each checkpoint
        results_summary = {}
        
        for checkpoint_name, checkpoint_path in checkpoints:
            if not os.path.exists(checkpoint_path):
                print(f"[warning] Checkpoint not found: {checkpoint_path}")
                continue
            
            try:
                # Load encoder
                encoder = load_encoder_from_checkpoint(checkpoint_path, device)
                
                # Run linear probe
                best_acc, results = run_linear_probe(
                    encoder, train_loader, test_loader, args, device, checkpoint_name
                )
                
                results_summary[checkpoint_name] = best_acc
                
                # Clean up
                del encoder
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"[error] Failed to run linear probe for {checkpoint_name}: {e}")
                continue
        
        # Print summary
        print("\n" + "="*60)
        print("LINEAR PROBE RESULTS SUMMARY")
        print("="*60)
        for checkpoint_name, best_acc in results_summary.items():
            print(f"{checkpoint_name:20s}: {best_acc*100:.2f}%")
        
        # Log summary to wandb
        if args.use_wandb:
            wandb.log({
                "summary/mask0.6_normfalse": results_summary.get("mask0.6_normfalse", 0),
                "summary/mask0.6_normtrue": results_summary.get("mask0.6_normtrue", 0),
                "summary/mask0.75_normfalse": results_summary.get("mask0.75_normfalse", 0),
                "summary/mask0.75_normtrue": results_summary.get("mask0.75_normtrue", 0),
            })
            wandb.finish()
        
        print("="*60)


if __name__ == "__main__":
    main()
