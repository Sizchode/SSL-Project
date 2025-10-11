#!/usr/bin/env python3

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

from transformers import ViTConfig, ViTModel

class HFWithLabel(torch.utils.data.Dataset):
    def __init__(self, hf_split, image_size=256, train=True, mean=None, std=None):
        self.split = hf_split
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
    p.add_argument("--probe_epochs", type=int, default=90, help="Linear probe epochs")
    p.add_argument("--probe_lr", type=float, default=1e-3, help="Linear probe learning rate")
    p.add_argument("--probe_wd", type=float, default=1e-4, help="Linear probe weight decay")
    p.add_argument("--batch_size", type=int, default=64, help="Linear probe batch size")
    p.add_argument("--comprehensive_eval", action="store_true", help="Run comprehensive evaluation")
    p.add_argument("--experiment", type=str, default="all",
                   choices=["all", "zero_random_probe", "ssl_probe", "random_ft", "ssl_ft"],
                   help="Which experiment to run (for parallel execution)")
    p.add_argument("--ssl_checkpoint",
                   default="./outputs/mae_lr3e-4_decl4_mask0.75_normfalse_172129/encoder.pth",
                   help="Path to SSL pretrained encoder for comprehensive eval")
    p.add_argument("--finetune_epochs", type=int, default=90, help="Full fine-tuning epochs")
    p.add_argument("--finetune_lr", type=float, default=1e-3, help="Full fine-tuning learning rate")
    p.add_argument("--finetune_batch_size", type=int, default=512, help="Fine-tuning batch size per step")
    p.add_argument("--finetune_grad_accum", type=int, default=4, help="Gradient accumulation steps for fine-tuning")
    p.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--project", default="ssl-linear-probe", help="Wandb project name")

    return p.parse_args()

def run_linear_probe(encoder, train_loader, test_loader, args, device, checkpoint_name):
    print(f"\n[probe] Starting linear probe for {checkpoint_name}")
    print(f"  Epochs: {args.probe_epochs}, LR: {args.probe_lr}, WD: {args.probe_wd}")
    NUM_CLASSES = 10
    head = nn.Linear(384, NUM_CLASSES).to(device)
    ce = nn.CrossEntropyLoss()
    opt_h = torch.optim.AdamW(head.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=args.probe_epochs, eta_min=0)

    def _cls_readout(last_hidden):
        return last_hidden.mean(dim=1)

    best_acc = 0.0
    results = []
    patience_counter = 0
    early_stop_patience = 30

    for ep in range(1, args.probe_epochs + 1):
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
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
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

        if args.use_wandb:
            wandb.log({
                f"probe/{checkpoint_name}/epoch": ep,
                f"probe/{checkpoint_name}/train_loss": train_loss,
                f"probe/{checkpoint_name}/test_acc": acc,
                f"probe/{checkpoint_name}/best_acc": best_acc,
                f"probe/{checkpoint_name}/learning_rate": current_lr
            }, step=ep)

        if patience_counter >= early_stop_patience:
            print(f"[probe] {checkpoint_name} | Early stopping at epoch {ep} (no improvement for {early_stop_patience} epochs)")
            break

    print(f"[probe] {checkpoint_name} | Final best test acc: {best_acc*100:.2f}%")
    return best_acc, results

def create_vit_model(device, pretrained_weights=None):
    
    enc_cfg = ViTConfig(
        image_size=256,
        patch_size=16,
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=384 * 4,
        qkv_bias=True,
    )

    encoder = ViTModel(enc_cfg).to(device)

    if pretrained_weights is not None:
        print(f"[load] Loading pretrained weights from: {pretrained_weights}")
        sd = torch.load(pretrained_weights, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[load] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[load] Using randomly initialized ViT")

    return encoder

def run_full_finetuning(encoder, train_dataset, test_loader, args, device, experiment_name):
    
    print(f"\n[finetune] Starting full fine-tuning for {experiment_name}")

    gradient_accumulation_steps = args.finetune_grad_accum
    effective_batch_size = args.finetune_batch_size * gradient_accumulation_steps
    print(f"[finetune] Batch size: {args.finetune_batch_size}, Grad accum: {gradient_accumulation_steps}, Effective batch: {effective_batch_size}")

    train_loader_ft = DataLoader(
        train_dataset, batch_size=args.finetune_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )

    for p in encoder.parameters():
        p.requires_grad_(True)
    encoder.train()

    NUM_CLASSES = 10

    head = nn.Linear(384, NUM_CLASSES).to(device)
    ce = nn.CrossEntropyLoss()

    all_params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(all_params, lr=args.finetune_lr, weight_decay=args.probe_wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.finetune_epochs, eta_min=0)

    def _cls_readout(last_hidden):
        return last_hidden.mean(dim=1)

    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 30

    for ep in range(1, args.finetune_epochs + 1):
        encoder.train()
        head.train()
        running_loss, seen = 0.0, 0
        for batch_idx, (x, y) in enumerate(train_loader_ft):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                feats = _cls_readout(encoder(pixel_values=x).last_hidden_state).float()
                logits = head(feats)
                loss = ce(logits, y)
                loss = loss / gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            b = x.size(0)
            running_loss += loss.item() * b * gradient_accumulation_steps
            seen += b

        train_loss = running_loss / max(1, seen)

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
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
        current_lr = opt.param_groups[0]['lr']

        print(f"[finetune] {experiment_name} | epoch {ep:03d}/{args.finetune_epochs} | "
              f"train_loss={train_loss:.4f} | test_acc={acc*100:.2f}% (best {best_acc*100:.2f}%) | "
              f"lr={current_lr:.2e} | patience={patience_counter}")

        if args.use_wandb:
            wandb.log({
                f"finetune/{experiment_name}/epoch": ep,
                f"finetune/{experiment_name}/train_loss": train_loss,
                f"finetune/{experiment_name}/test_acc": acc,
                f"finetune/{experiment_name}/best_acc": best_acc,
                f"finetune/{experiment_name}/learning_rate": current_lr
            }, step=ep)

        if patience_counter >= early_stop_patience:
            print(f"[finetune] {experiment_name} | Early stopping at epoch {ep}")
            break

    print(f"[finetune] {experiment_name} | Final best test acc: {best_acc*100:.2f}%")
    return best_acc

def run_zero_shot_evaluation(encoder, test_loader, device, experiment_name):
    
    print(f"\n[zeroshot] Starting zero-shot evaluation for {experiment_name}")

    encoder.eval()
    correct, count = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = torch.randint(0, 10, (y.size(0),), device=device)
            correct += (pred == y).sum().item()
            count += y.numel()

    acc = correct / max(1, count)
    print(f"[zeroshot] {experiment_name} | Zero-shot test acc: {acc*100:.2f}%")

    return acc

def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.use_wandb:
        if args.comprehensive_eval:
            run_name = f"linear-probe-{args.experiment}"
        else:
            run_name = "linear-probe-mask-comparison"

        wandb.init(
            project=args.project,
            entity="sizchode-brown-university",
            name=run_name,
            config=vars(args),
            settings=wandb.Settings(start_method="fork")
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    print(f"[info] Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    train_split, test_split = ds["train"], ds["test"]
    print(f"[info] Splits -> train: {len(train_split)} | test: {len(test_split)}")

    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std = [0.229, 0.224, 0.225]
    print(f"[info] Using ImageNet statistics: mean={dataset_mean}, std={dataset_std}")

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
        print("\n" + "="*80)
        print(f"COMPREHENSIVE EVALUATION: {args.experiment.upper()}")
        print("="*80)

        results = {}

        if args.experiment in ["all", "zero_random_probe"]:
            print("\n1. RANDOM ViT - ZERO-SHOT EVALUATION")
            print("-" * 50)
            random_encoder = create_vit_model(device, pretrained_weights=None)
            zero_shot_acc = run_zero_shot_evaluation(random_encoder, test_loader, device, "random_vit")
            results["random_vit_zeroshot"] = zero_shot_acc
            del random_encoder
            torch.cuda.empty_cache()

            print("\n2. RANDOM ViT - LINEAR PROBE")
            print("-" * 50)
            random_encoder = create_vit_model(device, pretrained_weights=None)
            random_probe_acc, _ = run_linear_probe(random_encoder, train_loader, test_loader, args, device, "random_vit")
            results["random_vit_probe"] = random_probe_acc
            del random_encoder
            torch.cuda.empty_cache()

        if args.experiment in ["all", "ssl_probe"]:
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

        if args.experiment in ["all", "random_ft"]:
            print("\n4. RANDOM ViT - FULL FINE-TUNING")
            print("-" * 50)
            random_encoder = create_vit_model(device, pretrained_weights=None)
            random_ft_acc = run_full_finetuning(random_encoder, train_sup, test_loader, args, device, "random_vit")
            results["random_vit_finetune"] = random_ft_acc
            del random_encoder
            torch.cuda.empty_cache()

        if args.experiment in ["all", "ssl_ft"]:
            print("\n5. SSL PRETRAINED - FULL FINE-TUNING")
            print("-" * 50)
            if os.path.exists(args.ssl_checkpoint):
                ssl_encoder = create_vit_model(device, pretrained_weights=args.ssl_checkpoint)
                ssl_ft_acc = run_full_finetuning(ssl_encoder, train_sup, test_loader, args, device, "ssl_pretrained")
                results["ssl_pretrained_finetune"] = ssl_ft_acc
                del ssl_encoder
                torch.cuda.empty_cache()
            else:
                print(f"[warning] SSL checkpoint not found: {args.ssl_checkpoint}")
                results["ssl_pretrained_finetune"] = 0.0

        print("\n" + "="*80)
        print(f"EVALUATION RESULTS: {args.experiment.upper()}")
        print("="*80)
        print(f"{'Experiment':<30s} {'Accuracy':<10s}")
        print("-" * 80)

        for exp_name, acc in results.items():
            print(f"{exp_name:<30s} {acc*100:>8.2f}%")

        print("="*80)

        if args.use_wandb:
            log_dict = {f"summary/{k}": v for k, v in results.items()}
            wandb.log(log_dict)
            wandb.finish()

if __name__ == "__main__":
    main()
