import os, argparse, random
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import wandb
import cv2
import matplotlib.pyplot as plt

from transformers import (
    ViTMAEConfig, ViTMAEForPreTraining, ViTMAEModel,
    Trainer, TrainingArguments, DataCollatorWithPadding
)




# -------------------------------
# Data
# -------------------------------

class MAEDataset(torch.utils.data.Dataset):
    """Dataset for MAE training with Trainer."""
    def __init__(self, hf_split, image_size: int = 256, mean=None, std=None):
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
    
    def __getitem__(self, idx):
        item = self.split[idx]
        img = item["image"]
        if not hasattr(img, "mode"):
            img = Image.fromarray(np.array(img))
        
        pixel_values = self.tx(img)
        
        return {
            "pixel_values": pixel_values
        }
class HFWithLabel(torch.utils.data.Dataset):
    """(x, y) for linear probe; light aug on train, plain resize on test."""
    def __init__(self, hf_split, image_size=256, train=True, mean=None, std=None):
        self.split = hf_split
        if train:
            if mean is not None and std is not None:
                self.tx = transforms.Compose([
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            else:
                self.tx = transforms.Compose([
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
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
    def __len__(self): return len(self.split)
    def __getitem__(self, i):
        r = self.split[i]
        img = r["image"]
        if not hasattr(img, "mode"):
            img = Image.fromarray(np.array(img))
        x = self.tx(img)
        y = int(r["label"])
        return x, y


# -------------------------------
# Model builders & summaries
# -------------------------------
def build_vitmae(
    image_size: int = 256,
    patch_size: int = 16,
    mask_ratio: float = 0.75,
    hidden_size: int = 384,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 6,
    decoder_hidden_size: int = 512,
    decoder_num_hidden_layers: int = 8,
    norm_pix_loss: bool = True,
) -> ViTMAEForPreTraining:
    cfg = ViTMAEConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        mask_ratio=mask_ratio,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        qkv_bias=True,
        decoder_hidden_size=decoder_hidden_size,
        decoder_num_hidden_layers=decoder_num_hidden_layers,
        norm_pix_loss=norm_pix_loss, 
    )
    model = ViTMAEForPreTraining(cfg)
    return model

def compute_dataset_stats(dataset, image_size=256, batch_size=64):
    """计算数据集的均值和标准差"""
    print("[info] Computing dataset statistics...")
    
    to_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    def collate_fn(batch):
        imgs = []
        for item in batch:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            imgs.append(to_tensor(img))
        return torch.stack(imgs)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    n_pixels = 0
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    
    for i, batch in enumerate(loader):
        if i % 100 == 0:
            print(f"  Processing batch {i}/{len(loader)}")
        
        b, c, h, w = batch.shape
        n_pixels += b * h * w
        
        channel_sum += batch.sum(dim=[0, 2, 3]).double()
        channel_sum_sq += (batch.double() ** 2).sum(dim=[0, 2, 3])
    
    image_mean = channel_sum / n_pixels
    image_std = torch.sqrt(channel_sum_sq / n_pixels - image_mean ** 2)
    
    print(f"[info] Dataset statistics:")
    print(f"  Mean: {image_mean}")
    print(f"  Std: {image_std}")
    
    return image_mean.float(), image_std.float()

def compute_dataset_stats(dataset, image_size=256, num_samples=1000):
    """Compute mean and std for dataset normalization."""
    print(f"[info] Computing dataset statistics from {min(num_samples, len(dataset))} samples...")
    
    # Sample a subset for efficiency
    sample_size = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    # Collect pixel values
    pixel_values = []
    for idx in indices:
        item = dataset[idx]
        img = item["image"]
        if not hasattr(img, "mode"):
            img = Image.fromarray(np.array(img))
        
        # Resize and convert to tensor
        img = img.resize((image_size, image_size))
        img_tensor = transforms.ToTensor()(img)  # (3, H, W)
        pixel_values.append(img_tensor)
    
    # Stack and compute statistics
    pixel_values = torch.stack(pixel_values)  # (N, 3, H, W)
    image_mean = pixel_values.mean(dim=(0, 2, 3))  # (3,)
    image_std = pixel_values.std(dim=(0, 2, 3))   # (3,)
    
    print(f"[info] Dataset statistics:")
    print(f"  Mean: {image_mean}")
    print(f"  Std: {image_std}")
    
    return image_mean.float(), image_std.float()

def train_with_trainer(args, model, train_dataset, dataset_mean, dataset_std):
    """Create and return trainer - same pattern as MAEVIT notebook."""
    # Create run directory
    run_dir = os.path.join(args.out_dir, f"mae_lr{args.learning_rate}_decl{args.decoder_num_layers}_mask{args.mask_ratio}_norm{args.norm_pix_loss}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Training arguments - same as MAEVIT notebook
    training_args = TrainingArguments(
        remove_unused_columns=False,
        output_dir=run_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.05,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.95,
        logging_steps=5,
        save_steps=800,
        eval_strategy="steps",
        eval_steps=800,
        greater_is_better=True,
        save_total_limit=3,
        report_to="wandb" if args.use_wandb else None,
        bf16=True,
        dataloader_num_workers=4,
        gradient_accumulation_steps=2,  # As requested by user
        label_names=["pixel_values"],
    )
    
    # Create trainer with image_mean and image_std (same as MAEVIT notebook)
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        image_mean=dataset_mean,
        image_std=dataset_std,
        run_dir=run_dir,
        image_size=args.image_size,
    )
    
    # Start training
    print("Starting MAE pre-training...")
    trainer.train()
    
    return trainer, run_dir

class MAETrainer(Trainer):
    """Custom Trainer for MAE with reconstruction visualization."""
    
    def __init__(self, *args, image_mean=None, image_std=None, **kwargs):
        # Extract custom arguments before passing to parent
        self.image_size = kwargs.pop('image_size', 256)
        self.run_dir = kwargs.pop('run_dir', 'outputs')
        
        # Allow external direct input of IMAGE_MEAN / IMAGE_STD (same as MAEVIT notebook)
        if image_mean is not None and image_std is not None:
            self.image_mean = torch.tensor(image_mean).view(1, 1, 3)
            self.image_std  = torch.tensor(image_std).view(1, 1, 3)
        else:
            self.image_mean = None
            self.image_std = None
            
        super().__init__(*args, **kwargs)
        
    def show_image(self, image, title="", image_mean=0, image_std=1, ax=None):
        """Show image - exact same as MAEVIT notebook."""
        # image is [H, W, 3]
        assert image.shape[2] == 3
        
        # Handle None values for image_mean and image_std
        if image_mean is None:
            image_mean = 0
        if image_std is None:
            image_std = 1
            
        ax.imshow(torch.clip((image * image_std + image_mean) * 255, 0, 255).int())
        ax.set_title(title, fontsize=16)
        ax.axis("off")
        return

    def visualize(self, pixel_values: torch.Tensor, model):
        """Visualize function - exact same as MAEVIT notebook."""
        model.eval()
        outputs = model(pixel_values.to(next(model.parameters()).device))

        # logits -> 重建的图像
        y = model.unpatchify(outputs.logits)
        y = torch.einsum('nchw->nhwc', y).cpu()

        # mask
        mask = outputs.mask.detach()
        p = model.config.patch_size
        mask = mask.unsqueeze(-1).repeat(1, 1, p*p*3)
        mask = model.unpatchify(mask)
        mask = torch.einsum('nchw->nhwc', mask).cpu()

        # 原图
        x = torch.einsum('nchw->nhwc', pixel_values).cpu()
        im_masked = x * (1 - mask)
        im_paste = x * (1 - mask) + y * mask

        # 显式创建 figure
        fig, axs = plt.subplots(1, 7, figsize=(24, 24))
        self.show_image(x[0], "original", image_mean=self.image_mean, image_std=self.image_std, ax=axs[0])
        self.show_image(x[0], "original(norm)", ax=axs[1])
        self.show_image(im_masked[0], "masked", ax=axs[2])
        self.show_image(y[0], "reconstruction", ax=axs[3])
        self.show_image(im_paste[0], "reconstruction + visible", image_mean=self.image_mean, image_std=self.image_std, ax=axs[4])
        self.show_image(im_paste[0], "reconstruction + visible(norm)", ax=axs[5])
        self.show_image(1 - mask[0], "mask", ax=axs[6])

        plt.tight_layout()
        return fig

    @torch.no_grad()
    def _log_eval_visuals(self, eval_dataset=None, image_index: int = 1, batch_key: str = "pixel_values"):
        """Log evaluation visuals - exact same as MAEVIT notebook."""
        if not self.is_world_process_zero():
            return
        dl = self.get_eval_dataloader(eval_dataset)
        try:
            batch = next(iter(dl))
        except StopIteration:
            return

        if batch_key not in batch:
            print(f"[MyTrainer] batch 中找不到 '{batch_key}'；可视化跳过。")
            return

        pixel_values = batch[batch_key][image_index].unsqueeze(0)
        self.model.eval()

        # 生成图并写入 W&B
        fig = self.visualize(pixel_values, self.model)
        if hasattr(self, 'state') and hasattr(self.state, 'global_step'):
            try:
                import wandb
                wandb.log({f"eval/mae_reconstruction_image{image_index}": wandb.Image(fig)}, step=int(self.state.global_step))
            except:
                pass
        # 同步保存本地文件（可选）
        if hasattr(self, 'state') and hasattr(self.state, 'global_step'):
            fname = f"mae_reconstruction_step{self.state.global_step}.png"
            if hasattr(self, 'run_dir'):
                save_path = os.path.join(self.run_dir, fname)
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Evaluate with visualization - exact same as MAEVIT notebook."""
        metrics = super().evaluate(eval_dataset=eval_dataset,
                                   ignore_keys=ignore_keys,
                                   metric_key_prefix=metric_key_prefix)

        try:
            for i in range(9):
              self._log_eval_visuals(eval_dataset=eval_dataset, image_index=i, batch_key="pixel_values")
        except Exception as e:
            print(f"[MyTrainer] fail{e}")

        return metrics
    
    def _save_recon_panel(self, orig_pix, pred_patches, epoch_idx, k_samples=8):
        """Save MAE visualization using exact same pipeline as MAEVIT notebook."""
        with torch.no_grad():
            fig = self.visualize(orig_pix[:1], self.model)  # Use first sample
            
            save_path = os.path.join(self.run_dir, f"reconstruction_epoch_{epoch_idx}.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            
            try:
                import wandb
                wandb.log({
                    f"reconstruction/epoch_{epoch_idx}": wandb.Image(fig)
                })
            except:
                pass
            
            plt.close(fig)
            print(f"Saved reconstruction visualization to {save_path}")
            
            if hasattr(self, 'dataset_mean') and hasattr(self, 'dataset_std'):
                mean = self.dataset_mean.view(1, 3, 1, 1)
                std = self.dataset_std.view(1, 3, 1, 1)
                orig = orig * std + mean
                recon = recon * std + mean
            
            for i in range(k):
                # 1. Original image
                orig_img = (orig[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                
                # 2. Create masked image
                mask_i = mask[i].detach().cpu().numpy()  # (N,)
                h = w = int(np.sqrt(len(mask_i)))
                mask_2d = mask_i.reshape(h, w)
                
                # Create masked version by setting masked patches to gray
                masked_img = orig_img.copy()
                patch_size = self.image_size // h
                for patch_h in range(h):
                    for patch_w in range(w):
                        if mask_2d[patch_h, patch_w] == 1:  # masked patch
                            start_h = patch_h * patch_size
                            end_h = (patch_h + 1) * patch_size
                            start_w = patch_w * patch_size
                            end_w = (patch_w + 1) * patch_size
                            masked_img[start_h:end_h, start_w:end_w] = [128, 128, 128]  # gray
                
                # 3. Pure reconstruction
                recon_img = (recon[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                
                recon_visible = orig_img.copy()
                for patch_h in range(h):
                    for patch_w in range(w):
                        if mask_2d[patch_h, patch_w] == 1:  # masked patch
                            start_h = patch_h * patch_size
                            end_h = (patch_h + 1) * patch_size
                            start_w = patch_w * patch_size
                            end_w = (patch_w + 1) * patch_size
                            recon_visible[start_h:end_h, start_w:end_w] = recon_img[start_h:end_h, start_w:end_w]
                
                # Create 4-panel visualization
                panel_height = self.image_size
                panel_width = self.image_size
                
                # Create combined image (2x2 grid)
                combined = np.zeros((panel_height * 2, panel_width * 2, 3), dtype=np.uint8)
                combined[0:panel_height, 0:panel_width] = orig_img
                combined[0:panel_height, panel_width:panel_width*2] = masked_img
                combined[panel_height:panel_height*2, 0:panel_width] = recon_img
                combined[panel_height:panel_height*2, panel_width:panel_width*2] = recon_visible
                
                # Add labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                color = (255, 255, 255)
                thickness = 2
                
                cv2.putText(combined, "Original", (10, 30), font, font_scale, color, thickness)
                cv2.putText(combined, "Masked", (panel_width + 10, 30), font, font_scale, color, thickness)
                cv2.putText(combined, "Reconstruction", (10, panel_height + 30), font, font_scale, color, thickness)
                cv2.putText(combined, "Recon + Visible", (panel_width + 10, panel_height + 30), font, font_scale, color, thickness)
                
                # Save 4-panel visualization
                Image.fromarray(combined).save(
                    os.path.join(self.run_dir, "recon", f"mae_4panel_ep{epoch_idx:03d}_sample{i}.png")
                )
                
                # Log to wandb if available
                if hasattr(self, 'custom_args') and self.custom_args.use_wandb:
                    import wandb
                    wandb.log({
                        f"mae_4panel/epoch_{epoch_idx:03d}_sample_{i}": wandb.Image(combined)
                    })
    
    def _save_mask_visualization(self, pixel_values, epoch_idx, k_samples=4):
        """Save mask visualization showing which patches are masked."""
        with torch.no_grad():
            # Get mask from model
            outputs = self.model(pixel_values=pixel_values)
            mask = outputs.mask  # (B, N)
            
            k = min(k_samples, pixel_values.size(0))
            for i in range(k):
                # Denormalize image
                img = pixel_values[i].detach().cpu()
                if hasattr(self, 'dataset_mean') and hasattr(self, 'dataset_std'):
                    mean = self.dataset_mean.view(1, 3, 1, 1)
                    std = self.dataset_std.view(1, 3, 1, 1)
                    img = img * std + mean
                
                # Handle tensor dimensions - ensure we have (C, H, W) format
                if img.dim() == 4:
                    img = img.squeeze(0)  # Remove batch dimension if present
                
                img_np = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                
                # Create mask visualization
                mask_i = mask[i].detach().cpu().numpy()  # (N,)
                h = w = int(np.sqrt(len(mask_i)))
                mask_2d = mask_i.reshape(h, w)
                
                # Create colored mask overlay
                mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
                mask_colored[mask_2d == 1] = [255, 0, 0]  # Red for masked
                mask_colored[mask_2d == 0] = [0, 255, 0]  # Green for unmasked
                
                # Resize mask to image size
                mask_resized = Image.fromarray(mask_colored).resize((self.image_size, self.image_size))
                mask_resized = np.array(mask_resized)
                
                # Create overlay
                overlay = cv2.addWeighted(img_np, 0.7, mask_resized, 0.3, 0)
                
                # Save mask visualization
                Image.fromarray(overlay).save(
                    os.path.join(self.run_dir, "recon", f"mask_ep{epoch_idx:03d}_sample{i}.png")
                )
                
                # Log to wandb
                if hasattr(self, 'custom_args') and self.custom_args.use_wandb:
                    import wandb
                    wandb.log({
                        f"mask/epoch_{epoch_idx:03d}_sample_{i}": wandb.Image(overlay)
                    })
    

def get_args():
    p = argparse.ArgumentParser(description="Inspect Galaxy10 + ViTMAE (random init) and export a few samples to PDF.")
    p.add_argument("--dataset", default="matthieulel/galaxy10_decals", help="HF dataset id")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--out_dir", default="outputs", help="where to save PDFs and logs")
    p.add_argument("--num_samples_pdf", type=int, default=2, help="how many train images to save as PDF")
    p.add_argument("--seed", type=int, default=42)
    
    # Wandb arguments
    p.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--run_name", type=str, default=None, help="Custom run name for wandb")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")  # Match notebook
    p.add_argument("--batch_size", type=int, default=512, help="Batch size")  # Match notebook
    p.add_argument("--epochs", type=int, default=800, help="Number of epochs")  # Match notebook
    p.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio for MAE")
    p.add_argument("--hidden_size", type=int, default=384, help="Hidden size")
    p.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    p.add_argument("--decoder_hidden_size", type=int, default=256, help="Decoder hidden size")  # Match notebook
    p.add_argument("--decoder_num_layers", type=int, default=4, help="Number of decoder layers")  # Match notebook
    p.add_argument("--norm_pix_loss", action="store_true", help="Enable normalized pixel loss")  # Default is False (matches notebook)
    
    return p.parse_args()

def train_with_trainer(args, mae, train_dataset, dataset_mean, dataset_std):
    """Train MAE using HuggingFace Trainer."""
    
    # Create run directory
    tag = args.run_name if (args.use_wandb and args.run_name) else \
          f"mae_lr{args.lr}_bs{args.batch_size}_mask{args.mask_ratio}_hs{args.hidden_size}_L{args.num_layers}_decH{args.decoder_hidden_size}_decL{args.decoder_num_layers}"
    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "recon"), exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,  # Effective batch size = 256 * 2 = 512
        learning_rate=args.lr,
        weight_decay=0.05,
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=1000,
        report_to="wandb" if args.use_wandb else None,
        run_name=args.run_name if args.use_wandb else None,
        bf16=True,  # Enables bfloat16 training AND evaluation in modern versions
        bf16_full_eval=True,  # Use bf16 for evaluation as well
        fp16_full_eval=False,  # Disable fp16 evaluation since we're using bf16
        lr_scheduler_type="cosine",  # Use cosine learning rate schedule
        warmup_ratio=0.1,  # 10% of training for warmup
        dataloader_num_workers=4,
        remove_unused_columns=False,
        adam_beta1=0.9,  # Match notebook's settings
        adam_beta2=0.95,  # Match notebook's settings
    )
    
    trainer = MAETrainer(
        model=mae,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Use same dataset for eval
        image_size=args.image_size,
        run_dir=run_dir,
    )
    
    # Store normalization parameters for visualization
    trainer.dataset_mean = dataset_mean
    trainer.dataset_std = dataset_std
    trainer.custom_args = args    
    # Train
    print(f"[train] Starting MAE training with Trainer for {args.epochs} epochs")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    
    return trainer, run_dir


def main():
    args = get_args()
        # Initialize wandb if requested
    if args.use_wandb:
        if args.run_name is None:
            args.run_name = (
                f"mae_lr{args.lr}_bs{args.batch_size}_mask{args.mask_ratio}"
                f"_hs{args.hidden_size}_L{args.num_layers}_decH{args.decoder_hidden_size}"
                f"_decL{args.decoder_num_layers}"
            )
        wandb.init(
            project="ssl",
            entity="sizchode-brown-university",
            name=args.run_name,
            config={
                "dataset": args.dataset,
                "image_size": args.image_size,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "mask_ratio": args.mask_ratio,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "decoder_hidden_size": args.decoder_hidden_size,
                "decoder_num_layers": args.decoder_num_layers,
                "seed": args.seed,
            },
            settings=wandb.Settings(start_method="fork")
        )
        # one metric only
        wandb.define_metric("epoch")
        wandb.define_metric("mae_loss", summary="min")
    
    # Set seed directly
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Create directories directly
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)

    # 1) Load dataset
    print("[info] Loading dataset:", args.dataset)
    ds = load_dataset(args.dataset)
    train_split, test_split = ds["train"], ds["test"]
    print(f"[info] Splits -> train: {len(train_split)}  |  test: {len(test_split)}")

    # 2) Use pre-trained processor statistics (same as MAEVIT notebook)
    from transformers import ViTImageProcessor
    processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
    dataset_mean = processor.image_mean
    dataset_std = processor.image_std
    print(f"[info] Using pre-trained processor statistics:")
    print(f"  Mean: {dataset_mean}")
    print(f"  Std: {dataset_std}")
    
    # 3) Save a couple training samples as PDFs
    for i in range(min(args.num_samples_pdf, len(train_split))):
        item = train_split[i]
        img = item["image"]
        if not hasattr(img, "mode"):
            img = Image.fromarray(np.array(img))
        # Resize to target size
        img = img.resize((args.image_size, args.image_size))
        out_pdf = os.path.join(args.out_dir, "samples", f"train_sample_{i}.pdf")
        # Save as PDF directly
        img.convert("RGB").save(out_pdf, "PDF")
        print(f"[save] Wrote {out_pdf}")

    # 3) Build random-initialized ViTMAE (encoder–decoder)
    mae = build_vitmae(
        image_size=args.image_size,
        mask_ratio=args.mask_ratio,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        decoder_hidden_size=args.decoder_hidden_size,
        decoder_num_hidden_layers=args.decoder_num_layers,
        norm_pix_loss=args.norm_pix_loss
    )
    
    # Use HuggingFace Trainer for training
    print("[info] Using HuggingFace Trainer for training")
    train_dataset = MAEDataset(train_split, args.image_size, dataset_mean, dataset_std)
    trainer, run_dir = train_with_trainer(args, mae, train_dataset, dataset_mean, dataset_std)
    
    # Save encoder for linear probe
    torch.save(mae.vit.state_dict(), os.path.join(run_dir, "encoder.pth"))
    torch.save(mae.state_dict(), os.path.join(run_dir, "mae_full.pth"))

    # sanity: ensure we didn't accidentally load a pretrained checkpoint
    print("name_or_path in config (should be empty):", getattr(mae.config, "_name_or_path", ""))

    # 4) Print basic model info
    total_params = sum(p.numel() for p in mae.parameters())
    print(f"\nModel: {mae.__class__.__name__}")
    print(f"Total params: {total_params/1e6:.2f}M")
    
    # Print nn.Linear modules info
    linear_modules = []
    for name, module in mae.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append((name, module))
    
    print(f"\nFound {len(linear_modules)} nn.Linear modules:")
    for name, module in linear_modules:
        in_features = module.in_features
        out_features = module.out_features
        bias_params = out_features if module.bias is not None else 0
        total_linear_params = in_features * out_features + bias_params
        print(f"  {name}: {in_features} -> {out_features} (params: {total_linear_params:,})")
    
    # Log model info to wandb
    if args.use_wandb:
        wandb.log({
            "model/total_params_millions": total_params/1e6,
            "model/num_linear_layers": len(linear_modules),
            "model/class_name": mae.__class__.__name__,
        })
        
        # Log individual linear layer info
        # linear_info = {}
        # for i, (name, module) in enumerate(linear_modules):
        #     linear_info[f"linear_{i}_name"] = name
        #     linear_info[f"linear_{i}_in_features"] = module.in_features
        #     linear_info[f"linear_{i}_out_features"] = module.out_features
        #     linear_info[f"linear_{i}_params"] = (
        #         module.in_features * module.out_features
        #         + (module.out_features if module.bias is not None else 0)
        #     )
        
        # wandb.log(linear_info)
    # Training is now handled by the chosen method above
    

    # ===================== Linear Probe (defaults baked-in) =====================
    # Custom settings: AdamW with lr=1e-3 and batch_size=2048
    PROBE_EPOCHS = 90  # Keep same epochs
    PROBE_LR = 1e-3    # Custom learning rate
    PROBE_WD = 1e-4    # Small weight decay for AdamW
    NUM_WORKERS = 4
    # Infer classes from HF metadata, else fallback to 10 (Galaxy10)
    try:
        NUM_CLASSES = len(getattr(ds["train"].features["label"], "names", [])) or 10
    except Exception:
        NUM_CLASSES = 10

    # Run probe by default
    print("\n[probe] starting linear probe with defaults "
          f"(epochs={PROBE_EPOCHS}, lr={PROBE_LR}, wd={PROBE_WD}, classes={NUM_CLASSES})")

    # Use the encoder we just saved in this run
    enc_ckpt = os.path.join(run_dir, "encoder.pth")
    if not os.path.isfile(enc_ckpt):
        raise FileNotFoundError(f"[probe] encoder checkpoint not found: {enc_ckpt}")

    # Supervised loaders
    train_sup = HFWithLabel(train_split, image_size=args.image_size, train=True, mean=dataset_mean, std=dataset_std)
    test_sup  = HFWithLabel(test_split,  image_size=args.image_size, train=False, mean=dataset_mean, std=dataset_std)

    train_loader_sup = DataLoader(
        train_sup, batch_size=2048, shuffle=True,  # Custom batch size
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
    test_loader_sup = DataLoader(
        test_sup, batch_size=2048, shuffle=False,  # Custom batch size
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    # Rebuild encoder-only with matching dims (use same config as pretrain)
    enc_cfg = ViTMAEConfig(
        image_size=args.image_size, patch_size=16, num_channels=3,
        hidden_size=args.hidden_size, num_hidden_layers=args.num_layers,
        num_attention_heads=6,  # Default from build_vitmae function
        intermediate_size=args.hidden_size * 4, qkv_bias=True,
    )
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warning] CUDA not available, using CPU (will be slow)")
    
    encoder = ViTMAEModel(enc_cfg).to(device)
    sd = torch.load(enc_ckpt, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[probe] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    # Linear head (encoder is frozen)
    head = nn.Linear(args.hidden_size, NUM_CLASSES).to(device)
    ce = nn.CrossEntropyLoss()
    opt_h = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    
    # Learning rate scheduler to match notebook (cosine with warmup)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(opt_h, T_max=PROBE_EPOCHS, eta_min=0)

    def _cls_readout(last_hidden):  # (B, N, D) -> (B, D)
        return last_hidden.mean(dim=1)  # Mean pooling over all tokens (MAE has no CLS token)

    best_acc = 0.0
    for ep in range(1, PROBE_EPOCHS + 1):
        # ---- train head ----
        head.train()
        running_loss, seen = 0.0, 0
        for x, y in train_loader_sup:
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
        
        # Step the scheduler
        scheduler.step()

        # ---- eval ----
        head.eval()
        correct, count = 0, 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for x, y in test_loader_sup:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                feats = _cls_readout(encoder(pixel_values=x).last_hidden_state).float()
                logits = head(feats)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                count += y.numel()
        acc = correct / max(1, count)
        best_acc = max(best_acc, acc)

        print(f"[probe] epoch {ep:03d}/{PROBE_EPOCHS} | train_loss={train_loss:.4f} | "
              f"test_acc={acc*100:.2f}% (best {best_acc*100:.2f}%)")
        if args.use_wandb:
            wandb.log({
                "probe/epoch": ep,
                "probe/train_loss": train_loss,
                "probe/test_acc": acc,
                "probe/best_acc": best_acc
            }, step=ep)

    # save the head for reproducibility
    torch.save(head.state_dict(), os.path.join(run_dir, "linear_head.pth"))
    print(f"[probe] done | best test acc: {best_acc*100:.2f}%")
    # =================== end Linear Probe ===================
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
