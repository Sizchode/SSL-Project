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

class MAEDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, image_size: int = 256, mean=None, std=None, 
                 use_random_crop: bool = False, crop_scale: Tuple[float, float] = (0.2, 1.0)):
        self.split = hf_split
        transform_list = []
        
        if use_random_crop:
            transform_list.append(transforms.RandomResizedCrop(image_size, scale=crop_scale))
            transform_list.append(transforms.RandomHorizontalFlip())
        else:
            transform_list.append(transforms.Resize((image_size, image_size)))
        
        transform_list.append(transforms.ToTensor())
        
        if mean is not None and std is not None:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        self.tx = transforms.Compose(transform_list)
    
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

class MAETrainer(Trainer):
    def __init__(self, *args, image_mean=None, image_std=None, **kwargs):
        self.image_size = kwargs.pop('image_size', 256)
        self.run_dir = kwargs.pop('run_dir', 'outputs')
        if image_mean is not None and image_std is not None:
            self.image_mean = torch.tensor(image_mean).view(1, 1, 3)
            self.image_std = torch.tensor(image_std).view(1, 1, 3)
        else:
            self.image_mean = None
            self.image_std = None
        super().__init__(*args, **kwargs)
    
    def show_image(self, image, title="", image_mean=0, image_std=1, ax=None):
        assert image.shape[2] == 3
        if image_mean is None:
            image_mean = 0
        if image_std is None:
            image_std = 1
        ax.imshow(torch.clip((image * image_std + image_mean) * 255, 0, 255).int())
        ax.set_title(title, fontsize=16)
        ax.axis("off")
        return

    def visualize(self, pixel_values: torch.Tensor, model):
        model.eval()
        outputs = model(pixel_values.to(next(model.parameters()).device))
        y = model.unpatchify(outputs.logits)
        y = torch.einsum('nchw->nhwc', y).cpu()
        mask = outputs.mask.detach()
        p = model.config.patch_size
        mask = mask.unsqueeze(-1).repeat(1, 1, p*p*3)
        mask = model.unpatchify(mask)
        mask = torch.einsum('nchw->nhwc', mask).cpu()
        x = torch.einsum('nchw->nhwc', pixel_values).cpu()
        im_masked = x * (1 - mask)
        im_paste = x * (1 - mask) + y * mask
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        self.show_image(x[0], "original image", image_mean=self.image_mean, image_std=self.image_std, ax=axs[0])
        self.show_image(im_masked[0], "masked image", ax=axs[1])
        self.show_image(im_paste[0], "reconstruction image", image_mean=self.image_mean, image_std=self.image_std, ax=axs[2])
        plt.tight_layout()
        return fig

    @torch.no_grad()
    def _log_eval_visuals(self, eval_dataset=None, image_index: int = 1, batch_key: str = "pixel_values"):
        if not self.is_world_process_zero():
            return
        dl = self.get_eval_dataloader(eval_dataset)
        try:
            batch = next(iter(dl))
        except StopIteration:
            return
        if batch_key not in batch:
            print(f"[MyTrainer] batch key '{batch_key}' not found, skipping visualization.")
            return
        pixel_values = batch[batch_key][image_index].unsqueeze(0)
        self.model.eval()
        fig = self.visualize(pixel_values, self.model)
        if hasattr(self, 'state') and hasattr(self.state, 'global_step'):
            try:
                import wandb
                wandb.log({f"eval/mae_reconstruction_image{image_index}": wandb.Image(fig)}, step=int(self.state.global_step))
            except:
                pass
        if hasattr(self, 'state') and hasattr(self.state, 'global_step'):
            fname = f"mae_reconstruction_step{self.state.global_step}.png"
            if hasattr(self, 'run_dir'):
                save_path = os.path.join(self.run_dir, fname)
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset,
                                   ignore_keys=ignore_keys,
                                   metric_key_prefix=metric_key_prefix)
        try:
            for i in range(9):
                self._log_eval_visuals(eval_dataset=eval_dataset, image_index=i, batch_key="pixel_values")
        except Exception as e:
            print(f"[MyTrainer] fail{e}")
        return metrics

def get_args():
    p = argparse.ArgumentParser(description="Inspect Galaxy10 + ViTMAE (random init) and export a few samples to PDF.")
    p.add_argument("--dataset", default="matthieulel/galaxy10_decals", help="HF dataset id")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--out_dir", default="outputs", help="where to save PDFs and logs")
    p.add_argument("--num_samples_pdf", type=int, default=2, help="how many train images to save as PDF")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--run_name", type=str, default=None, help="Custom run name for wandb")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size")
    p.add_argument("--epochs", type=int, default=800, help="Number of epochs")
    p.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio for MAE")
    p.add_argument("--hidden_size", type=int, default=384, help="Hidden size")
    p.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    p.add_argument("--decoder_hidden_size", type=int, default=256, help="Decoder hidden size")
    p.add_argument("--decoder_num_layers", type=int, default=4, help="Number of decoder layers")
    p.add_argument("--norm_pix_loss", action="store_true", help="Enable normalized pixel loss")
    p.add_argument("--use_random_crop", action="store_true", help="Enable random cropping for data augmentation")
    p.add_argument("--crop_scale_min", type=float, default=0.2, help="Minimum crop scale ratio")
    p.add_argument("--crop_scale_max", type=float, default=1.0, help="Maximum crop scale ratio")
    return p.parse_args()

def train_with_trainer(args, mae, train_dataset, dataset_mean, dataset_std):
    tag = args.run_name if (args.use_wandb and args.run_name) else \
          f"mae_lr{args.lr}_bs{args.batch_size}_mask{args.mask_ratio}_hs{args.hidden_size}_L{args.num_layers}_decH{args.decoder_hidden_size}_decL{args.decoder_num_layers}"
    if args.use_random_crop and not (args.use_wandb and args.run_name):
        tag += f"_crop{args.crop_scale_min}-{args.crop_scale_max}"
    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "recon"), exist_ok=True)
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
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
        bf16=True,
        bf16_full_eval=True,
        fp16_full_eval=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
    )
    
    trainer = MAETrainer(
        model=mae,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        image_size=args.image_size,
        run_dir=run_dir,
    )
    trainer.dataset_mean = dataset_mean
    trainer.dataset_std = dataset_std
    trainer.custom_args = args
    print(f"[train] Starting MAE training with Trainer for {args.epochs} epochs")
    trainer.train()
    trainer.save_model()
    return trainer, run_dir

def main():
    args = get_args()
    if args.use_wandb:
        if args.run_name is None:
            args.run_name = (
                f"mae_lr{args.lr}_bs{args.batch_size}_mask{args.mask_ratio}"
                f"_hs{args.hidden_size}_L{args.num_layers}_decH{args.decoder_hidden_size}"
                f"_decL{args.decoder_num_layers}"
            )
            if args.use_random_crop:
                args.run_name += f"_crop{args.crop_scale_min}-{args.crop_scale_max}"
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
                "use_random_crop": args.use_random_crop,
                "crop_scale_min": args.crop_scale_min,
                "crop_scale_max": args.crop_scale_max,
            },
            settings=wandb.Settings(start_method="fork")
        )
        wandb.define_metric("epoch")
        wandb.define_metric("mae_loss", summary="min")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)

    print("[info] Loading dataset:", args.dataset)
    ds = load_dataset(args.dataset)
    train_split, test_split = ds["train"], ds["test"]
    print(f"[info] Splits -> train: {len(train_split)}  |  test: {len(test_split)}")

    from transformers import ViTImageProcessor
    processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
    dataset_mean = processor.image_mean
    dataset_std = processor.image_std
    print(f"[info] Using pre-trained processor statistics:")
    print(f"  Mean: {dataset_mean}")
    print(f"  Std: {dataset_std}")
    for i in range(min(args.num_samples_pdf, len(train_split))):
        item = train_split[i]
        img = item["image"]
        if not hasattr(img, "mode"):
            img = Image.fromarray(np.array(img))
        img = img.resize((args.image_size, args.image_size))
        out_pdf = os.path.join(args.out_dir, "samples", f"train_sample_{i}.pdf")
        img.convert("RGB").save(out_pdf, "PDF")
        print(f"[save] Wrote {out_pdf}")

    mae = build_vitmae(
        image_size=args.image_size,
        mask_ratio=args.mask_ratio,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        decoder_hidden_size=args.decoder_hidden_size,
        decoder_num_hidden_layers=args.decoder_num_layers,
        norm_pix_loss=args.norm_pix_loss
    )
    
    print("[info] Using HuggingFace Trainer for training")
    if args.use_random_crop:
        print(f"[info] Using random crop augmentation with scale range: [{args.crop_scale_min}, {args.crop_scale_max}]")
    train_dataset = MAEDataset(
        train_split, args.image_size, dataset_mean, dataset_std,
        use_random_crop=args.use_random_crop,
        crop_scale=(args.crop_scale_min, args.crop_scale_max)
    )
    trainer, run_dir = train_with_trainer(args, mae, train_dataset, dataset_mean, dataset_std)
    torch.save(mae.vit.state_dict(), os.path.join(run_dir, "encoder.pth"))
    torch.save(mae.state_dict(), os.path.join(run_dir, "mae_full.pth"))
    print("name_or_path in config (should be empty):", getattr(mae.config, "_name_or_path", ""))
    total_params = sum(p.numel() for p in mae.parameters())
    print(f"\nModel: {mae.__class__.__name__}")
    print(f"Total params: {total_params/1e6:.2f}M")
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
    if args.use_wandb:
        wandb.log({
            "model/total_params_millions": total_params/1e6,
            "model/num_linear_layers": len(linear_modules),
            "model/class_name": mae.__class__.__name__,
        })
        wandb.finish()

if __name__ == "__main__":
    main()
