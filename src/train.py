# src/train.py
"""
LoRA fine-tuning script for Stable Diffusion v1.5.

This module Implements the LoRA fine-tuning pipeline for Stable Diffusion v1.5 by injecting low-rank adapters into the UNet, 
performing mixed-precision training with gradient accumulation, logging training loss, 
and saving the trained LoRA weights for later inference.
"""

import random
import json
from pathlib import Path
from dataclasses import dataclass
import warnings
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)


# Configuration


@dataclass
class TrainConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 1500
    mixed_precision: str = "fp16"
    rank: int = 4
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# Dataset
class LoraDataset(Dataset):

    def __init__(self, img_dir: Path, resolution: int):
        self.img_dir = img_dir
        self.image_files = [
            f for f in img_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]

        if not self.image_files:
            raise RuntimeError("No images found in dataset folder!")

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        print(f"Dataset loaded with {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        txt_path = img_path.with_suffix(".txt")

        caption = txt_path.read_text(encoding="utf-8").strip()
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "pixel_values": image,
            "caption": caption,
        }



# Training Pipeline
def train_lora(img_dir: Path, output_dir: Path, metrics_dir: Path, cfg: TrainConfig):

    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {img_dir}")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    # Load Base Model
    print("Loadinig Stable Diffusion base model")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16 if cfg.mixed_precision == "fp16" else torch.float32,
        safety_checker=None,
    ).to(device)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    scheduler = pipe.scheduler

    # Freeze Base Model
    log_msg_f = "Freezing VAE and Text Encoder--"
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    # Inject LoRA
    log_msg_U="Injecting LoRA into UNet--"
    lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank * 4,
        lora_dropout=0.1,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    
    unet = get_peft_model(unet, lora_config).to(device)
    unet.train()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    log_msg_P=f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}"
    print(f"{log_msg_f} | {log_msg_U} | {log_msg_P}")

    # DataLoader
    dataset = LoraDataset(img_dir, cfg.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Optimizer & AMP
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    use_fp16 = cfg.mixed_precision == "fp16" and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)


    # Training Loop (Progress Bar)
    train_losses = []
    step_history = []
    global_step = 0

    progress_bar = tqdm(total=cfg.max_train_steps, desc="Training LoRA")

    while global_step < cfg.max_train_steps:
        for batch in dataloader:

            if global_step >= cfg.max_train_steps:
                break

            with torch.cuda.amp.autocast(enabled=use_fp16):

                captions = batch["caption"]
                encoding = tokenizer(
                    list(captions),
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    text_embeddings = text_encoder(encoding.input_ids)[0]

                pixels = batch["pixel_values"].to(device)

                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample()
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                t = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                )

                noisy_latents = scheduler.add_noise(latents, noise, t)

                preds = unet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

                loss = nn.functional.mse_loss(preds, noise)
                loss = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            true_loss = loss.item() * cfg.gradient_accumulation_steps
            train_losses.append(true_loss)
            step_history.append(global_step)

            progress_bar.set_postfix({"loss": f"{true_loss:.5f}"})
            progress_bar.update(1)
            global_step += 1
    progress_bar.close()

    # Save Metrics
    with open(metrics_dir / "train_logs.json", "w") as f:
        json.dump({"steps": step_history, "train_loss": train_losses}, f)

    # Save LoRA Weights
    unet.save_pretrained(output_dir)
    print(f"Training logs saved to : {metrics_dir} | LoRA weights saved to : {output_dir}")


# Standalone Execution
if __name__ == "__main__":
    cfg = TrainConfig()

    train_lora(
        img_dir=Path("data/processed/lora_ready"),
        output_dir=Path("model/lora_peft_checkpoint"),
        metrics_dir=Path("results/Metrics_json"),
        cfg=cfg,
    )