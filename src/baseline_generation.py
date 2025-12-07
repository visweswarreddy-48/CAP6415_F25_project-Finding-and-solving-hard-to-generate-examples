# src/baseline_generation.py
"""
Baseline image generation using Stable Diffusion.

This module generates a baseline image using the pretrained
Stable Diffusion v1.5 model and saves the output in results/before/.
"""

from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline


def generate_baseline_image(prompt: str, output_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    
    pipe.set_progress_bar_config(desc="Generating baseline image")
    image = pipe(prompt).images[0]
    save_path = output_dir / "baseline_speedbump.png"
    image.save(save_path)

    print(f"Baseline image saved to: {save_path}")
    return save_path