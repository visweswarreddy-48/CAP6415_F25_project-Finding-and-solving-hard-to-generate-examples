# src/main.py
"""
Main execution script for the complete LoRA training pipeline.

This script runs:
1. Image preprocessing
2. Caption generation
3. Baseline image generation
4. LoRA fine-tuning
5. LoRA inference + CLIP evaluation
6. Training & evaluation plots
"""

from pathlib import Path

from .preprocessing import preprocess_images
from .captioning import generate_captions_for_directory
from .baseline_generation import generate_baseline_image
from .train import train_lora, TrainConfig
from .inference import run_inference
from .plots import generate_all_plots


def main():

    print("\n---------- PIPELINE STARTED ----------\n")


    # Step 1: Image Preprocessing

    print("STEP 1: Preprocessing Images---")
    preprocess_images(
        src_dir=Path("data/raw/quality_images"),
        out_dir=Path("data/processed/lora_ready"),
        resolution=512,
    )


    # Step 2: Image Captioning

    print("\nSTEP 2: Generating Captions---")
    generate_captions_for_directory(
        image_dir=Path("data/processed/lora_ready")
    )


    # Step 3: Baseline Image Generation

    print("\nSTEP 3: Generating Baseline Image---")
    baseline_prompt = (
        "a speed bump on a suburban asphalt road, daylight, realistic photo"
    )

    generate_baseline_image(
        prompt=baseline_prompt,
        output_dir=Path("results/before"),
    )


    # Step 4: LoRA Training

    print("\nSTEP 4: Training LoRA Model---")
    cfg = TrainConfig()

    train_lora(
        img_dir=Path("data/processed/lora_ready"),
        output_dir=Path("model/lora_peft_checkpoint"),
        metrics_dir=Path("results/Metrics_json"),
        cfg=cfg,
    )


    # Step 5: LoRA Inference + CLIP Evaluation

    print("\nSTEP 5: Running Inference + CLIP Evaluation---")
    run_inference()


    # Step 6: Plot Generation

    print("\nSTEP 6: Generating Training & Evaluation Plots---")
    generate_all_plots()

    print("\n---------- PIPELINE COMPLETED SUCCESSFULLY ----------\n")


if __name__ == "__main__":
    main()