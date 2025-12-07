# src/inference.py
"""
Inference and CLIP evaluation for LoRA fine-tuned Stable Diffusion.

This script generates images using the trained LoRA adapter and
computes CLIP similarity scores between prompts and generated images.
"""

import json
from pathlib import Path
import warnings
import logging

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
import clip

warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)


# Configuration

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_DIR = Path("model/lora_peft_checkpoint")
OUTPUT_DIR = Path("results/After")
CLIP_OUT_DIR = Path("results/Metrics_json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIP_OUT_DIR.mkdir(parents=True, exist_ok=True)


PROMPTS = [
    "a close-up of a speed bump on an asphalt road surface, realistic photo",
    "a detailed view of a speed bump on an asphalt road, realistic photo",
    "a wide view of a speed bump on a residential asphalt road, realistic photo",
    "a ground-level view of a yellow and black speed bump on an asphalt street",
    "a clear view of a speed bump on a paved asphalt road, realistic photo",

    "a speed bump on a wet asphalt road after rain, realistic photo",
    "a speed bump on a dry asphalt road in bright daylight, realistic photo",
    "a speed bump on an asphalt road under cloudy lighting, realistic photo",
    "a speed bump on an asphalt road in soft evening light, realistic photo",
    "a speed bump on an asphalt road under street lighting at night, realistic photo",

    "a newly painted yellow and black speed bump on an asphalt road, realistic photo",
    "a slightly worn speed bump on an asphalt street, realistic photo",
    "a faded yellow speed bump on an asphalt road surface, realistic photo",

    "a speed bump near a pedestrian crossing on an asphalt road, realistic photo",
    "a speed bump in a residential asphalt street, realistic photo",
    "a speed bump in a parking area on asphalt, realistic photo",

    "a speed bump on a narrow asphalt road, realistic photo",
    "a speed bump on a straight asphalt road, realistic photo",
    "a speed bump on an empty asphalt road, realistic photo",
    "a speed bump on an urban asphalt street, realistic photo",
]

NEGATIVE_PROMPT = (
    "blurry, low resolution, bad anatomy, distorted, deformed, extra objects, "
    "cartoon, anime, painting, illustration, unrealistic, fake, CGI, pothole, "
    "flat road, smooth road surface, no speed bump"
)



# Model Loader

def load_lora_pipeline(device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet, LORA_DIR
    ).merge_and_unload()
    pipe.set_progress_bar_config(disable=True)
    return pipe



# Image Generation
def generate_images(pipe, device: str):
    generated_paths = []
    for idx, prompt in enumerate(tqdm(PROMPTS, desc="Generating Images")):
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]
        out_path = OUTPUT_DIR / f"lora_result_{idx:03d}.png"
        image.save(out_path)
        generated_paths.append(out_path)
    return generated_paths



# CLIP Evaluation

def compute_clip_scores(device: str):
    model, preprocess = clip.load("ViT-B/32", device=device)
    clip_scores = []
    for idx, prompt in enumerate(tqdm(PROMPTS, desc="Computing CLIP Scores")):
        img_path = OUTPUT_DIR / f"lora_result_{idx:03d}.png"
        image = preprocess(
            Image.open(img_path).convert("RGB")
        ).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
        clip_scores.append(similarity)
    return clip_scores


# Main Inference Pipeline
def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running inference on device:", device)
    pipe = load_lora_pipeline(device)
    generated_images = generate_images(pipe, device)
    clip_scores = compute_clip_scores(device)
    # Save CLIP scores
    with open(CLIP_OUT_DIR / "clip_scores.json", "w") as f:
        json.dump(clip_scores, f)
    avg_score = sum(clip_scores) / len(clip_scores)
    print(f"Inference completed | Images: {len(generated_images)} saved to : {OUTPUT_DIR} | Average CLIP Score: {avg_score:.4f} | CLIP Scores saved to : {CLIP_OUT_DIR}")

# Standalone Execution
if __name__ == "__main__":
    run_inference()