# src/captioning.py
"""
Image caption generation module using BLIP with safe prompt variations.

This module generates realistic and diverse captions for each image
using a pretrained BLIP model and controlled prompt variations.
"""
# src/captioning.py

import random
import re
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


VIEWS = [
    "a close-up of", "a detailed view of", "a wide view of",
    "a clear view of", "a ground-level view of"
]

LOCATIONS = [
    "on an asphalt road surface", "on an asphalt street",
    "on a paved roadway", "in a parking area on asphalt",
    "on a residential asphalt road"
]


def load_blip(device):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model.eval()
    return processor, model


def extract_color_hint(text):
    text = text.lower()
    if "yellow" in text and "black" in text:
        return random.choice(["yellow and black", "black and yellow"])
    if "yellow" in text:
        return "yellow"
    return ""


def generate_safe_caption(raw_blip):
    view = random.choice(VIEWS)
    location = random.choice(LOCATIONS)
    color = extract_color_hint(raw_blip)

    if color:
        caption = f"{view} {color} speed bump {location}, realistic photo"
    else:
        caption = f"{view} a speed bump {location}, realistic photo"

    return re.sub(r"\s+", " ", caption).strip()


def generate_captions_for_directory(image_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_blip(device)

    image_paths = sorted(image_dir.glob("*.jpg"))
    count = 0

    for img_path in tqdm(image_paths, desc="Generating Captions"):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=40)

            raw_caption = processor.decode(output[0], skip_special_tokens=True)
            final_caption = generate_safe_caption(raw_caption)

            img_path.with_suffix(".txt").write_text(final_caption, encoding="utf-8")
            count += 1

        except Exception:
            continue

    print(f"Captioning complete | Total images: {count} | saved to : {image_dir}")
    return count