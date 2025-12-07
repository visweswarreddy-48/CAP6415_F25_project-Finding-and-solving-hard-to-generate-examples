# src/preprocessing.py
"""
Image preprocessing module.

This module loads raw images, resizes them to a fixed resolution,
renames them in a clean sequential format, and saves the processed
images for LoRA training.
"""

import cv2
from pathlib import Path
from tqdm import tqdm

def preprocess_images(src_dir: Path, out_dir: Path, resolution: int = 512):
    """Resize and standardize raw images for training."""

    out_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(src_dir.glob("*.*"))

    processed, skipped = 0, 0

    for img_path in tqdm(image_files, desc="Preprocessing Images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"speedbump_{processed:05d}.jpg"

            if out_path.exists():
                skipped += 1
                continue

            cv2.imwrite(str(out_path), img)
            processed += 1

        except Exception:
            skipped += 1

    print(f"Preprocessing complete | Processed: {processed} | Skipped: {skipped} | saved to : {out_dir}")
    return processed, skipped