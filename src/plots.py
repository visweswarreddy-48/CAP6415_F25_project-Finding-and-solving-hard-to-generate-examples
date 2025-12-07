# src/plots.py
"""
Training loss and CLIP score visualization module.

This script loads training logs and CLIP scores from JSON files,
generates smoothed loss curves and CLIP score distributions,
and saves the plots for analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



# Configuration
TRAIN_LOG = Path("results/Metrics_json/train_logs.json")
CLIP_PATH = Path("results/Metrics_json/clip_scores.json")
PLOT_DIR = Path("results/plots")

PLOT_DIR.mkdir(parents=True, exist_ok=True)



# Utility Functions
def smooth_curve(values, window: int = 20):
    """Apply moving average smoothing."""
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")



# Plotting Functions
def plot_training_loss():
    with open(TRAIN_LOG) as f:
        log_data = json.load(f)

    steps = np.array(log_data["steps"])
    loss = np.array(log_data["train_loss"])

    loss_smooth = smooth_curve(loss, window=20)
    steps_smooth = steps[: len(loss_smooth)]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, loss, alpha=0.3, label="Raw Loss")
    plt.plot(steps_smooth, loss_smooth, linewidth=2, label="Smoothed Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("MSE Loss")
    plt.title("LoRA Training Loss (Raw vs Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / "training_loss.png")
    plt.close()


def plot_clip_scores():
    with open(CLIP_PATH) as f:
        clip_scores = np.array(json.load(f))

    plt.figure(figsize=(8, 5))
    plt.plot(clip_scores, marker="o", label="CLIP Score per Prompt")
    plt.axhline(clip_scores.mean(), linestyle="--", label="Average CLIP Score")
    plt.xlabel("Prompt Index")
    plt.ylabel("CLIP Score")
    plt.title("CLIP Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / "clip_scores.png")
    plt.close()

# Main Execution
def generate_all_plots():
    plot_training_loss()
    plot_clip_scores()
    print(f"Plots generated | Plots saved {PLOT_DIR}")

if __name__ == "__main__":
    generate_all_plots()
