# CAP6415_F25_project – Finding and Solving Hard-to-Generate Examples

## Project Overview

Modern text-to-image diffusion models such as **Stable Diffusion** can generate high-quality images but still struggle with **rare or visually ambiguous objects**.  
One such object is the **speed bump**, which appears in many shapes, colors, and lighting conditions yet is **under-represented in large-scale datasets**.

This project investigates how to:

1. Identify **hard-to-generate objects**.
2. Evaluate baseline Stable Diffusion performance.
3. Improve generation quality using **LoRA fine-tuning**.
4. Quantitatively and qualitatively evaluate improvements.

The project uses **speed bumps as a case study**, applying targeted data augmentation and **LoRA adaptation of Stable Diffusion v1.5**.

---

## Objectives

- Test baseline Stable Diffusion performance on *speed bump* prompts  
- Collect and preprocess a real-world speed bump dataset  
- Generate captions using BLIP  
- Fine-tune Stable Diffusion using **LoRA (PEFT)**  
- Evaluate improvements using **CLIP similarity** and visual inspection  
- Visualize training curves and evaluation metrics  

---

## Framework and Tools

- **Language:** Python 3.10.11  
- **Base Model:** Stable Diffusion v1.5  
- **Fine-Tuning:** LoRA (via PEFT)  
- **Captioning:** BLIP  
- **Evaluation:** CLIP similarity  
- **Core Libraries:** PyTorch, Diffusers, Transformers, PEFT, OpenCV  
- **System Requirements:**
  - Windows 10 / 11  
  - NVIDIA GPU with ≥ 6 GB VRAM  
  - CUDA 12.1  
  - ≥ 16 GB RAM recommended  

---

## Repository Structure

```bash
CAP6415_F25_project-Finding-and-solving-hard-to-generate-examples/

├── data/
│   ├── raw/quality_images/            # Original high-quality images
│   └── processed/lora_ready/          # Processed images + captions
│
├── model/
│   └── lora_peft_checkpoint/          # Trained LoRA weights
│
├── notebooks/                         # Development notebooks
│   ├── baseline_generation.ipynb
│   ├── Data_preprocessing.ipynb
│   ├── image_captioning.ipynb
│   ├── LoRA_PEFT_train.ipynb
│   ├── test.ipynb
│   └── plots.ipynb
│
├── src/                               # Final production pipeline
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── captioning.py
│   ├── baseline_generation.py
│   ├── train.py
│   ├── inference.py
│   ├── plots.py
│   └── main.py                        # Full pipeline runner
│
├── results/
│   ├── before/                        # Baseline images
│   ├── After/                         # LoRA generated images
│   ├── Metrics_json/                  # Training loss & CLIP scores
│   └── plots/                         # Training & evaluation plots
│                         
├── .gitignore                         # To ignore few files
├── weekly_logs/                       # Weekly progress logs
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```
## Environment Setup and Installation

### Step 1 — Clone this repository

Open your terminal or PowerShell and run:
```bash
git clone https://github.com/visweswarreddy-48/CAP6415_F25_project-Finding-and-solving-hard-to-generate-examples.git
cd CAP6415_F25_project-Finding-and-solving-hard-to-generate-examples
```
### Step 2 — Create and activate a virtual environment

Windows (PowerShell or CMD):
```bash
python -m venv env_name
env_name\Scripts\activate
``` 
macOS / Linux:
```bash
python3 -m venv env_name
source env_name/bin/activate

```
### Step 3 Install dependencies (CUDA 12.1)
```bash
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
If you do not have a GPU, install CPU versions of PyTorch from
https://pytorch.org

### Full Pipeline Execution
Run the complete pipeline from the project root:
```bash
python -m src.main
```
This automatically performs:
-Image preprocessing
-Caption generation
-Baseline image generation
-LoRA fine-tuning
-LoRA inference + CLIP evaluation
-Training & evaluation plots

## Outputs
``` bash 
| Output Type                 | Directory                     |
| --------------------------- | ----------------------------- |
| Baseline Images             | `results/before/`             |
| LoRA Generated Images       | `results/After/`              |
| Training Logs & CLIP Scores | `results/Metrics_json/`       |
| Training & Evaluation Plots | `results/plots/`              |
| Trained LoRA Model          | `model/lora_peft_checkpoint/` |
```
## Model Details
**Base Model**: Stable Diffusion v1.5
**Adaptation Method**: LoRA (PEFT)
**Captioning Model**: BLIP
**Text Encoder**: CLIP
**Evaluation Metric**: CLIP Similarity Score

## Weekly Progress Logs
All development progress is recorded in:
```bash
weekly_logs/
```
## Final Notes
All production-ready code is located in src/
Jupyter notebooks were used only for development and experimentation
The pipeline is fully reproducible using src/main.py
CUDA-compatible setup verified for PyTorch 2.2.2 + cu121