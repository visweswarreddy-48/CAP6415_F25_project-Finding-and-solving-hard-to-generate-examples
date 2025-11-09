# CAP6415_F25_project-Finding-and-solving-hard-to-generate-examples

---

## 🧠 Project Overview

Modern text-to-image diffusion models such as **Stable Diffusion** can generate stunning visuals but still struggle with certain rare or visually ambiguous objects.  
One such object is the **speed bump**, which appears in many shapes, colours, and lighting conditions yet is under-represented in most large-scale datasets.

This project explores how to:
1. Identify such **hard-to-generate objects**.
2. Evaluate how poorly they are represented by pretrained diffusion models.
3. Improve the model’s performance through **fine-tuning** (e.g. LoRA or Textual Inversion).
4. Quantitatively and qualitatively evaluate improvements.

The project focuses on **“speed bump”** as a case study and aims to generate more accurate and realistic images through targeted data augmentation and model adaptation.

---

## 🎯 Objectives
- ✅ Test baseline Stable Diffusion performance on “speed bump” prompts.
- ✅ Collect a small dataset of real-world speed-bump images.
- ✅ Apply lightweight fine-tuning using LoRA or Textual Inversion.
- ✅ Evaluate improvement with CLIP similarity, FID, and visual inspection.

---

## 🧩 Framework and Tools
- **Language:** Python 3.10  
- **Core Libraries:** PyTorch, Hugging Face Diffusers, Transformers  
- **Fine-tuning:** LoRA / Textual Inversion (planned)  
- **Evaluation:** CLIP similarity, FID  
- **System Requirements:**  
  - Windows 10 / 11 or Linux  
  - GPU with ≥6 GB VRAM (recommended: NVIDIA RTX 3060 or better)  
  - CUDA 12.1 or higher  

---

## 📂 Repository Structure

CAP6415_F25_project-Speed-Bump-Generation/
│
├── README.md # Project overview and setup guide
├── requirements.txt # Required dependencies
├── weekly_logs
│ └── week1log.txt # Week 1 activity log
├── notebooks/
│ └── baseline_generation.ipynb # Baseline Stable Diffusion test
│
├── src/ # Source scripts (to be added)
│
└── results/
└── baseline_speedbump.png


---

## ⚙️ Environment Setup and Installation

### Step 1️⃣ — Clone this repository
Open your terminal or PowerShell and run:
```bash
git clone https://github.com/YOUR_USERNAME/CAP6415_F25_project-Speed-Bump-Generation.git
cd CAP6415_F25_project-Speed-Bump-Generation

```
### Step 2️⃣ — Create and activate a virtual environment

Windows (PowerShell or CMD):
```bash
python -m venv deep_learning_env
deep_learning_env\Scripts\activate
``` 
macOS / Linux:
```bash
python3 -m venv deep_learning_env
source deep_learning_env/bin/activate

```
### Step 3️⃣ Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
