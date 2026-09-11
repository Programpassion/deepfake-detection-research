# An Attention-Enhanced EfficientNet-B3 Framework for Robust DeepFake Detection on Compressed Social Media Images

**B.Tech Capstone Project (7th/8th Semester)**  
**Department of Computer Science & Engineering – AI & AIML**  

---

## 📌 Project Overview
The rapid rise of generative artificial intelligence (GANs, Diffusion Models) has led to hyper-realistic synthetic media (DeepFakes), threatening information integrity, digital privacy, and online security. While deep architectures like **EfficientNet-B3** achieve high accuracy on clean benchmark datasets, they experience significant degradation when images are shared across social media platforms (WhatsApp, Instagram, Facebook) due to aggressive lossy compression (JPEG, downsampling).

This project investigates and benchmarks attention mechanisms (**CBAM**, **Triplet Attention**, and **Coordinate Attention**) integrated with **EfficientNet-B3** to enhance spatial and channel-wise feature extraction from compressed facial images.

---

## 🔬 Base Research Paper Reference
This project replicates, validates, and extends:
> **B. G. Deepa, C. K. Lokesh, D. Umamaheswari, B. Ayshwarya, P. V. Yethish, and K. P. Suhaas**,  
> *"An enhanced deep learning framework for DeepFake detection using EfficientNet-B3 comparative evaluation of deep and machine learning techniques,"*  
> *Discover Computing*, vol. 29, no. 18, 2026.  
> DOI: [10.1007/s10791-025-09890-x](https://doi.org/10.1007/s10791-025-09890-x)

---

## 📊 Current Experimental Benchmark (Phase 1 & Phase 2 - Batch Size = 8)

Evaluated on an identical holdout test set of **2,000 balanced images (1,000 Real, 1,000 Fake)** under an 80:20 stratified split with exact base paper batch size ($B=8$):

| Model Configuration | Attention Mechanism | Trainable Parameters | Overall Accuracy | Precision (Fake) | Recall (Fake) | F1-Score | ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Paper Reported** *(Deepa et al.)* | *None* | ~10.91 M | **97.95%** | 0.9800 | 0.9800 | 0.9800 | — | Published |
| **Model 1: Baseline Replicated ($B=8$)** | *None* | 10.91 M | **98.10%** | **0.9859** | **0.9760** | **0.9809** | 0.9944 | **Completed** |
| **Model 2: EfficientNet-B3 + CBAM ($B=8$)** | Channel + Spatial ($7\times7$) | 11.20 M (+2.7%) | **97.90%** | **0.9858** | 0.9720 | 0.9789 | **0.9960** | **Completed** |
| **Model 3: EfficientNet-B3 + Triplet Attention** | Cross-Dimension Rotation | ~10.91 M (+0.002%) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Upcoming |
| **Model 4: Proposed Coord-EfficientNet-B3** | 1D Positional ($X, Y$) | ~10.93 M (+0.18%) | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Upcoming |

---

## 📁 Repository Structure

```text
├── src/
│   ├── model.py           # Baseline EfficientNet-B3 + 3-stage custom MLP head
│   ├── model_cbam.py      # EfficientNet-B3 + CBAM attention module
│   ├── dataset.py         # PyTorch Dataset, augmentations, and stratified loaders
│   ├── train.py           # General training engine (supports physical/custom splits)
│   ├── train_80_20.py     # Base paper replication training script (80:20 split, B=8)
│   ├── train_cbam.py      # CBAM training script (80:20 split, B=8)
│   └── evaluate.py        # Evaluation, confusion matrix, and metrics generator
├── models/                # Saved checkpoints (.pth) and visual plots (.png)
├── benchmark_comparison.md# Complete descriptive comparison log
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 🚀 Quickstart & Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Programpassion/deepfake-detection-research.git
cd "deepfake-detection-research"
```

### 2. Create and Activate Virtual Environment
Using Python 3.10 or 3.11:
```bash
# Using standard venv:
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

# Or using uv (recommended for ultra-fast setup):
uv venv .venv --python 3.11
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA support (for NVIDIA GPUs):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install required packages:
pip install -r requirements.txt
```

### 4. Dataset Setup
Download the **"DeepFake and Real Images"** dataset from Kaggle:
[Kaggle Dataset Link](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

Extract the folders into the project root:
```text
Deep Fake Detection Research/
├── Train/
│   ├── Real/
│   └── Fake/
├── Validation/
└── Test/
```

### 5. Running Training & Evaluation
To train the baseline model ($B=8$):
```bash
python src/train_80_20.py
```
To train the CBAM model ($B=8$):
```bash
python src/train_cbam.py
```
To evaluate any model checkpoint:
```bash
python src/evaluate.py --model_path models/efficientnet_b3_base_8020.pth
```
