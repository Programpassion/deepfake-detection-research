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

## 📊 Current Experimental Benchmark: Clean vs. Social Media Compression (WhatsApp QF=50)

Evaluated on an identical holdout test set of **2,000 balanced images (1,000 Real, 1,000 Fake)** under an 80:20 stratified split with exact base paper batch size ($B=8$):

### 1. Clean / Uncompressed Evaluation
| Model Configuration | Attention Mechanism | Trainable Params | Accuracy (Clean) | Precision | Recall | F1-Score | ROC-AUC | False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Paper Reported** *(Deepa et al.)* | *None* | ~10.91 M | **97.95%** | 0.9800 | 0.9800 | 0.9800 | — | 21 / 1,009 |
| **Model 1: Baseline Replicated ($B=8$)** | *None* | 10.91 M | **98.10%** | 0.9859 | 0.9760 | **0.9809** | 0.9944 | 14 / 1,000 |
| **Model 2: Residual CBAM ($B=8$, Fixed)** | Channel + Spatial ($7\times7$) | 11.20 M (+2.7%) | **98.05%** | **0.9898** 🏆 | 0.9710 | 0.9803 | 0.9953 | **10 / 1,000** 🏆 |
| **Model 3: Residual Triplet ($B=8$, Fixed)** | Cross-Dimension Rotation | **10.91 M (+0.003%)** | **98.05%** | 0.9868 | 0.9740 | 0.9804 | **0.9968** 🏆 | 13 / 1,000 |
| **Model 4: Proposed Residual CoordAttn** | 1D Positional ($X, Y$) | **11.12 M (+1.96%)** | **97.95%** | 0.9762 | **0.9830** 🏆 | 0.9796 | **0.9966** | 24 / 1,000 |

### 2. Social Media Robustness Benchmark: WhatsApp Compression (JPEG QF=50)
| Model Configuration | Accuracy (QF=50) | Accuracy Drop ($\Delta$) | Precision (QF=50) | Recall (QF=50) | F1-Score | ROC-AUC | False Positives ($FP$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 3: Residual Triplet Attention** | **97.95%** 🏆 | **-0.10%** 🏆 | **0.9819** 🏆 | 0.9770 | **0.9794** 🏆 | **0.9959** 🏆 | **18 / 1,000** 🏆 |
| **Model 1: Baseline EfficientNet-B3** | 97.80% | -0.30% | 0.9790 | 0.9770 | 0.9780 | 0.9926 | 21 / 1,000 |
| **Model 2: Residual CBAM** | 96.95% | -1.10% | 0.9681 | 0.9710 | 0.9695 | 0.9939 | 32 / 1,000 |
| **Model 4: Proposed Residual CoordAttn** | 96.85% | -1.10% | 0.9553 | **0.9830** 🏆 | 0.9690 | 0.9957 | 46 / 1,000 |

---

## 📁 Repository Structure

```text
├── src/
│   ├── model.py                  # Baseline EfficientNet-B3 + 3-stage custom MLP head
│   ├── model_cbam.py             # EfficientNet-B3 + CBAM attention module
│   ├── model_triplet.py          # EfficientNet-B3 + Triplet Attention module (WACV 2021)
│   ├── model_coord.py            # Proposed Coord-EfficientNet-B3 (Hou et al., CVPR 2021)
│   ├── dataset.py                # PyTorch Dataset, augmentations, and stratified loaders
│   ├── train_80_20.py            # Base paper replication training script (80:20 split, B=8)
│   ├── train_cbam.py             # CBAM training script (80:20 split, B=8)
│   ├── train_triplet.py          # Triplet Attention training script (80:20 split, B=8)
│   ├── train_coord.py            # Coordinate Attention training script (80:20 split, B=8)
│   └── benchmark_compression.py  # Social media compression benchmark & multi-level degradation
├── models/                       # Checkpoints (.pth), confusion matrices, and robustness curve (.png)
├── benchmark_comparison.md       # Complete descriptive comparison log
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
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
