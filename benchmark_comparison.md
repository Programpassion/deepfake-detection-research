# Benchmark Comparison: Coordinate Attention vs. Base EfficientNet-B3 Under Social Media Compression

This benchmark provides the empirical comparative analysis between:
1. **`M3-B16` (Control Baseline):** ImageNet-pretrained EfficientNet-B3 with compression-aware training, without attention.
2. **`M6-B16` (Proposed Architecture):** ImageNet-pretrained EfficientNet-B3 enhanced with **Coordinate Attention** (Hou et al., CVPR 2021) and compression-aware training.

Both architectures use identical training protocols ($B=16$, $\text{LR}=10^{-4}$, Adam, Seed 42, 10 epochs with Stage 1 frozen backbone and Stage 2 full network fine-tuning; stochastic JPEG training across Clean, QF80, QF60, QF50, QF40, QF20) and are evaluated across 12,000 frozen test inferences (2,000 images per condition).

---

## 1. Master Comparative Leaderboard

### A. Aggregate Compressed Performance (QF80 to QF20 — 10,000 Test Inferences)

| Metric | M3-B16 (Base Model) | M6-B16 (Proposed CoordAtt) | Absolute Improvement ($\Delta$) | Relative Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 96.46% | **96.97%** | **+0.51 pp** | — |
| **Precision** | 97.40% | **98.02%** | **+0.62 pp** | — |
| **Recall** | 95.48% | **95.88%** | **+0.40 pp** | — |
| **F1 Score** | 96.43% | **96.94%** | **+0.51 pp** | — |
| **MCC** | 0.9295 | **0.9397** | **+0.0102** | — |
| **False Positives (FP)** | 128 | **97** | **-31 FP** | **24.2% fewer false alarms** |
| **False Negatives (FN)** | 226 | **206** | **-20 FN** | **8.8% fewer missed fakes** |
| **Total Errors** | 354 | **303** | **-51 Errors** | **14.4% net error reduction** |

---

## 2. Condition-by-Condition Analysis (2,000 images per evaluation condition)

| Condition | JPEG QF | M3-B16 Acc | M6-B16 Acc | $\Delta$ Acc | M3 Precision | M6 Precision | $\Delta$ Prec | M3 Recall | M6 Recall | $\Delta$ Rec | M3 FP | M6 FP | $\Delta$ FP | M3 FN | M6 FN | $\Delta$ FN | M3 Errors | M6 Errors | Net Error $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** | NA | 97.40% | **97.80%** | **+0.40%** | 98.57% | **98.78%** | +0.21% | 96.20% | **96.80%** | +0.60% | 14 | **12** | **-2** | 38 | **32** | **-6** | 52 | **44** | **-8** |
| **QF80** | 80 | 97.20% | **97.65%** | **+0.45%** | 98.36% | **98.97%** | +0.61% | 96.00% | **96.30%** | +0.30% | 16 | **10** | **-6** | 40 | **37** | **-3** | 56 | **47** | **-9** |
| **QF60** | 60 | 97.40% | **97.55%** | **+0.15%** | 97.98% | **98.57%** | +0.59% | **96.80%** | 96.50% | -0.30% | 20 | **14** | **-6** | **32** | 35 | +3 | 52 | **49** | **-3** |
| **QF50** | 50 | 96.65% | **97.15%** | **+0.50%** | 96.42% | **97.48%** | +1.06% | **96.90%** | 96.80% | -0.10% | 36 | **25** | **-11** | **31** | 32 | +1 | 67 | **57** | **-10** |
| **QF40** | 40 | 96.25% | **96.65%** | **+0.40%** | 98.13% | **98.34%** | +0.21% | 94.30% | **94.90%** | +0.60% | 18 | **16** | **-2** | 57 | **51** | **-6** | 75 | **67** | **-8** |
| **QF20** | 20 | 94.80% | **95.85%** | **+1.05%** | 96.09% | **96.74%** | +0.65% | 93.40% | **94.90%** | +1.50% | 38 | **32** | **-6** | 66 | **51** | **-15** | 104 | **83** | **-21** |

---

## 3. Core Scientific Insights

### 1. Robustness Under Severe Compression (QF20)
At **JPEG Quality Factor 20** (mimicking harsh social media re-compression on platforms like WhatsApp and Facebook):
* Base EfficientNet-B3 suffers from severe degradation: accuracy drops to 94.80% with 104 errors (38 FP, 66 FN).
* Proposed Coord-EfficientNet-B3 maintains **95.85% accuracy** (+1.05 percentage points).
* M6 cuts QF20 errors from **104 down to 83 (-21 errors)**, reducing missed deepfakes (FN) from $66 \to 51$ (-15 FN).

### 2. Dual-Dimensional Coordinate Attention Immunity to Compression Noise
Standard channel attention collapses spatial information entirely, while 2D convolutions easily get distracted by 2D JPEG 8x8 block DCT grid ringing. Coordinate Attention:
* Factorizes channel attention into 1D horizontal and 1D vertical spatial encoding ($H \times 1$ and $1 \times W$).
* Operates along 1D slices, making it naturally low-pass filter against 2D checkerboard grid noise.
* Cuts False Positives from **128 down to 97 (-31 false alarms)** across compressed regimes.

### 3. Consistency Across Every Quality Factor
M6-B16 outperforms M3-B16 in accuracy and error count across **every single evaluation condition tested**:
* Clean: 52 $\to$ 44 errors (-8)
* QF80: 56 $\to$ 47 errors (-9)
* QF60: 52 $\to$ 49 errors (-3)
* QF50: 67 $\to$ 57 errors (-10)
* QF40: 75 $\to$ 67 errors (-8)
* QF20: 104 $\to$ 83 errors (-21)
