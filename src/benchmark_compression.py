import os
import sys
import time
import random
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import EfficientNetB3Base
from model_cbam import EfficientNetB3CBAM
from model_triplet import EfficientNetB3Triplet
from model_coord import CoordEfficientNetB3

class CompressedDeepFakeDataset(Dataset):
    """
    Applies on-the-fly lossy JPEG compression (Quality Factor QF) to simulate social media platforms (WhatsApp, etc.)
    followed by standard ImageNet normalization.
    """
    def __init__(self, samples, quality=50, transform=None):
        self.samples = samples
        self.quality = quality
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        
        # Apply JPEG compression if quality < 100
        if self.quality < 100:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
            
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)

def get_holdout_samples(data_dir, total_per_class=5000, seed=42):
    train_dir = os.path.join(data_dir, "Train")
    real_dir = os.path.join(train_dir, "Real")
    fake_dir = os.path.join(train_dir, "Fake")

    valid_exts = ('.jpg', '.jpeg', '.png')
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(valid_exts)]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(valid_exts)]

    real_files.sort()
    fake_files.sort()

    rng = random.Random(seed)
    sampled_real = rng.sample(real_files, total_per_class)
    sampled_fake = rng.sample(fake_files, total_per_class)

    all_paths = sampled_real + sampled_fake
    all_labels = [0.0] * total_per_class + [1.0] * total_per_class

    _, test_paths, _, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=seed
    )
    test_samples = list(zip(test_paths, test_labels))
    return test_samples

def evaluate_model(model, loader, device):
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.squeeze().cpu().numpy().tolist())
            all_targets.extend(targets.squeeze().numpy().tolist())

    targets = np.array(all_targets)
    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(targets, preds) * 100.0
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    auc = roc_auc_score(targets, probs)
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc,
        'cm': cm, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'targets': targets, 'probs': probs, 'preds': preds
    }

def save_cm_plot(cm, title, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real (0)', 'Fake (1)'])
    plt.yticks(tick_marks, ['Real (0)', 'Fake (1)'])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    data_dir = "C:\\Users\\hdutt\\Desktop\\Deep Fake Detection Research"
    models_dir = os.path.join(data_dir, "models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print("\n--- Loading 2,000 Holdout Test Samples (Identical Seed 42 Split) ---")
    test_samples = get_holdout_samples(data_dir, total_per_class=5000, seed=42)
    print(f"Loaded {len(test_samples)} holdout test images.")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.244, 0.225])
    ])

    # Model definitions and paths
    models_info = [
        ("Baseline EfficientNet-B3", EfficientNetB3Base, os.path.join(models_dir, "efficientnet_b3_base_8020.pth"), "base"),
        ("EfficientNet-B3 + CBAM", EfficientNetB3CBAM, os.path.join(models_dir, "efficientnet_b3_cbam_8020.pth"), "cbam"),
        ("EfficientNet-B3 + Triplet", EfficientNetB3Triplet, os.path.join(models_dir, "efficientnet_b3_triplet_8020.pth"), "triplet"),
        ("Coord-EfficientNet-B3 (Proposed)", CoordEfficientNetB3, os.path.join(models_dir, "efficientnet_b3_coord_8020.pth"), "coord"),
    ]

    # Instantiate and load all 4 models
    loaded_models = []
    print("\n--- Loading Pretrained Checkpoints ---")
    for name, model_cls, path, tag in models_info:
        model = model_cls(pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        loaded_models.append((name, model, tag))
        print(f"  [OK] Loaded {name} from {os.path.basename(path)}")

    # 1. Primary Benchmark: JPEG QF=50 (WhatsApp / Social Media Standard)
    print("\n" + "="*85)
    print("      BENCHMARK 1: SOCIAL MEDIA COMPRESSION (JPEG QF=50 / WHATSAPP STANDARD)")
    print("="*85)
    
    qf50_loader = DataLoader(
        CompressedDeepFakeDataset(test_samples, quality=50, transform=test_transform),
        batch_size=32, shuffle=False, num_workers=0
    )

    qf50_results = {}
    for name, model, tag in loaded_models:
        t0 = time.time()
        res = evaluate_model(model, qf50_loader, device)
        elapsed = time.time() - t0
        qf50_results[tag] = res
        cm_path = os.path.join(models_dir, f"confusion_matrix_{tag}_qf50.png")
        save_cm_plot(res['cm'], f"Confusion Matrix: {name} (JPEG QF=50)", cm_path)
        print(f"Evaluated {name} in {elapsed:.1f}s -> Acc: {res['acc']:.2f}%, Prec: {res['prec']:.4f}, Rec: {res['rec']:.4f}, F1: {res['f1']:.4f}, AUC: {res['auc']:.4f}")

    # Print Comparative Table for QF=50
    print("\n" + "-"*105)
    print(f"{'Model Architecture':<35} | {'Acc (QF=50)':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'FP':<4} | {'FN':<4}")
    print("-"*105)
    for name, _, tag in loaded_models:
        r = qf50_results[tag]
        print(f"{name:<35} | {r['acc']:>10.2f}% | {r['prec']:>10.4f} | {r['rec']:>10.4f} | {r['f1']:>10.4f} | {r['auc']:>10.4f} | {r['fp']:>4} | {r['fn']:>4}")
    print("-"*105)

    # 2. Multi-Level Compression Degradation Benchmark (Clean, QF=75, QF=50, QF=30)
    print("\n" + "="*85)
    print("      BENCHMARK 2: MULTI-LEVEL COMPRESSION ROBUSTNESS ANALYSIS")
    print("      Quality Factors: Clean (100) -> Light (75) -> Standard/WhatsApp (50) -> Aggressive (30)")
    print("="*85)

    quality_levels = [100, 75, 50, 30]
    ql_labels = ["Clean (100)", "Light (75)", "WhatsApp (50)", "Heavy (30)"]
    degradation_history = {tag: [] for _, _, tag in loaded_models}

    for q in quality_levels:
        print(f"\n--> Evaluating all models at JPEG QF = {q}...")
        loader = DataLoader(
            CompressedDeepFakeDataset(test_samples, quality=q, transform=test_transform),
            batch_size=32, shuffle=False, num_workers=0
        )
        for name, model, tag in loaded_models:
            res = evaluate_model(model, loader, device)
            degradation_history[tag].append(res['acc'])
            print(f"    {name:<32}: {res['acc']:.2f}% Accuracy")

    # Generate Robustness Curve Plot
    plt.figure(figsize=(10, 6))
    colors = {'base': '#d9534f', 'cbam': '#f0ad4e', 'triplet': '#0275d8', 'coord': '#5cb85c'}
    markers = {'base': 'o', 'cbam': 's', 'triplet': '^', 'coord': 'D'}
    styles = {'base': '--', 'cbam': '-.', 'triplet': ':', 'coord': '-'}

    for name, _, tag in loaded_models:
        plt.plot(ql_labels, degradation_history[tag],
                 label=name, color=colors[tag], marker=markers[tag],
                 linestyle=styles[tag], linewidth=2.5, markersize=8)

    plt.title("DeepFake Detection Robustness Across Social Media Compression Levels", fontsize=13, pad=15)
    plt.xlabel("JPEG Quality Factor / Compression Level", fontsize=11)
    plt.ylabel("Holdout Classification Accuracy (%)", fontsize=11)
    plt.ylim(80, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='lower left')
    plt.tight_layout()

    robustness_plot_path = os.path.join(models_dir, "compression_robustness_curve.png")
    plt.savefig(robustness_plot_path, dpi=300)
    plt.close()
    print(f"\n[Saved] Robustness degradation curve saved to {robustness_plot_path}")

    # Generate Clean vs QF=50 Degradation Summary Table
    print("\n" + "="*95)
    print("      SUMMARY: CLEAN vs. WHATSAPP (QF=50) ACCURACY DEGRADATION")
    print("="*95)
    print(f"{'Model Architecture':<35} | {'Clean Acc':<12} | {'QF=50 Acc':<12} | {'Degradation (Drop)':<20} | {'Robustness Rank':<15}")
    print("-"*95)
    
    table_rows = []
    for name, _, tag in loaded_models:
        clean_acc = degradation_history[tag][0]
        qf50_acc = degradation_history[tag][2]
        drop = clean_acc - qf50_acc
        table_rows.append((name, clean_acc, qf50_acc, drop))

    # Sort by lowest degradation drop
    table_rows.sort(key=lambda x: x[3])
    for rank, (name, clean_acc, qf50_acc, drop) in enumerate(table_rows, 1):
        print(f"{name:<35} | {clean_acc:>10.2f}% | {qf50_acc:>10.2f}% | {f'-{drop:.2f}%':>18} | Rank {rank} {'(Most Robust!)' if rank == 1 else ''}")
    print("="*95)

if __name__ == "__main__":
    main()
