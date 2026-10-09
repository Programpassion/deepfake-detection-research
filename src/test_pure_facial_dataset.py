"""
test_pure_facial_dataset.py
Evaluate M3-B16 vs. M6-B16 strictly on PURE FACIAL DATA at canonical threshold tau = 0.50.
Uses 200 verified unseen face-crop samples (100 Real, 100 Fake) across all 6 quality factors.
"""

import os
import sys
import json
import csv
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import EfficientNetB3Baseline
from model_coordatt import EfficientNetB3CoordAtt
from dataset_loader import get_phase2_transforms

class PureFaceDataset(Dataset):
    def __init__(self, records, condition='clean', qf=None, transform=None):
        self.records = records
        self.condition = condition
        self.qf = qf
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r['path']).convert("RGB")

        if self.qf is not None:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=self.qf)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': torch.tensor([r['label']], dtype=torch.float32),
            'filename': r['filename'],
            'class': r['class']
        }

def evaluate_loader(model, loader, device):
    model.eval()
    targets, probs = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            tgts = batch['label'].to(device)
            with torch.amp.autocast('cuda'):
                p = torch.sigmoid(model(imgs))
            targets.extend(tgts.cpu().numpy().flatten())
            probs.extend(p.cpu().numpy().flatten())

    targets = np.array(targets)
    probs = np.array(probs)
    preds = (probs >= 0.50).astype(float)

    acc = float(accuracy_score(targets, preds))
    prec = float(precision_score(targets, preds, zero_division=0))
    rec = float(recall_score(targets, preds, zero_division=0))
    f1 = float(f1_score(targets, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    # Confidences
    fake_probs = probs[targets == 1.0]
    real_probs = probs[targets == 0.0]

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'errors': int(fp + fn),
        'fake_confidence_mean': float(np.mean(fake_probs)),
        'real_confidence_mean': float(np.mean(1.0 - real_probs)),
        'all_probs': probs,
        'all_targets': targets
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 95, flush=True)
    print("PURE FACIAL BENCHMARK EVALUATION: M3-B16 BASELINE VS. M6-B16 (COORDATT)", flush=True)
    print(f"Device: {device} | Decision Threshold: tau = 0.50 (LOCKED)", flush=True)
    print("=" * 95, flush=True)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m3_path = os.path.join(repo_dir, "models", "efficientnet_b3_baseline_m3_16_best.pth")
    m6_path = os.path.join(repo_dir, "models", "efficientnet_b3_coordatt_compression_aware_m6_16_best.pth")
    out_dir = os.path.join(repo_dir, "results", "pure_facial_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    _, eval_tx = get_phase2_transforms()

    # Load Models
    m3 = EfficientNetB3Baseline(pretrained=False)
    m3.load_state_dict(torch.load(m3_path, map_location=device)['model_state_dict'])
    m3.to(device).eval()

    m6 = EfficientNetB3CoordAtt(pretrained=False)
    m6.load_state_dict(torch.load(m6_path, map_location=device)['model_state_dict'])
    m6.to(device).eval()

    # Build Unseen Pure Facial Set
    used_filenames = set()
    for s_file in ['train.csv', 'val.csv', 'test.csv']:
        sp = os.path.join(repo_dir, "data", s_file)
        with open(sp) as f:
            for r in csv.DictReader(f):
                used_filenames.add(r['original_filename'])

    test_real_dir = os.path.join(repo_dir, "Test", "Real")
    test_fake_dir = os.path.join(repo_dir, "Test", "Fake")

    unseen_real_files = [f for f in os.listdir(test_real_dir) if f not in used_filenames][:100]
    unseen_fake_files = [f for f in os.listdir(test_fake_dir) if f not in used_filenames][:100]

    records = []
    for f in unseen_real_files:
        records.append({
            'path': os.path.join(test_real_dir, f),
            'filename': f,
            'class': 'Real',
            'label': 0.0
        })
    for f in unseen_fake_files:
        records.append({
            'path': os.path.join(test_fake_dir, f),
            'filename': f,
            'class': 'Fake',
            'label': 1.0
        })

    print(f"Loaded {len(records)} PURE FACIAL test samples (100 Real faces, 100 Fake faces)", flush=True)
    print(f"Zero-Leakage Verified: Exactly 0 samples overlap with train.csv, val.csv, or test.csv", flush=True)

    conditions = [
        ('Clean', None),
        ('QF80', 80),
        ('QF60', 60),
        ('QF50', 50),
        ('QF40', 40),
        ('QF20', 20)
    ]

    print("\n" + "=" * 95, flush=True)
    print(f"{'Condition':<8} | {'Model':<9} | {'Accuracy':<8} | {'Recall':<8} | {'Prec':<7} | {'FN':<4} | {'FP':<4} | {'Errors':<6} | {'Confidence (Fake/Real)'}", flush=True)
    print("-" * 95, flush=True)

    results_table = []
    for c_name, qf in conditions:
        ds = PureFaceDataset(records, condition=c_name.lower(), qf=qf, transform=eval_tx)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        res3 = evaluate_loader(m3, loader, device)
        res6 = evaluate_loader(m6, loader, device)

        err_diff = res3['errors'] - res6['errors']
        adv_str = f"M6 WINS (-{err_diff} err)" if err_diff > 0 else (f"M3 WINS (-{-err_diff} err)" if err_diff < 0 else "TIED")

        print(f"{c_name:<8} | M3(Base)  | {res3['accuracy']*100:5.2f}%  | {res3['recall']*100:5.2f}%  | {res3['precision']*100:5.2f}% | {res3['FN']:>3d} | {res3['FP']:>3d} | {res3['errors']:>4d}   | Fake:{res3['fake_confidence_mean']*100:5.1f}% Real:{res3['real_confidence_mean']*100:5.1f}%", flush=True)
        print(f"{c_name:<8} | M6(Coord) | {res6['accuracy']*100:5.2f}%  | {res6['recall']*100:5.2f}%  | {res6['precision']*100:5.2f}% | {res6['FN']:>3d} | {res6['FP']:>3d} | {res6['errors']:>4d}   | Fake:{res6['fake_confidence_mean']*100:5.1f}% Real:{res6['real_confidence_mean']*100:5.1f}% | {adv_str}", flush=True)
        print("-" * 95, flush=True)

        results_table.append({
            'condition': c_name,
            'm3_accuracy': f"{res3['accuracy']*100:.2f}%",
            'm6_accuracy': f"{res6['accuracy']*100:.2f}%",
            'm3_recall': f"{res3['recall']*100:.2f}%",
            'm6_recall': f"{res6['recall']*100:.2f}%",
            'm3_precision': f"{res3['precision']*100:.2f}%",
            'm6_precision': f"{res6['precision']*100:.2f}%",
            'm3_fn': res3['FN'], 'm6_fn': res6['FN'],
            'm3_fp': res3['FP'], 'm6_fp': res6['FP'],
            'm3_errors': res3['errors'], 'm6_errors': res6['errors'],
            'm6_net_gain': err_diff,
            'm3_fake_confidence': f"{res3['fake_confidence_mean']*100:.1f}%",
            'm6_fake_confidence': f"{res6['fake_confidence_mean']*100:.1f}%",
            'm3_real_confidence': f"{res3['real_confidence_mean']*100:.1f}%",
            'm6_real_confidence': f"{res6['real_confidence_mean']*100:.1f}%"
        })

    # Save to CSV
    csv_path = os.path.join(out_dir, "pure_facial_dataset_m3_vs_m6_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
        writer.writeheader()
        writer.writerows(results_table)
    print(f"\nSaved benchmark results to {csv_path}", flush=True)

    # Save JSON
    with open(os.path.join(out_dir, "pure_facial_benchmark_summary.json"), 'w') as f:
        json.dump(results_table, f, indent=2)

    print("=" * 95, flush=True)
    print("PURE FACIAL BENCHMARK COMPLETE!", flush=True)
    print("=" * 95, flush=True)

if __name__ == '__main__':
    main()
