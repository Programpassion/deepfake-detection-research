"""
evaluate_multi_qf.py
Multi-QF Evaluation Suite for M3-B16 (Base) vs. M6-B16 (Coordinate Attention)
Evaluates across 6 test conditions (Clean, QF80, QF60, QF50, QF40, QF20; 2,000 images each).
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import EfficientNetB3Baseline
from model_coordatt import EfficientNetB3CoordAtt
from dataset_loader import Phase2DeepFakeDataset, get_phase2_transforms

def evaluate_model(model, dataloader, device):
    model.eval()
    all_targets = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                probs = torch.sigmoid(logits)
            preds = (probs >= 0.50).float()

            all_targets.extend(targets.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, zero_division=0))
    rec = float(recall_score(all_targets, all_preds, zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, zero_division=0))
    roc_auc = float(roc_auc_score(all_targets, all_probs))
    mcc = float(matthews_corrcoef(all_targets, all_preds))
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': roc_auc, 'mcc': mcc, 'TP': int(tp), 'TN': int(tn),
        'FP': int(fp), 'FN': int(fn), 'errors': int(fp + fn)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("MULTI-QF EVALUATION: M3-B16 (BASE) VS. M6-B16 (COORDINATE ATTENTION)")
    print(f"Device: {device}")
    print("=" * 80)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(repo_dir, "data", "test.csv")
    m3_path = os.path.join(repo_dir, "models", "efficientnet_b3_baseline_m3_16_best.pth")
    m6_path = os.path.join(repo_dir, "models", "efficientnet_b3_coordatt_compression_aware_m6_16_best.pth")

    _, eval_tx = get_phase2_transforms()

    m3 = EfficientNetB3Baseline(pretrained=False)
    m3.load_state_dict(torch.load(m3_path, map_location=device)['model_state_dict'])
    m3.to(device)

    m6 = EfficientNetB3CoordAtt(pretrained=False)
    m6.load_state_dict(torch.load(m6_path, map_location=device)['model_state_dict'])
    m6.to(device)

    conditions = [
        ('Clean', None),
        ('QF80', 80),
        ('QF60', 60),
        ('QF50', 50),
        ('QF40', 40),
        ('QF20', 20)
    ]

    print(f"{'Condition':<8} | {'Model':<10} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'MCC':<7} | {'FP':<4} | {'FN':<4} | {'Errors':<6}")
    print("-" * 80)

    for cond_name, qf in conditions:
        ds = Phase2DeepFakeDataset(test_csv, condition=cond_name.lower(), qf=qf, transform=eval_tx)
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        res3 = evaluate_model(m3, loader, device)
        res6 = evaluate_model(m6, loader, device)

        print(f"{cond_name:<8} | M3 (Base)  | {res3['accuracy']*100:5.2f}% | {res3['precision']*100:5.2f}% | {res3['recall']*100:5.2f}% | {res3['f1']*100:5.2f}% | {res3['mcc']:6.4f} | {res3['FP']:>3d} | {res3['FN']:>3d} | {res3['errors']:>4d}")
        print(f"{cond_name:<8} | M6 (Coord) | {res6['accuracy']*100:5.2f}% | {res6['precision']*100:5.2f}% | {res6['recall']*100:5.2f}% | {res6['f1']*100:5.2f}% | {res6['mcc']:6.4f} | {res6['FP']:>3d} | {res6['FN']:>3d} | {res6['errors']:>4d}")
        print("-" * 80)

if __name__ == '__main__':
    main()
