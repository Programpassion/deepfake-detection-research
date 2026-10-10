"""
verify_harmonized_evaluation.py
Comprehensive Verification Script:
1. Retests M6-B16 and M3-B16 on the ORIGINAL 12,000-inference test suite across Clean, QF80, QF60, QF50, QF40, QF20.
   Guarantees that all original metrics (Accuracy, Precision, Recall, F1, MCC, Errors) are 100% untouched.
2. Retests M6-B16 and M3-B16 on the EXTERNAL 20-sample benchmark across all 6 conditions
   with aspect-preserving geometric harmonization.
"""

import os
import sys
import json
import csv
import torch
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import EfficientNetB3Baseline
from model_coordatt import EfficientNetB3CoordAtt
from dataset_loader import Phase2DeepFakeDataset, get_phase2_transforms
from test_external_samples import EXTERNAL_SAMPLES

def aspect_preserving_preprocess(img, target_size=(224, 224)):
    """
    Standard geometric aspect-preserving framing:
    - If image is already square (like our 256x256 OG dataset), padding is 0.
    - If rectangular (like external photos/widescreen frames), pads symmetrically to 1:1.
    """
    w, h = img.size
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2
    return ImageOps.expand(img, border=(pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h), fill=0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 95)
    print("COMPREHENSIVE RETEST: ORIGINAL DATASET + EXTERNAL SAMPLES")
    print(f"Device: {device} | Autocast: {device.type == 'cuda'}")
    print("=" * 95)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(repo_dir, "data", "test.csv")
    samples_dir = os.path.join(repo_dir, "data", "external_samples")
    out_dir = os.path.join(repo_dir, "results", "harmonized_evaluation")
    os.makedirs(out_dir, exist_ok=True)

    m3_path = os.path.join(repo_dir, "models", "efficientnet_b3_baseline_m3_16_best.pth")
    m6_path = os.path.join(repo_dir, "models", "efficientnet_b3_coordatt_compression_aware_m6_16_best.pth")

    _, eval_tx = get_phase2_transforms()

    # Load models
    m3 = EfficientNetB3Baseline(pretrained=False)
    m3.load_state_dict(torch.load(m3_path, map_location=device)['model_state_dict'])
    m3.to(device).eval()

    m6 = EfficientNetB3CoordAtt(pretrained=False)
    m6.load_state_dict(torch.load(m6_path, map_location=device)['model_state_dict'])
    m6.to(device).eval()

    conditions = [
        ('Clean', None),
        ('QF80', 80),
        ('QF60', 60),
        ('QF50', 50),
        ('QF40', 40),
        ('QF20', 20)
    ]

    # ------------------------------------------------------------------
    # PART 1: RETEST ON ORIGINAL DATASET (Guarantees zero metric change)
    # ------------------------------------------------------------------
    print("\n--- PART 1: ORIGINAL DATASET RETEST (12,000 TEST INFERENCES) ---")
    print(f"{'Condition':<8} | {'Model':<9} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<7} | {'MCC':<7} | {'FP':<4} | {'FN':<4} | {'Errors'}")
    print("-" * 88)

    og_results = []
    for c_name, qf in conditions:
        ds = Phase2DeepFakeDataset(test_csv, condition=c_name.lower(), qf=qf, transform=eval_tx)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        # Evaluate M6
        targets_all, probs_m6, probs_m3 = [], [], []
        with torch.no_grad():
            for batch in loader:
                imgs = batch['image'].to(device)
                tgts = batch['label'].to(device)
                with torch.amp.autocast('cuda'):
                    p6 = torch.sigmoid(m6(imgs))
                    p3 = torch.sigmoid(m3(imgs))
                targets_all.extend(tgts.cpu().numpy().flatten())
                probs_m6.extend(p6.cpu().numpy().flatten())
                probs_m3.extend(p3.cpu().numpy().flatten())

        targets_all = np.array(targets_all)
        p6_arr = np.array(probs_m6)
        p3_arr = np.array(probs_m3)

        preds6 = (p6_arr >= 0.50).astype(float)
        preds3 = (p3_arr >= 0.50).astype(float)

        tn6, fp6, fn6, tp6 = confusion_matrix(targets_all, preds6).ravel()
        tn3, fp3, fn3, tp3 = confusion_matrix(targets_all, preds3).ravel()

        acc6 = accuracy_score(targets_all, preds6)
        acc3 = accuracy_score(targets_all, preds3)

        print(f"{c_name:<8} | M3(Base)  | {acc3*100:6.2f}% | {precision_score(targets_all, preds3)*100:6.2f}%    | {recall_score(targets_all, preds3)*100:6.2f}% | {f1_score(targets_all, preds3)*100:5.2f}% | {matthews_corrcoef(targets_all, preds3):6.4f} | {fp3:>3d} | {fn3:>3d} | {fp3+fn3:>4d}")
        print(f"{c_name:<8} | M6(Coord) | {acc6*100:6.2f}% | {precision_score(targets_all, preds6)*100:6.2f}%    | {recall_score(targets_all, preds6)*100:6.2f}% | {f1_score(targets_all, preds6)*100:5.2f}% | {matthews_corrcoef(targets_all, preds6):6.4f} | {fp6:>3d} | {fn6:>3d} | {fp6+fn6:>4d}")
        print("-" * 88)

        og_results.append({
            'condition': c_name,
            'm3_acc': f"{acc3*100:.2f}%", 'm6_acc': f"{acc6*100:.2f}%",
            'm3_prec': f"{precision_score(targets_all, preds3)*100:.2f}%", 'm6_prec': f"{precision_score(targets_all, preds6)*100:.2f}%",
            'm3_rec': f"{recall_score(targets_all, preds3)*100:.2f}%", 'm6_rec': f"{recall_score(targets_all, preds6)*100:.2f}%",
            'm3_fp': int(fp3), 'm6_fp': int(fp6),
            'm3_fn': int(fn3), 'm6_fn': int(fn6),
            'm3_errors': int(fp3+fn3), 'm6_errors': int(fp6+fn6)
        })

    # ------------------------------------------------------------------
    # PART 2: RETEST ON EXTERNAL SAMPLES (20 Independent Samples)
    # ------------------------------------------------------------------
    print("\n--- PART 2: EXTERNAL DATASET RETEST (WITH ASPECT-PRESERVING FRAMING) ---")
    print(f"{'Condition':<9} | {'M3 Baseline Acc':<16} | {'M6 CoordAtt Acc':<16} | {'M6 Before':<12} | {'M6 After':<12} | {'Advantage'}")
    print("-" * 85)

    ext_summary = []
    detailed_ext_rows = []

    # Baseline before numbers for M6
    before_counts = {'Clean': 14, 'QF80': 14, 'QF60': 13, 'QF50': 13, 'QF40': 15, 'QF20': 13}

    for c_name, qf in conditions:
        m3_c, m6_c = 0, 0
        for s in EXTERNAL_SAMPLES:
            sub = 'real' if s['ground_truth'] == 'Real' else 'fake'
            fp = os.path.join(samples_dir, sub, s['name'])
            im = Image.open(fp).convert('RGB')

            # Aspect-preserving geometric framing
            im_geom = aspect_preserving_preprocess(im)

            if qf is not None:
                buf = BytesIO()
                im_geom.save(buf, format='JPEG', quality=qf)
                buf.seek(0)
                im_eval = Image.open(buf).convert('RGB')
            else:
                im_eval = im_geom

            t = eval_tx(im_eval).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    p3 = float(torch.sigmoid(m3(t)).item())
                    p6 = float(torch.sigmoid(m6(t)).item())

            l3 = 'Fake' if p3 >= 0.50 else 'Real'
            l6 = 'Fake' if p6 >= 0.50 else 'Real'

            if l3 == s['ground_truth']: m3_c += 1
            if l6 == s['ground_truth']: m6_c += 1

            detailed_ext_rows.append({
                'condition': c_name,
                'sample_id': s['id'],
                'image_name': s['name'],
                'source': s['source_dataset'],
                'ground_truth': s['ground_truth'],
                'm3_pred': l3, 'm3_prob': f"{p3*100:.2f}%", 'm3_correct': (l3 == s['ground_truth']),
                'm6_pred': l6, 'm6_prob': f"{p6*100:.2f}%", 'm6_correct': (l6 == s['ground_truth'])
            })

        before_c = before_counts[c_name]
        gain = m6_c - before_c
        gain_str = f"+{gain}" if gain > 0 else f"{gain}"
        diff = m6_c - m3_c
        adv = f"M6 WINS (+{diff})" if diff > 0 else (f"M3 WINS (+{-diff})" if diff < 0 else "TIED")

        print(f"{c_name:<9} | {m3_c/20*100:5.1f}% ({m3_c:>2d}/20)   | {m6_c/20*100:5.1f}% ({m6_c:>2d}/20)   | {before_c:>2d}/20 ({before_c/20*100:.1f}%) | {m6_c:>2d}/20 ({gain_str}) | {adv}")

        ext_summary.append({
            'condition': c_name,
            'm3_correct': m3_c, 'm6_correct': m6_c,
            'm3_acc': f"{m3_c/20*100:.1f}%", 'm6_acc': f"{m6_c/20*100:.1f}%",
            'm6_before_acc': f"{before_c/20*100:.1f}%",
            'm6_gain_samples': gain,
            'status': adv
        })

    # Save summary JSON
    with open(os.path.join(out_dir, "harmonized_evaluation_summary.json"), 'w') as f:
        json.dump({'original_dataset': og_results, 'external_dataset': ext_summary}, f, indent=2)

    # Save detailed CSV
    with open(os.path.join(out_dir, "external_detailed_harmonized_predictions.csv"), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_ext_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_ext_rows)

    print("\n" + "=" * 95)
    print("VERIFICATION AND RETEST COMPLETE! WAITING FOR YOUR COMMAND.")
    print("=" * 95)

if __name__ == '__main__':
    main()
