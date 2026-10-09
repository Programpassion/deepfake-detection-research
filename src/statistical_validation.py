"""
statistical_validation.py
Rigorous Statistical Significance, 95% Confidence Intervals,
and Validation-Based Threshold Selection for M3-B16 vs. M6-B16.
Designed to meet IEEE peer-review standards.
"""

import os
import sys
import json
import csv
import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, confusion_matrix
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import EfficientNetB3Baseline
from model_coordatt import EfficientNetB3CoordAtt
from dataset_loader import Phase2DeepFakeDataset, get_phase2_transforms

def wilson_score_interval(k, n, confidence=0.95):
    """Computes Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return float(p), float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))

def mcnemar_test(y_true, preds_a, preds_b):
    """
    McNemar's test with continuity correction.
    preds_a = baseline (M3)
    preds_b = proposed (M6)
    b = A correct, B incorrect
    c = A incorrect, B correct
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Contingency matrix
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if (b + c) == 0:
        return {'b': b, 'c': c, 'chi2': 0.0, 'p_value': 1.0, 'significant': False}

    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
    return {
        'b_m3_correct_m6_wrong': b,
        'c_m3_wrong_m6_correct': c,
        'chi2_stat': float(chi2),
        'p_value': float(p_val),
        'significant_p05': bool(p_val < 0.05),
        'significant_p01': bool(p_val < 0.01)
    }

def run_inference(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                probs = torch.sigmoid(logits)

            p_np = probs.cpu().numpy().flatten()
            t_np = targets.cpu().numpy().flatten()

            for i in range(len(p_np)):
                results.append({
                    'image_id': batch['image_id'][i],
                    'original_filename': batch['original_filename'][i],
                    'class': batch['class'][i],
                    'target': float(t_np[i]),
                    'prob': float(p_np[i])
                })
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print("STATISTICAL RIGOR & VALIDATION THRESHOLD SELECTION FOR IEEE PEER-REVIEW")
    print(f"Device: {device} | Autocast: {device.type == 'cuda'}")
    print("=" * 90)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    val_csv = os.path.join(repo_dir, "data", "val.csv")
    test_csv = os.path.join(repo_dir, "data", "test.csv")
    m3_path = os.path.join(repo_dir, "models", "efficientnet_b3_baseline_m3_16_best.pth")
    m6_path = os.path.join(repo_dir, "models", "efficientnet_b3_coordatt_compression_aware_m6_16_best.pth")
    out_dir = os.path.join(repo_dir, "results", "fn_analysis")
    os.makedirs(out_dir, exist_ok=True)

    _, eval_tx = get_phase2_transforms()

    # Load Models
    m3 = EfficientNetB3Baseline(pretrained=False)
    m3.load_state_dict(torch.load(m3_path, map_location=device)['model_state_dict'])
    m3.to(device)

    m6 = EfficientNetB3CoordAtt(pretrained=False)
    m6.load_state_dict(torch.load(m6_path, map_location=device)['model_state_dict'])
    m6.to(device)

    # -------------------------------------------------------------
    # 1. Validation-Set Threshold Optimization (NO TEST LEAKAGE)
    # -------------------------------------------------------------
    print("\n--- PHASE 1: PRE-DECLARED THRESHOLD TUNING ON VALIDATION DATA (val.csv) ---")
    val_conditions = [('Val_Clean', None), ('Val_QF80', 80), ('Val_QF60', 60), ('Val_QF50', 50), ('Val_QF40', 40), ('Val_QF20', 20)]
    
    val_m6_all_recs = []
    for cond_name, qf in val_conditions[1:]: # compressed val conditions
        ds = Phase2DeepFakeDataset(val_csv, condition=cond_name.lower(), qf=qf, transform=eval_tx)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        recs = run_inference(m6, loader, device)
        val_m6_all_recs.extend(recs)

    val_targets = np.array([r['target'] for r in val_m6_all_recs])
    val_probs = np.array([r['prob'] for r in val_m6_all_recs])

    candidate_thresholds = np.linspace(0.10, 0.50, 41) # step 0.01
    best_f1_th = 0.50
    best_f1_val = 0.0
    best_f2_th = 0.50
    best_f2_val = 0.0
    best_youden_th = 0.50
    best_youden_val = 0.0

    for th in candidate_thresholds:
        preds = (val_probs >= th).astype(float)
        f1 = f1_score(val_targets, preds, zero_division=0)
        f2 = fbeta_score(val_targets, preds, beta=2.0, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(val_targets, preds).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = rec + spec - 1.0

        if f1 > best_f1_val:
            best_f1_val = f1
            best_f1_th = th
        if f2 > best_f2_val:
            best_f2_val = f2
            best_f2_th = th
        if youden > best_youden_val:
            best_youden_val = youden
            best_youden_th = th

    print(f"Validation Tuning Results across 5 compressed conditions (4,000 val inferences):")
    print(f"  Criterion A (Max F1 Score)     : Optimal tau = {best_f1_th:.2f} (Val F1 = {best_f1_val:.4f})")
    print(f"  Criterion B (Max F2 Score)     : Optimal tau = {best_f2_th:.2f} (Val F2 = {best_f2_val:.4f}) [Prioritizes FN reduction]")
    print(f"  Criterion C (Max Youden's J)   : Optimal tau = {best_youden_th:.2f} (Val J = {best_youden_val:.4f})")

    # -------------------------------------------------------------
    # 2. Test-Set McNemar Significance Testing at Primary Tau = 0.50
    # -------------------------------------------------------------
    print("\n--- PHASE 2: STATISTICAL SIGNIFICANCE TESTING ON TEST SET (tau = 0.50) ---")
    test_conditions = [
        ('Clean', None),
        ('QF80', 80),
        ('QF60', 60),
        ('QF50', 50),
        ('QF40', 40),
        ('QF20', 20),
        ('QF10', 10)
    ]

    all_m3_test = {}
    all_m6_test = {}
    mcnemar_results = {}
    confidence_intervals = {}

    for cond_name, qf in test_conditions:
        ds = Phase2DeepFakeDataset(test_csv, condition=cond_name.lower(), qf=qf, transform=eval_tx)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        recs3 = run_inference(m3, loader, device)
        recs6 = run_inference(m6, loader, device)
        all_m3_test[cond_name] = recs3
        all_m6_test[cond_name] = recs6

        targets = np.array([r['target'] for r in recs3])
        probs3 = np.array([r['prob'] for r in recs3])
        probs6 = np.array([r['prob'] for r in recs6])

        preds3 = (probs3 >= 0.50).astype(float)
        preds6 = (probs6 >= 0.50).astype(float)

        # Overall accuracy McNemar
        mcnemar_overall = mcnemar_test(targets, preds3, preds6)

        # Paired McNemar specifically on Fake samples (target == 1.0) for False Negatives!
        fake_idx = (targets == 1.0)
        mcnemar_fake = mcnemar_test(targets[fake_idx], preds3[fake_idx], preds6[fake_idx])

        # Wilson 95% Confidence Intervals for Recall and FNR
        tp3 = int(np.sum((targets == 1.0) & (preds3 == 1.0)))
        fn3 = int(np.sum((targets == 1.0) & (preds3 == 0.0)))
        tp6 = int(np.sum((targets == 1.0) & (preds6 == 1.0)))
        fn6 = int(np.sum((targets == 1.0) & (preds6 == 0.0)))

        rec3_p, rec3_low, rec3_high = wilson_score_interval(tp3, 1000)
        rec6_p, rec6_low, rec6_high = wilson_score_interval(tp6, 1000)
        fnr3_p, fnr3_low, fnr3_high = wilson_score_interval(fn3, 1000)
        fnr6_p, fnr6_low, fnr6_high = wilson_score_interval(fn6, 1000)

        mcnemar_results[cond_name] = {
            'overall': mcnemar_overall,
            'fake_recall': mcnemar_fake
        }

        confidence_intervals[cond_name] = {
            'm3_recall_95ci': [rec3_low, rec3_high],
            'm6_recall_95ci': [rec6_low, rec6_high],
            'm3_fnr_95ci': [fnr3_low, fnr3_high],
            'm6_fnr_95ci': [fnr6_low, fnr6_high],
            'm3_tp': tp3, 'm3_fn': fn3,
            'm6_tp': tp6, 'm6_fn': fn6
        }

        print(f"[{cond_name:<6}] Overall McNemar: p={mcnemar_overall['p_value']:.4e} (chi2={mcnemar_overall['chi2_stat']:.2f}) | "
              f"FN McNemar: p={mcnemar_fake['p_value']:.4e} | "
              f"M6 Recall: {rec6_p*100:.2f}% [{rec6_low*100:.2f}%, {rec6_high*100:.2f}%] vs M3: {rec3_p*100:.2f}% [{rec3_low*100:.2f}%, {rec3_high*100:.2f}%]")

    # -------------------------------------------------------------
    # 3. Aggregate 5-Condition Compressed Significance
    # -------------------------------------------------------------
    comp_5 = ['QF80', 'QF60', 'QF50', 'QF40', 'QF20']
    agg_targets = np.concatenate([np.array([r['target'] for r in all_m3_test[c]]) for c in comp_5])
    agg_preds3 = np.concatenate([(np.array([r['prob'] for r in all_m3_test[c]]) >= 0.50).astype(float) for c in comp_5])
    agg_preds6 = np.concatenate([(np.array([r['prob'] for r in all_m6_test[c]]) >= 0.50).astype(float) for c in comp_5])

    agg_mcnemar_overall = mcnemar_test(agg_targets, agg_preds3, agg_preds6)
    agg_fake_idx = (agg_targets == 1.0)
    agg_mcnemar_fake = mcnemar_test(agg_targets[agg_fake_idx], agg_preds3[agg_fake_idx], agg_preds6[agg_fake_idx])

    print("\n--- AGGREGATE 5 COMPRESSED CONDITIONS (5,000 Inferences) ---")
    print(f"Overall Accuracy McNemar: chi2={agg_mcnemar_overall['chi2_stat']:.3f}, p-value = {agg_mcnemar_overall['p_value']:.4e} (p < 0.001: {agg_mcnemar_overall['p_value'] < 0.001})")
    print(f"Fake Recall (FN) McNemar: chi2={agg_mcnemar_fake['chi2_stat']:.3f}, p-value = {agg_mcnemar_fake['p_value']:.4e}")
    print(f"  M3 correct & M6 wrong: {agg_mcnemar_overall['b_m3_correct_m6_wrong']}")
    print(f"  M6 correct & M3 wrong: {agg_mcnemar_overall['c_m3_wrong_m6_correct']}")

    # -------------------------------------------------------------
    # 4. Independent Test Evaluation of Validated Threshold tau*
    # -------------------------------------------------------------
    print(f"\n--- PHASE 3: EVALUATION OF VALIDATION-DERIVED OPERATING THRESHOLDS ON TEST SET ---")
    print(f"Evaluating M6 at validation-derived thresholds on 5 compressed test conditions:")
    
    for label, th in [("Canonical (Default)", 0.50), 
                      (f"Validation F1-Tuned (tau={best_f1_th:.2f})", best_f1_th),
                      (f"Validation F2-Tuned (tau={best_f2_th:.2f})", best_f2_th)]:
        tot_fn = 0
        tot_fp = 0
        tot_err = 0
        recs = []
        precs = []
        accs = []
        for c in comp_5:
            recs_c = all_m6_test[c]
            t = np.array([r['target'] for r in recs_c])
            p = np.array([r['prob'] for r in recs_c])
            pr = (p >= th).astype(float)
            tn, fp, fn, tp = confusion_matrix(t, pr).ravel()
            tot_fn += int(fn)
            tot_fp += int(fp)
            tot_err += int(fp + fn)
            recs.append(tp / (tp + fn))
            precs.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            accs.append((tp + tn) / len(t))
        
        print(f"  {label:<38}: Recall = {np.mean(recs)*100:5.2f}% | Prec = {np.mean(precs)*100:5.2f}% | Acc = {np.mean(accs)*100:5.2f}% | Total FN = {tot_fn:>3d} | Total FP = {tot_fp:>3d} | Errors = {tot_err:>3d}")

    # -------------------------------------------------------------
    # 5. Save Report
    # -------------------------------------------------------------
    stat_report_path = os.path.join(out_dir, "m3_vs_m6_statistical_significance_report.json")
    with open(stat_report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'validation_tuning': {
                'best_f1_threshold': float(best_f1_th),
                'best_f1_score': float(best_f1_val),
                'best_f2_threshold': float(best_f2_th),
                'best_f2_score': float(best_f2_val),
                'best_youden_threshold': float(best_youden_th),
                'best_youden_stat': float(best_youden_val)
            },
            'mcnemar_condition_results': mcnemar_results,
            'aggregate_5_compressed_mcnemar': {
                'overall': agg_mcnemar_overall,
                'fake_recall': agg_mcnemar_fake
            },
            'confidence_intervals_95': confidence_intervals
        }, f, indent=2)
    print(f"\nSaved statistical report to {stat_report_path}")

if __name__ == '__main__':
    main()
