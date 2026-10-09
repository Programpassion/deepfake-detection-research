"""
evaluate_m6_fn_focus.py
Deep-Dive Evaluation of M6-B16 (Coordinate Attention) Across Quality Factors
with Strict Focus on False Negatives (FN) Behavior and Compression Sensitivity.
"""

import os
import sys
import json
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, matthews_corrcoef, confusion_matrix,
    average_precision_score
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import EfficientNetB3Baseline
from model_coordatt import EfficientNetB3CoordAtt
from dataset_loader import Phase2DeepFakeDataset, get_phase2_transforms

def run_condition_inference(model, dataloader, device):
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

def compute_metrics_at_threshold(records, threshold=0.50):
    targets = np.array([r['target'] for r in records])
    probs = np.array([r['prob'] for r in records])
    preds = (probs >= threshold).astype(float)

    acc = float(accuracy_score(targets, preds))
    prec = float(precision_score(targets, preds, zero_division=0))
    rec = float(recall_score(targets, preds, zero_division=0))
    f1 = float(f1_score(targets, preds, zero_division=0))
    f2 = float(fbeta_score(targets, preds, beta=2.0, zero_division=0))
    roc_auc = float(roc_auc_score(targets, probs))
    pr_auc = float(average_precision_score(targets, probs))
    mcc = float(matthews_corrcoef(targets, preds))
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()

    # Probability analysis on Fake samples (target == 1.0)
    fake_probs = probs[targets == 1.0]
    real_probs = probs[targets == 0.0]

    # FN breakdown by confidence bands
    fn_probs = fake_probs[fake_probs < threshold]
    borderline_fn = int(np.sum((fake_probs >= 0.40) & (fake_probs < threshold)))
    moderate_fn = int(np.sum((fake_probs >= 0.20) & (fake_probs < 0.40)))
    severe_fn = int(np.sum(fake_probs < 0.20))

    return {
        'threshold': float(threshold),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'f2': f2,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'total_errors': int(fp + fn),
        'FNR': float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'FPR': float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'fn_error_pct': float(fn / (fp + fn) * 100.0) if (fp + fn) > 0 else 0.0,
        'fp_error_pct': float(fp / (fp + fn) * 100.0) if (fp + fn) > 0 else 0.0,
        'fake_prob_mean': float(np.mean(fake_probs)),
        'fake_prob_median': float(np.median(fake_probs)),
        'fake_prob_std': float(np.std(fake_probs)),
        'real_prob_mean': float(np.mean(real_probs)),
        'real_prob_median': float(np.median(real_probs)),
        'real_prob_std': float(np.std(real_probs)),
        'borderline_fn_count': borderline_fn,
        'moderate_fn_count': moderate_fn,
        'severe_fn_count': severe_fn,
        'fn_mean_prob': float(np.mean(fn_probs)) if len(fn_probs) > 0 else 0.0,
        'fn_median_prob': float(np.median(fn_probs)) if len(fn_probs) > 0 else 0.0
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print("FALSE NEGATIVE (FN) DEEP-DIVE EVALUATION: M6-B16 (COORDINATE ATTENTION)")
    print(f"Device: {device} | Autocast: {device.type == 'cuda'}")
    print("=" * 90)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(repo_dir, "data", "test.csv")
    m6_path = os.path.join(repo_dir, "models", "efficientnet_b3_coordatt_compression_aware_m6_16_best.pth")
    m3_path = os.path.join(repo_dir, "models", "efficientnet_b3_baseline_m3_16_best.pth")
    out_dir = os.path.join(repo_dir, "results", "fn_analysis")
    viz_dir = os.path.join(repo_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    _, eval_tx = get_phase2_transforms()

    # Load M6
    print(f"Loading M6-B16 from {os.path.basename(m6_path)}...")
    m6 = EfficientNetB3CoordAtt(pretrained=False)
    m6.load_state_dict(torch.load(m6_path, map_location=device)['model_state_dict'])
    m6.to(device)

    # Load M3 for baseline FN comparison
    print(f"Loading M3-B16 from {os.path.basename(m3_path)}...")
    m3 = EfficientNetB3Baseline(pretrained=False)
    m3.load_state_dict(torch.load(m3_path, map_location=device)['model_state_dict'])
    m3.to(device)

    conditions = [
        ('Clean', None),
        ('QF90', 90),
        ('QF80', 80),
        ('QF70', 70),
        ('QF60', 60),
        ('QF50', 50),
        ('QF40', 40),
        ('QF30', 30),
        ('QF20', 20),
        ('QF10', 10)
    ]

    all_m6_records = {}
    all_m3_records = {}
    m6_summary = []
    m3_summary = []

    print("\nRunning multi-condition inference across 10 Quality Factors...")
    print(f"{'Condition':<8} | {'Model':<9} | {'Acc':<7} | {'Recall':<7} | {'FN':<5} | {'FP':<5} | {'Errors':<6} | {'FNR %':<7} | {'% Err is FN':<11} | {'Fake P(Med)':<11}")
    print("-" * 95)

    for cond_name, qf in conditions:
        ds = Phase2DeepFakeDataset(test_csv, condition=cond_name.lower(), qf=qf, transform=eval_tx)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        # M6 Inference
        m6_recs = run_condition_inference(m6, loader, device)
        all_m6_records[cond_name] = m6_recs
        m6_met = compute_metrics_at_threshold(m6_recs, threshold=0.50)
        m6_met['condition'] = cond_name
        m6_met['qf'] = qf if qf is not None else 100
        m6_summary.append(m6_met)

        # M3 Inference
        m3_recs = run_condition_inference(m3, loader, device)
        all_m3_records[cond_name] = m3_recs
        m3_met = compute_metrics_at_threshold(m3_recs, threshold=0.50)
        m3_met['condition'] = cond_name
        m3_met['qf'] = qf if qf is not None else 100
        m3_summary.append(m3_met)

        print(f"{cond_name:<8} | M6(Coord) | {m6_met['accuracy']*100:5.2f}% | {m6_met['recall']*100:5.2f}% | {m6_met['FN']:>4d} | {m6_met['FP']:>4d} | {m6_met['total_errors']:>5d} | {m6_met['FNR']*100:5.2f}% | {m6_met['fn_error_pct']:8.2f}% | {m6_met['fake_prob_median']:9.4f}")
        print(f"{cond_name:<8} | M3(Base)  | {m3_met['accuracy']*100:5.2f}% | {m3_met['recall']*100:5.2f}% | {m3_met['FN']:>4d} | {m3_met['FP']:>4d} | {m3_met['total_errors']:>5d} | {m3_met['FNR']*100:5.2f}% | {m3_met['fn_error_pct']:8.2f}% | {m3_met['fake_prob_median']:9.4f}")
        print("-" * 95)

    # -------------------------------------------------------------
    # 1. Sample-level FN Tracking across conditions (M6)
    # -------------------------------------------------------------
    fake_samples = [r['image_id'] for r in all_m6_records['Clean'] if r['target'] == 1.0]
    print(f"\nTotal Fake test samples analyzed: {len(fake_samples)}")

    # For each fake sample, track whether it is FN across conditions
    fn_status_by_sample = {}
    probs_by_sample = {}
    for sample_id in fake_samples:
        fn_status_by_sample[sample_id] = {}
        probs_by_sample[sample_id] = {}

    for cond_name, _ in conditions:
        for r in all_m6_records[cond_name]:
            if r['target'] == 1.0:
                sid = r['image_id']
                is_fn = (r['prob'] < 0.50)
                fn_status_by_sample[sid][cond_name] = is_fn
                probs_by_sample[sid][cond_name] = r['prob']

    # Classify each Fake sample:
    # 1. Always Correct (Never FN in any of 10 conditions)
    # 2. Hard FN (FN in Clean + all or most QFs)
    # 3. Compression-Induced FN (Correct at Clean, becomes FN at lower QFs)
    # 4. Sporadic / Fluctuation
    never_fn = []
    always_fn = []
    clean_tp_qf20_fn = []
    clean_tp_qf10_fn = []

    for sid in fake_samples:
        statuses = [fn_status_by_sample[sid][c[0]] for c in conditions]
        if not any(statuses):
            never_fn.append(sid)
        elif all(statuses):
            always_fn.append(sid)
        
        if not fn_status_by_sample[sid]['Clean'] and fn_status_by_sample[sid]['QF20']:
            clean_tp_qf20_fn.append(sid)
        if not fn_status_by_sample[sid]['Clean'] and fn_status_by_sample[sid]['QF10']:
            clean_tp_qf10_fn.append(sid)

    print("\n--- SAMPLE-LEVEL FN TAXONOMY (M6-B16) ---")
    print(f"Never FN across all 10 conditions (Rock-Solid Detections): {len(never_fn)} / 1000 ({len(never_fn)/10.0:.1f}%)")
    print(f"Always FN across all 10 conditions (Intrinsic Hard FNs):   {len(always_fn)} / 1000 ({len(always_fn)/10.0:.1f}%)")
    print(f"Clean TP -> QF20 FN (Compression-Induced Failures):        {len(clean_tp_qf20_fn)} / 1000 ({len(clean_tp_qf20_fn)/10.0:.1f}%)")
    print(f"Clean TP -> QF10 FN (Extreme Compression Failures):        {len(clean_tp_qf10_fn)} / 1000 ({len(clean_tp_qf10_fn)/10.0:.1f}%)")

    # -------------------------------------------------------------
    # 2. Threshold Sensitivity Analysis for FN Recovery
    # -------------------------------------------------------------
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    thresh_eval_results = []

    print("\n--- THRESHOLD SENSITIVITY ANALYSIS (M6-B16 across all 5 standard compressed conditions QF80-QF20) ---")
    print(f"{'Thresh':<7} | {'Avg Rec':<8} | {'Avg Prec':<8} | {'Avg Acc':<8} | {'Total FN':<9} | {'Total FP':<9} | {'Total Err':<10} | {'Avg F2':<8}")
    print("-" * 85)

    comp_conds = ['QF80', 'QF60', 'QF50', 'QF40', 'QF20']

    for th in thresholds:
        tot_fn = 0
        tot_fp = 0
        tot_err = 0
        recs = []
        precs = []
        accs = []
        f2s = []

        for c in comp_conds:
            met = compute_metrics_at_threshold(all_m6_records[c], threshold=th)
            tot_fn += met['FN']
            tot_fp += met['FP']
            tot_err += met['total_errors']
            recs.append(met['recall'])
            precs.append(met['precision'])
            accs.append(met['accuracy'])
            f2s.append(met['f2'])

        avg_rec = float(np.mean(recs))
        avg_prec = float(np.mean(precs))
        avg_acc = float(np.mean(accs))
        avg_f2 = float(np.mean(f2s))

        thresh_eval_results.append({
            'threshold': th,
            'avg_recall': avg_rec,
            'avg_precision': avg_prec,
            'avg_accuracy': avg_acc,
            'avg_f2': avg_f2,
            'total_fn_5cond': tot_fn,
            'total_fp_5cond': tot_fp,
            'total_errors_5cond': tot_err
        })

        print(f"{th:<7.2f} | {avg_rec*100:6.2f}% | {avg_prec*100:6.2f}% | {avg_acc*100:6.2f}% | {tot_fn:>8d} | {tot_fp:>8d} | {tot_err:>9d} | {avg_f2:7.4f}")

    # -------------------------------------------------------------
    # 3. Save Summary CSV & JSON
    # -------------------------------------------------------------
    csv_path = os.path.join(out_dir, "m6_fn_deepdive_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(m6_summary[0].keys()))
        writer.writeheader()
        writer.writerows(m6_summary)
    print(f"\nSaved CSV metrics to {csv_path}")

    # Save M3 vs M6 comparison CSV
    m3_vs_m6_fn_csv = os.path.join(out_dir, "m3_vs_m6_fn_comparison.csv")
    with open(m3_vs_m6_fn_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['condition', 'qf', 
                      'm6_fn', 'm3_fn', 'fn_diff_m6_vs_m3',
                      'm6_fnr', 'm3_fnr', 'fnr_diff_pp',
                      'm6_fp', 'm3_fp',
                      'm6_recall', 'm3_recall', 'recall_diff_pp',
                      'm6_acc', 'm3_acc', 'acc_diff_pp',
                      'm6_errors', 'm3_errors', 'error_diff']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(m6_summary)):
            s6 = m6_summary[i]
            s3 = m3_summary[i]
            writer.writerow({
                'condition': s6['condition'],
                'qf': s6['qf'],
                'm6_fn': s6['FN'],
                'm3_fn': s3['FN'],
                'fn_diff_m6_vs_m3': s6['FN'] - s3['FN'],
                'm6_fnr': f"{s6['FNR']*100:.2f}%",
                'm3_fnr': f"{s3['FNR']*100:.2f}%",
                'fnr_diff_pp': f"{(s6['FNR'] - s3['FNR'])*100:+.2f}",
                'm6_fp': s6['FP'],
                'm3_fp': s3['FP'],
                'm6_recall': f"{s6['recall']*100:.2f}%",
                'm3_recall': f"{s3['recall']*100:.2f}%",
                'recall_diff_pp': f"{(s6['recall'] - s3['recall'])*100:+.2f}",
                'm6_acc': f"{s6['accuracy']*100:.2f}%",
                'm3_acc': f"{s3['accuracy']*100:.2f}%",
                'acc_diff_pp': f"{(s6['accuracy'] - s3['accuracy'])*100:+.2f}",
                'm6_errors': s6['total_errors'],
                'm3_errors': s3['total_errors'],
                'error_diff': s6['total_errors'] - s3['total_errors']
            })
    print(f"Saved M3 vs M6 comparison to {m3_vs_m6_fn_csv}")

    # Threshold results CSV
    th_csv_path = os.path.join(out_dir, "m6_threshold_tuning_fn_analysis.csv")
    with open(th_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(thresh_eval_results[0].keys()))
        writer.writeheader()
        writer.writerows(thresh_eval_results)
    print(f"Saved Threshold Analysis to {th_csv_path}")

    # Full JSON report
    report_json_path = os.path.join(out_dir, "m6_fn_deepdive_report.json")
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'm6_summary': m6_summary,
            'm3_summary': m3_summary,
            'threshold_analysis': thresh_eval_results,
            'taxonomy': {
                'total_fake_samples': len(fake_samples),
                'never_fn_count': len(never_fn),
                'always_fn_count': len(always_fn),
                'clean_tp_qf20_fn_count': len(clean_tp_qf20_fn),
                'clean_tp_qf10_fn_count': len(clean_tp_qf10_fn)
            }
        }, f, indent=2)
    print(f"Saved complete JSON report to {report_json_path}")

    # -------------------------------------------------------------
    # 4. Generate Visualizations
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: FN, FP, and Errors vs Quality Factor
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cond_labels = [s['condition'] for s in m6_summary]
    m6_fns = [s['FN'] for s in m6_summary]
    m6_fps = [s['FP'] for s in m6_summary]
    m3_fns = [s['FN'] for s in m3_summary]
    m3_fps = [s['FP'] for s in m3_summary]

    x = np.arange(len(cond_labels))
    width = 0.35

    # Left plot: FN Comparison M6 vs M3
    axes[0].plot(x, m6_fns, marker='o', linewidth=2.5, markersize=8, color='#1E88E5', label='M6-B16 (CoordAtt) FN')
    axes[0].plot(x, m3_fns, marker='s', linewidth=2.0, markersize=7, color='#E53935', linestyle='--', label='M3-B16 (Base) FN')
    for i, txt in enumerate(m6_fns):
        axes[0].annotate(f"{txt}", (x[i], m6_fns[i] + 2), fontsize=10, ha='center', color='#1E88E5', weight='bold')
    axes[0].set_title('False Negatives (Missed Deepfakes) Across Quality Factors', fontsize=13, weight='bold')
    axes[0].set_xlabel('Compression Condition', fontsize=11)
    axes[0].set_ylabel('Number of False Negatives (out of 1,000 Fakes)', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cond_labels)
    axes[0].set_ylim(0, max(max(m3_fns), max(m6_fns)) + 15)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=11)

    # Right plot: Error Composition (% FN vs % FP in total errors)
    m6_fn_pct = [s['fn_error_pct'] for s in m6_summary]
    m6_fp_pct = [s['fp_error_pct'] for s in m6_summary]

    axes[1].bar(x, m6_fn_pct, width=0.5, label='False Negatives (% of errors)', color='#FB8C00', alpha=0.85)
    axes[1].bar(x, m6_fp_pct, width=0.5, bottom=m6_fn_pct, label='False Positives (% of errors)', color='#43A047', alpha=0.85)
    for i in range(len(cond_labels)):
        axes[1].text(x[i], m6_fn_pct[i] / 2, f"{m6_fn_pct[i]:.1f}%", ha='center', va='center', color='white', weight='bold', fontsize=10)
    axes[1].set_title('M6-B16 Error Composition: FN vs. FP Ratio', fontsize=13, weight='bold')
    axes[1].set_xlabel('Compression Condition', fontsize=11)
    axes[1].set_ylabel('Percentage of Total Errors (%)', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cond_labels)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower left', fontsize=11)

    plt.tight_layout()
    fig1_path = os.path.join(viz_dir, "m6_fn_analysis_qf_curves.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig1_path}")

    # Figure 2: Probability Distribution of Fake Samples across Key QFs
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    selected_conds = ['Clean', 'QF80', 'QF60', 'QF50', 'QF20', 'QF10']

    for idx, cname in enumerate(selected_conds):
        ax = axes[idx]
        fake_probs = [r['prob'] for r in all_m6_records[cname] if r['target'] == 1.0]
        ax.hist(fake_probs, bins=40, range=(0, 1), color='#1976D2', alpha=0.75, edgecolor='black', linewidth=0.5)
        ax.axvline(0.50, color='red', linestyle='--', linewidth=2.0, label='Decision Boundary (0.50)')
        
        fn_count = sum(1 for p in fake_probs if p < 0.50)
        tp_count = sum(1 for p in fake_probs if p >= 0.50)
        med_p = np.median(fake_probs)

        ax.set_title(f"{cname} | FN: {fn_count} ({fn_count/10:.1f}%) | Median P: {med_p:.4f}", fontsize=12, weight='bold')
        ax.set_xlabel('Predicted Probability P(Fake)', fontsize=10)
        ax.set_ylabel('Sample Count', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=10)

    plt.suptitle("M6-B16 Coordinate Attention: Predicted Probability Distribution on 1,000 FAKE Test Samples\n(Left of red line = False Negatives / Missed Deepfakes)", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = os.path.join(viz_dir, "m6_fn_probability_distributions.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig2_path}")

    # Figure 3: Threshold Tuning Curve (FN vs FP trade-off)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    th_vals = [r['threshold'] for r in thresh_eval_results]
    th_fns = [r['total_fn_5cond'] for r in thresh_eval_results]
    th_fps = [r['total_fp_5cond'] for r in thresh_eval_results]
    th_recs = [r['avg_recall'] * 100 for r in thresh_eval_results]
    th_precs = [r['avg_precision'] * 100 for r in thresh_eval_results]

    ax1.plot(th_vals, th_fns, marker='o', color='#D32F2F', linewidth=2.5, label='Total FN (5 QFs combined)')
    ax1.plot(th_vals, th_fps, marker='s', color='#388E3C', linewidth=2.5, label='Total FP (5 QFs combined)')
    ax1.set_xlabel('Decision Threshold Tau', fontsize=12, weight='bold')
    ax1.set_ylabel('Total Error Count across 5,000 Test Inferences', fontsize=12, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(th_vals, th_recs, marker='^', color='#1976D2', linestyle='--', linewidth=2.0, label='Average Recall (%)')
    ax2.plot(th_vals, th_precs, marker='v', color='#7B1FA2', linestyle='--', linewidth=2.0, label='Average Precision (%)')
    ax2.set_ylabel('Rate (%)', fontsize=12, weight='bold')
    ax2.set_ylim(92, 100)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=11)

    plt.title('M6-B16 Threshold Sensitivity Analysis: FN Recovery vs. False Alarm Trade-off', fontsize=13, weight='bold')
    plt.tight_layout()
    fig3_path = os.path.join(viz_dir, "m6_fn_threshold_tradeoff.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig3_path}")

    print("\n" + "=" * 90)
    print("FALSE NEGATIVE EVALUATION & VISUALIZATION COMPLETE!")
    print("=" * 90)

if __name__ == '__main__':
    main()
