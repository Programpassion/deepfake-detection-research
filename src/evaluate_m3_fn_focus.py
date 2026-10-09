"""
evaluate_m3_fn_focus.py
Deep-Dive Evaluation of M3-B16 (Baseline EfficientNet-B3) Across Quality Factors
and Decision Thresholds with Strict Focus on False Negatives (FN) Behavior.
Generates Direct Head-to-Head Comparisons with M6-B16 (Coordinate Attention).
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
from torch.utils.data import DataLoader
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

    fake_probs = probs[targets == 1.0]
    real_probs = probs[targets == 0.0]

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
    print("FALSE NEGATIVE (FN) THRESHOLD ANALYSIS: M3-B16 BASELINE VS. M6-B16")
    print(f"Device: {device} | Autocast: {device.type == 'cuda'}")
    print("=" * 90)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(repo_dir, "data", "test.csv")
    m3_path = os.path.join(repo_dir, "models", "efficientnet_b3_baseline_m3_16_best.pth")
    m6_path = os.path.join(repo_dir, "models", "efficientnet_b3_coordatt_compression_aware_m6_16_best.pth")
    out_dir = os.path.join(repo_dir, "results", "fn_analysis")
    viz_dir = os.path.join(repo_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    _, eval_tx = get_phase2_transforms()

    # Load Models
    print(f"Loading M3-B16 (Baseline) from {os.path.basename(m3_path)}...")
    m3 = EfficientNetB3Baseline(pretrained=False)
    m3.load_state_dict(torch.load(m3_path, map_location=device)['model_state_dict'])
    m3.to(device)

    print(f"Loading M6-B16 (CoordAtt) from {os.path.basename(m6_path)}...")
    m6 = EfficientNetB3CoordAtt(pretrained=False)
    m6.load_state_dict(torch.load(m6_path, map_location=device)['model_state_dict'])
    m6.to(device)

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

    all_m3_records = {}
    all_m6_records = {}
    m3_summary = []
    m6_summary = []

    print("\nRunning multi-condition inference across 10 Quality Factors...")
    for cond_name, qf in conditions:
        ds = Phase2DeepFakeDataset(test_csv, condition=cond_name.lower(), qf=qf, transform=eval_tx)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        # M3 Inference
        m3_recs = run_condition_inference(m3, loader, device)
        all_m3_records[cond_name] = m3_recs
        m3_met = compute_metrics_at_threshold(m3_recs, threshold=0.50)
        m3_met['condition'] = cond_name
        m3_met['qf'] = qf if qf is not None else 100
        m3_summary.append(m3_met)

        # M6 Inference
        m6_recs = run_condition_inference(m6, loader, device)
        all_m6_records[cond_name] = m6_recs
        m6_met = compute_metrics_at_threshold(m6_recs, threshold=0.50)
        m6_met['condition'] = cond_name
        m6_met['qf'] = qf if qf is not None else 100
        m6_summary.append(m6_met)

    # -------------------------------------------------------------
    # 1. Sample-level FN Taxonomy on 1,000 Fake test samples (M3 vs M6)
    # -------------------------------------------------------------
    fake_samples = [r['image_id'] for r in all_m3_records['Clean'] if r['target'] == 1.0]

    def get_taxonomy(all_records):
        never_fn = []
        always_fn = []
        clean_tp_qf20_fn = []
        clean_tp_qf10_fn = []
        for sid in fake_samples:
            statuses = [r['prob'] < 0.50 for c in conditions for r in all_records[c[0]] if r['image_id'] == sid]
            clean_fn = [r['prob'] < 0.50 for r in all_records['Clean'] if r['image_id'] == sid][0]
            qf20_fn = [r['prob'] < 0.50 for r in all_records['QF20'] if r['image_id'] == sid][0]
            qf10_fn = [r['prob'] < 0.50 for r in all_records['QF10'] if r['image_id'] == sid][0]

            if not any(statuses):
                never_fn.append(sid)
            elif all(statuses):
                always_fn.append(sid)
            if not clean_fn and qf20_fn:
                clean_tp_qf20_fn.append(sid)
            if not clean_fn and qf10_fn:
                clean_tp_qf10_fn.append(sid)
        return never_fn, always_fn, clean_tp_qf20_fn, clean_tp_qf10_fn

    m3_never, m3_always, m3_c_qf20, m3_c_qf10 = get_taxonomy(all_m3_records)
    m6_never, m6_always, m6_c_qf20, m6_c_qf10 = get_taxonomy(all_m6_records)

    print("\n--- SAMPLE-LEVEL FN TAXONOMY: M3 BASELINE VS. M6 COORDINATE ATTENTION ---")
    print(f"{'Category':<45} | {'M3 Baseline':<15} | {'M6 CoordAtt':<15} | {'Advantage':<15}")
    print("-" * 95)
    print(f"{'Never FN (Rock-Solid Detections)':<45} | {len(m3_never):>4d} ({(len(m3_never)/10):.1f}%)    | {len(m6_never):>4d} ({(len(m6_never)/10):.1f}%)    | M6 +{len(m6_never)-len(m3_never)} samples")
    print(f"{'Always FN (Intrinsic Hard Misses)':<45} | {len(m3_always):>4d} ({(len(m3_always)/10):.1f}%)    | {len(m6_always):>4d} ({(len(m6_always)/10):.1f}%)    | M6 -{len(m3_always)-len(m6_always)} misses")
    print(f"{'Clean TP -> QF20 FN (Compression Failures)':<45} | {len(m3_c_qf20):>4d} ({(len(m3_c_qf20)/10):.1f}%)    | {len(m6_c_qf20):>4d} ({(len(m6_c_qf20)/10):.1f}%)    | M6 -{len(m3_c_qf20)-len(m6_c_qf20)} failures")
    print(f"{'Clean TP -> QF10 FN (Extreme Failures)':<45} | {len(m3_c_qf10):>4d} ({(len(m3_c_qf10)/10):.1f}%)    | {len(m6_c_qf10):>4d} ({(len(m6_c_qf10)/10):.1f}%)    | M6 -{len(m3_c_qf10)-len(m6_c_qf10)} failures")

    # -------------------------------------------------------------
    # 2. Detailed Threshold Sensitivity Analysis: M3 vs. M6
    # -------------------------------------------------------------
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    comp_conds = ['QF80', 'QF60', 'QF50', 'QF40', 'QF20']

    m3_thresh_results = []
    m6_thresh_results = []
    comparison_thresh_rows = []

    print("\n--- THRESHOLD SENSITIVITY COMPARISON (Combined across QF80, QF60, QF50, QF40, QF20; 5,000 Inferences) ---")
    print(f"{'Thresh':<7} | {'M3 FN':<6} | {'M6 FN':<6} | {'FN Diff':<8} | {'M3 FP':<6} | {'M6 FP':<6} | {'M3 Rec':<7} | {'M6 Rec':<7} | {'M3 Err':<7} | {'M6 Err':<7} | {'M3 F2':<7} | {'M6 F2':<7}")
    print("-" * 105)

    for th in thresholds:
        # M3
        m3_tot_fn = 0
        m3_tot_fp = 0
        m3_tot_err = 0
        m3_recs = []
        m3_precs = []
        m3_accs = []
        m3_f2s = []
        for c in comp_conds:
            met = compute_metrics_at_threshold(all_m3_records[c], threshold=th)
            m3_tot_fn += met['FN']
            m3_tot_fp += met['FP']
            m3_tot_err += met['total_errors']
            m3_recs.append(met['recall'])
            m3_precs.append(met['precision'])
            m3_accs.append(met['accuracy'])
            m3_f2s.append(met['f2'])

        m3_res = {
            'threshold': th,
            'avg_recall': float(np.mean(m3_recs)),
            'avg_precision': float(np.mean(m3_precs)),
            'avg_accuracy': float(np.mean(m3_accs)),
            'avg_f2': float(np.mean(m3_f2s)),
            'total_fn_5cond': m3_tot_fn,
            'total_fp_5cond': m3_tot_fp,
            'total_errors_5cond': m3_tot_err
        }
        m3_thresh_results.append(m3_res)

        # M6
        m6_tot_fn = 0
        m6_tot_fp = 0
        m6_tot_err = 0
        m6_recs = []
        m6_precs = []
        m6_accs = []
        m6_f2s = []
        for c in comp_conds:
            met = compute_metrics_at_threshold(all_m6_records[c], threshold=th)
            m6_tot_fn += met['FN']
            m6_tot_fp += met['FP']
            m6_tot_err += met['total_errors']
            m6_recs.append(met['recall'])
            m6_precs.append(met['precision'])
            m6_accs.append(met['accuracy'])
            m6_f2s.append(met['f2'])

        m6_res = {
            'threshold': th,
            'avg_recall': float(np.mean(m6_recs)),
            'avg_precision': float(np.mean(m6_precs)),
            'avg_accuracy': float(np.mean(m6_accs)),
            'avg_f2': float(np.mean(m6_f2s)),
            'total_fn_5cond': m6_tot_fn,
            'total_fp_5cond': m6_tot_fp,
            'total_errors_5cond': m6_tot_err
        }
        m6_thresh_results.append(m6_res)

        fn_diff = m6_tot_fn - m3_tot_fn
        print(f"{th:<7.2f} | {m3_tot_fn:>5d} | {m6_tot_fn:>5d} | {fn_diff:>+7d} | {m3_tot_fp:>5d} | {m6_tot_fp:>5d} | {m3_res['avg_recall']*100:5.2f}% | {m6_res['avg_recall']*100:5.2f}% | {m3_tot_err:>6d} | {m6_tot_err:>6d} | {m3_res['avg_f2']:6.4f} | {m6_res['avg_f2']:6.4f}")

        comparison_thresh_rows.append({
            'threshold': th,
            'm3_total_fn': m3_tot_fn,
            'm6_total_fn': m6_tot_fn,
            'fn_reduction_m6_vs_m3': m3_tot_fn - m6_tot_fn,
            'm3_total_fp': m3_tot_fp,
            'm6_total_fp': m6_tot_fp,
            'fp_reduction_m6_vs_m3': m3_tot_fp - m6_tot_fp,
            'm3_avg_recall': f"{m3_res['avg_recall']*100:.2f}%",
            'm6_avg_recall': f"{m6_res['avg_recall']*100:.2f}%",
            'm3_avg_precision': f"{m3_res['avg_precision']*100:.2f}%",
            'm6_avg_precision': f"{m6_res['avg_precision']*100:.2f}%",
            'm3_avg_accuracy': f"{m3_res['avg_accuracy']*100:.2f}%",
            'm6_avg_accuracy': f"{m6_res['avg_accuracy']*100:.2f}%",
            'm3_total_errors': m3_tot_err,
            'm6_total_errors': m6_tot_err,
            'error_reduction_m6_vs_m3': m3_tot_err - m6_tot_err,
            'm3_avg_f2': m3_res['avg_f2'],
            'm6_avg_f2': m6_res['avg_f2']
        })

    # -------------------------------------------------------------
    # 3. Save CSVs and JSON Reports
    # -------------------------------------------------------------
    m3_th_csv = os.path.join(out_dir, "m3_threshold_tuning_fn_analysis.csv")
    with open(m3_th_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(m3_thresh_results[0].keys()))
        writer.writeheader()
        writer.writerows(m3_thresh_results)
    print(f"\nSaved M3 Threshold Analysis to {m3_th_csv}")

    comp_th_csv = os.path.join(out_dir, "m3_vs_m6_threshold_comparison.csv")
    with open(comp_th_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_thresh_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_thresh_rows)
    print(f"Saved Head-to-Head Threshold Comparison to {comp_th_csv}")

    m3_summary_csv = os.path.join(out_dir, "m3_fn_deepdive_summary.csv")
    with open(m3_summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(m3_summary[0].keys()))
        writer.writeheader()
        writer.writerows(m3_summary)
    print(f"Saved M3 Condition Summary to {m3_summary_csv}")

    full_report = {
        'm3_summary': m3_summary,
        'm3_threshold_analysis': m3_thresh_results,
        'head_to_head_threshold': comparison_thresh_rows,
        'taxonomy': {
            'total_fakes': len(fake_samples),
            'm3_never_fn': len(m3_never),
            'm6_never_fn': len(m6_never),
            'm3_always_fn': len(m3_always),
            'm6_always_fn': len(m6_always),
            'm3_clean_tp_qf20_fn': len(m3_c_qf20),
            'm6_clean_tp_qf20_fn': len(m6_c_qf20),
            'm3_clean_tp_qf10_fn': len(m3_c_qf10),
            'm6_clean_tp_qf10_fn': len(m6_c_qf10)
        }
    }
    report_json_path = os.path.join(out_dir, "m3_vs_m6_fn_threshold_report.json")
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved Full JSON Report to {report_json_path}")

    # -------------------------------------------------------------
    # 4. Generate Publication-Quality Visualizations
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Head-to-Head Threshold Comparison: M3 vs M6 (FN & FP Curves)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    th_vals = [r['threshold'] for r in m3_thresh_results]
    m3_fns = [r['total_fn_5cond'] for r in m3_thresh_results]
    m6_fns = [r['total_fn_5cond'] for r in m6_thresh_results]
    m3_fps = [r['total_fp_5cond'] for r in m3_thresh_results]
    m6_fps = [r['total_fp_5cond'] for r in m6_thresh_results]

    # Subplot 1: Total FN vs Threshold
    axes[0].plot(th_vals, m3_fns, marker='s', linewidth=2.5, markersize=8, color='#E53935', linestyle='--', label='M3-B16 (Baseline) FN')
    axes[0].plot(th_vals, m6_fns, marker='o', linewidth=2.5, markersize=8, color='#1E88E5', label='M6-B16 (CoordAtt) FN')
    for i in range(len(th_vals)):
        axes[0].annotate(f"{m6_fns[i]}", (th_vals[i], m6_fns[i] - 7), fontsize=9, ha='center', color='#1E88E5', weight='bold')
        axes[0].annotate(f"{m3_fns[i]}", (th_vals[i], m3_fns[i] + 4), fontsize=9, ha='center', color='#E53935')
    axes[0].set_title('False Negatives vs. Decision Threshold (5 QFs Combined)', fontsize=13, weight='bold')
    axes[0].set_xlabel('Decision Threshold Tau', fontsize=11, weight='bold')
    axes[0].set_ylabel('Total Missed Deepfakes (out of 5,000 Inferences)', fontsize=11, weight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=11)

    # Subplot 2: Total FP vs Threshold (False Alarms)
    axes[1].plot(th_vals, m3_fps, marker='s', linewidth=2.5, markersize=8, color='#FB8C00', linestyle='--', label='M3-B16 (Baseline) FP')
    axes[1].plot(th_vals, m6_fps, marker='o', linewidth=2.5, markersize=8, color='#43A047', label='M6-B16 (CoordAtt) FP')
    for i in range(len(th_vals)):
        axes[1].annotate(f"{m6_fps[i]}", (th_vals[i], m6_fps[i] - 12), fontsize=9, ha='center', color='#43A047', weight='bold')
        axes[1].annotate(f"{m3_fps[i]}", (th_vals[i], m3_fps[i] + 7), fontsize=9, ha='center', color='#FB8C00')
    axes[1].set_title('False Positives vs. Decision Threshold (5 QFs Combined)', fontsize=13, weight='bold')
    axes[1].set_xlabel('Decision Threshold Tau', fontsize=11, weight='bold')
    axes[1].set_ylabel('Total False Alarms (out of 5,000 Inferences)', fontsize=11, weight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    fig1_path = os.path.join(viz_dir, "m3_vs_m6_threshold_fn_fp_tradeoff.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Figure: {fig1_path}")

    # Figure 2: M3 Predicted Probability Distribution on 1,000 Fake Samples Across Key QFs
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    selected_conds = ['Clean', 'QF80', 'QF60', 'QF50', 'QF20', 'QF10']

    for idx, cname in enumerate(selected_conds):
        ax = axes[idx]
        fake_probs = [r['prob'] for r in all_m3_records[cname] if r['target'] == 1.0]
        ax.hist(fake_probs, bins=40, range=(0, 1), color='#E53935', alpha=0.75, edgecolor='black', linewidth=0.5)
        ax.axvline(0.50, color='blue', linestyle='--', linewidth=2.0, label='Decision Boundary (0.50)')

        fn_count = sum(1 for p in fake_probs if p < 0.50)
        med_p = np.median(fake_probs)

        ax.set_title(f"M3 {cname} | FN: {fn_count} ({fn_count/10:.1f}%) | Median P: {med_p:.4f}", fontsize=12, weight='bold')
        ax.set_xlabel('Predicted Probability P(Fake)', fontsize=10)
        ax.set_ylabel('Sample Count', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=10)

    plt.suptitle("M3-B16 Baseline EfficientNet-B3: Predicted Probability Distribution on 1,000 FAKE Test Samples\n(Left of blue line = False Negatives / Missed Deepfakes)", fontsize=14, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = os.path.join(viz_dir, "m3_fn_probability_distributions.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved Figure: {fig2_path}")

    print("\n" + "=" * 90)
    print("M3 THRESHOLD & FN EVALUATION COMPLETE!")
    print("=" * 90)

if __name__ == '__main__':
    main()
