import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

base_dir = r"C:\Users\hdutt\Desktop\compressed images deepfake detection research"
results_dir = os.path.join(base_dir, "results")
viz_dir = os.path.join(base_dir, "visualizations", "m3_vs_m6_comparison")
os.makedirs(viz_dir, exist_ok=True)

conditions = ['Clean', 'QF80', 'QF60', 'QF50', 'QF40', 'QF20']
qf_labels = ['Clean\n(None)', 'QF80\n(Mild)', 'QF60\n(Moderate)', 'QF50\n(Medium)', 'QF40\n(Heavy)', 'QF20\n(Severe)']

m3_data = []
m6_data = []

for c in conditions:
    with open(os.path.join(results_dir, f"m3_16_{c.lower()}_metrics.json")) as f:
        m3_data.append(json.load(f))
    with open(os.path.join(results_dir, f"m6_16_{c.lower()}_metrics.json")) as f:
        m6_data.append(json.load(f))

# 1. Accuracy vs. Compression Degradation Curve
plt.figure(figsize=(9, 5.5), dpi=300)
x = np.arange(len(conditions))
m3_acc = [d['accuracy'] * 100 for d in m3_data]
m6_acc = [d['accuracy'] * 100 for d in m6_data]

plt.plot(x, m3_acc, marker='o', linewidth=2.5, markersize=8, color='#d9534f', label='M3-B16 (Base EfficientNet-B3)')
plt.plot(x, m6_acc, marker='s', linewidth=2.5, markersize=8, color='#0275d8', label='M6-B16 (Proposed Coord-EfficientNet-B3)')

for i in range(len(conditions)):
    plt.annotate(f"{m3_acc[i]:.2f}%", (x[i], m3_acc[i]), textcoords="offset points", xytext=(0, -18), ha='center', fontsize=9, color='#d9534f', fontweight='bold')
    plt.annotate(f"{m6_acc[i]:.2f}%", (x[i], m6_acc[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#0275d8', fontweight='bold')

plt.title('DeepFake Detection Accuracy Under JPEG Compression Degradation\nM3-B16 (Base) vs. M6-B16 (Proposed Coordinate Attention)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Evaluation Regime (Compression Quality Factor)', fontsize=11, fontweight='bold')
plt.ylabel('Classification Accuracy (%)', fontsize=11, fontweight='bold')
plt.xticks(x, qf_labels, fontsize=10)
plt.ylim(93.5, 98.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='lower left', frameon=True)
plt.tight_layout()
p1 = os.path.join(viz_dir, "m3_vs_m6_accuracy_vs_qf.png")
plt.savefig(p1)
plt.close()

# 2. FP and FN Breakdown Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
bar_width = 0.35

m3_fp = [d['FP'] for d in m3_data]
m6_fp = [d['FP'] for d in m6_data]
m3_fn = [d['FN'] for d in m3_data]
m6_fn = [d['FN'] for d in m6_data]

# FP Subplot
ax1.bar(x - bar_width/2, m3_fp, bar_width, label='M3-B16 (Base)', color='#f0ad4e', edgecolor='black', alpha=0.85)
ax1.bar(x + bar_width/2, m6_fp, bar_width, label='M6-B16 (Proposed)', color='#5cb85c', edgecolor='black', alpha=0.85)
ax1.set_title('False Positives (False Alarms on Real Faces)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Condition', fontsize=10, fontweight='bold')
ax1.set_ylabel('False Positive Count (lower is better)', fontsize=10, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(qf_labels, fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
ax1.legend(fontsize=10)
for i in range(len(x)):
    ax1.text(x[i] - bar_width/2, m3_fp[i] + 0.8, str(m3_fp[i]), ha='center', fontsize=9, fontweight='bold')
    ax1.text(x[i] + bar_width/2, m6_fp[i] + 0.8, str(m6_fp[i]), ha='center', fontsize=9, fontweight='bold')

# FN Subplot
ax2.bar(x - bar_width/2, m3_fn, bar_width, label='M3-B16 (Base)', color='#d9534f', edgecolor='black', alpha=0.85)
ax2.bar(x + bar_width/2, m6_fn, bar_width, label='M6-B16 (Proposed)', color='#0275d8', edgecolor='black', alpha=0.85)
ax2.set_title('False Negatives (Missed Fakes under Compression)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Condition', fontsize=10, fontweight='bold')
ax2.set_ylabel('False Negative Count (lower is better)', fontsize=10, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(qf_labels, fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
ax2.legend(fontsize=10)
for i in range(len(x)):
    ax2.text(x[i] - bar_width/2, m3_fn[i] + 1.0, str(m3_fn[i]), ha='center', fontsize=9, fontweight='bold')
    ax2.text(x[i] + bar_width/2, m6_fn[i] + 1.0, str(m6_fn[i]), ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Error Analysis: False Positives & False Negatives (M3-B16 vs. M6-B16)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
p2 = os.path.join(viz_dir, "m3_vs_m6_fp_fn_breakdown.png")
plt.savefig(p2)
plt.close()

# 3. Master Dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

# (A) Accuracy
axes[0, 0].plot(x, m3_acc, marker='o', linewidth=2.5, color='#d9534f', label='M3 Base EfficientNet-B3')
axes[0, 0].plot(x, m6_acc, marker='s', linewidth=2.5, color='#0275d8', label='M6 Proposed CoordAtt')
axes[0, 0].set_title('A. Classification Accuracy vs. Compression', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(qf_labels, fontsize=9)
axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)
axes[0, 0].legend()

# (B) MCC
m3_mcc = [d['mcc'] for d in m3_data]
m6_mcc = [d['mcc'] for d in m6_data]
axes[0, 1].plot(x, m3_mcc, marker='o', linewidth=2.5, color='#d9534f', label='M3 Base EfficientNet-B3')
axes[0, 1].plot(x, m6_mcc, marker='s', linewidth=2.5, color='#0275d8', label='M6 Proposed CoordAtt')
axes[0, 1].set_title('B. Matthews Correlation Coefficient (MCC)', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(qf_labels, fontsize=9)
axes[0, 1].set_ylabel('MCC Score', fontweight='bold')
axes[0, 1].grid(True, linestyle='--', alpha=0.6)
axes[0, 1].legend()

# (C) Total Errors
m3_tot = [d['FP'] + d['FN'] for d in m3_data]
m6_tot = [d['FP'] + d['FN'] for d in m6_data]
axes[1, 0].bar(x - bar_width/2, m3_tot, bar_width, color='#d9534f', label='M3 Base Errors', alpha=0.85, edgecolor='black')
axes[1, 0].bar(x + bar_width/2, m6_tot, bar_width, color='#0275d8', label='M6 Proposed Errors', alpha=0.85, edgecolor='black')
axes[1, 0].set_title('C. Total Error Count (FP + FN)', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(qf_labels, fontsize=9)
axes[1, 0].set_ylabel('Total Errors per 2,000 images', fontweight='bold')
axes[1, 0].grid(True, linestyle='--', alpha=0.6, axis='y')
axes[1, 0].legend()
for i in range(len(x)):
    axes[1, 0].text(x[i] - bar_width/2, m3_tot[i] + 1.2, str(m3_tot[i]), ha='center', fontsize=8.5, fontweight='bold')
    axes[1, 0].text(x[i] + bar_width/2, m6_tot[i] + 1.2, str(m6_tot[i]), ha='center', fontsize=8.5, fontweight='bold')

# (D) Compressed Cumulative Summary
comp_m3_err = sum(m3_tot[1:])
comp_m6_err = sum(m6_tot[1:])
comp_m3_fp = sum(m3_fp[1:])
comp_m6_fp = sum(m6_fp[1:])
comp_m3_fn = sum(m3_fn[1:])
comp_m6_fn = sum(m6_fn[1:])

categories = ['Total Errors', 'False Positives', 'False Negatives']
m3_cum = [comp_m3_err, comp_m3_fp, comp_m3_fn]
m6_cum = [comp_m6_err, comp_m6_fp, comp_m6_fn]
cx = np.arange(len(categories))

axes[1, 1].bar(cx - bar_width/2, m3_cum, bar_width, color='#d9534f', label='M3 Base (10,000 Inferences)', alpha=0.85, edgecolor='black')
axes[1, 1].bar(cx + bar_width/2, m6_cum, bar_width, color='#0275d8', label='M6 Proposed (10,000 Inferences)', alpha=0.85, edgecolor='black')
axes[1, 1].set_title('D. Aggregate Compressed Errors (QF80 - QF20)', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(cx)
axes[1, 1].set_xticklabels(categories, fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('Cumulative Count', fontweight='bold')
axes[1, 1].grid(True, linestyle='--', alpha=0.6, axis='y')
axes[1, 1].legend()
for i in range(len(cx)):
    axes[1, 1].text(cx[i] - bar_width/2, m3_cum[i] + 4, str(m3_cum[i]), ha='center', fontsize=9.5, fontweight='bold')
    axes[1, 1].text(cx[i] + bar_width/2, m6_cum[i] + 4, str(m6_cum[i]), ha='center', fontsize=9.5, fontweight='bold')

plt.suptitle('Research Paper Master Benchmark: M3-B16 (Base) vs. M6-B16 (Coordinate Attention)', fontsize=15, fontweight='bold', y=0.99)
plt.tight_layout()
p3 = os.path.join(viz_dir, "m3_vs_m6_master_dashboard.png")
plt.savefig(p3)
plt.close()

# 4. Generate Master Comparison CSV
csv_path = os.path.join(results_dir, "m3_vs_m6_b16_master_comparison.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Condition', 'JPEG_QF', 
        'M3_Accuracy', 'M6_Accuracy', 'Delta_Accuracy',
        'M3_Precision', 'M6_Precision', 'Delta_Precision',
        'M3_Recall', 'M6_Recall', 'Delta_Recall',
        'M3_F1', 'M6_F1', 'Delta_F1',
        'M3_MCC', 'M6_MCC', 'Delta_MCC',
        'M3_FP', 'M6_FP', 'Delta_FP',
        'M3_FN', 'M6_FN', 'Delta_FN',
        'M3_Errors', 'M6_Errors', 'Delta_Errors'
    ])
    for i, c in enumerate(conditions):
        m3_c = m3_data[i]
        m6_c = m6_data[i]
        qf = m3_c['jpeg_quality']
        acc3, acc6 = m3_c['accuracy']*100, m6_c['accuracy']*100
        prec3, prec6 = m3_c['precision']*100, m6_c['precision']*100
        rec3, rec6 = m3_c['recall']*100, m6_c['recall']*100
        f1_3, f1_6 = m3_c['f1_score']*100, m6_c['f1_score']*100
        mcc3, mcc6 = m3_c['mcc'], m6_c['mcc']
        fp3, fp6 = m3_c['FP'], m6_c['FP']
        fn3, fn6 = m3_c['FN'], m6_c['FN']
        err3 = m3_c['FP'] + m3_c['FN']
        err6 = m6_c['FP'] + m6_c['FN']
        
        writer.writerow([
            c, qf,
            f"{acc3:.2f}", f"{acc6:.2f}", f"{acc6 - acc3:+.2f}",
            f"{prec3:.2f}", f"{prec6:.2f}", f"{prec6 - prec3:+.2f}",
            f"{rec3:.2f}", f"{rec6:.2f}", f"{rec6 - rec3:+.2f}",
            f"{f1_3:.2f}", f"{f1_6:.2f}", f"{f1_6 - f1_3:+.2f}",
            f"{mcc3:.4f}", f"{m6_c['mcc']:.4f}", f"{mcc6 - mcc3:+.4f}",
            fp3, fp6, fp6 - fp3,
            fn3, fn6, fn6 - fn3,
            err3, err6, err6 - err3
        ])

print("Generated high-res comparative figures and master CSV successfully!")
