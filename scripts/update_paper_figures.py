"""
update_paper_figures.py
1. Generates experimental_workflow.png as an authentic IEEE-style engineering flowchart
   using standard flowchart entities (ovals, rectangles, diamonds, parallelograms, flowlines).
   Explicitly maps to Table I, Table II, and Table III.
2. Re-generates accuracy_vs_qf.png and error_breakdown.png with M3-B16 and M6-B16 completely removed.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

def generate_flowchart(output_path):
    # Dimensions for vertical layout similar to Figure 2 from the IEEE paper reference
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')

    # Color scheme inspired by IEEE paper (gold/amber tones and clean borders)
    c_start_end = '#FDEBD0'      # soft peach / gold
    c_process   = '#FAD7A0'      # amber / orange
    c_data      = '#EDBB99'      # pale bronze
    c_decision  = '#F8C471'      # gold decision
    c_table     = '#FCF3CF'      # light yellow for tables
    c_border    = '#B9770E'      # deep amber border
    c_text      = '#1A1A1A'
    c_line      = '#1A1A1A'

    def draw_oval(x, y, w, h, text, subtext=""):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={h/2}",
                             facecolor=c_start_end, edgecolor=c_border, linewidth=1.8)
        ax.add_patch(box)
        if subtext:
            ax.text(x + w/2, y + h*0.62, text, ha='center', va='center', fontsize=9, fontweight='bold', color=c_text)
            ax.text(x + w/2, y + h*0.30, subtext, ha='center', va='center', fontsize=7.5, color='#333333')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', color=c_text)

    def draw_rect(x, y, w, h, text, subtext=""):
        box = patches.Rectangle((x, y), w, h, facecolor=c_process, edgecolor=c_border, linewidth=1.6)
        ax.add_patch(box)
        if subtext:
            ax.text(x + w/2, y + h*0.64, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_text)
            ax.text(x + w/2, y + h*0.30, subtext, ha='center', va='center', fontsize=7.2, color='#2C3E50')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_text)

    def draw_parallelogram(x, y, w, h, text, subtext=""):
        skew = 3.5
        pts = np.array([[x + skew, y], [x + w, y], [x + w - skew, y + h], [x, y + h]])
        poly = Polygon(pts, closed=True, facecolor=c_data, edgecolor=c_border, linewidth=1.6)
        ax.add_patch(poly)
        if subtext:
            ax.text(x + w/2, y + h*0.64, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_text)
            ax.text(x + w/2, y + h*0.30, subtext, ha='center', va='center', fontsize=7.2, color='#2C3E50')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_text)

    def draw_diamond(x, y, w, h, text, subtext=""):
        pts = np.array([[x + w/2, y + h], [x + w, y + h/2], [x + w/2, y], [x, y + h/2]])
        poly = Polygon(pts, closed=True, facecolor=c_decision, edgecolor=c_border, linewidth=1.8)
        ax.add_patch(poly)
        if subtext:
            ax.text(x + w/2, y + h*0.60, text, ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_text)
            ax.text(x + w/2, y + h*0.35, subtext, ha='center', va='center', fontsize=7, color='#2C3E50')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_text)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=c_line, lw=1.5, mutation_scale=11))
        if label:
            ax.text((x1+x2)/2 + 1.2, (y1+y2)/2, label, ha='left', va='center', fontsize=7.5, fontweight='bold', color='#7D6608')

    # Top to bottom coordinates
    # 1. Start (Oval)
    draw_oval(30, 131, 40, 6.5, "Start: Experimental Pipeline", "DeepFake Detection under Lossy Compression")
    draw_arrow(50, 131, 50, 124.5)

    # 2. Data Input (Parallelogram)
    draw_parallelogram(18, 118, 64, 6.5, "Dataset Ingestion (140,000 Faces)", "Real Faces: FFHQ  |  Synthetic Faces: StyleGAN")
    draw_arrow(50, 118, 50, 111.5)

    # 3. Deduplication and Partitioning (Rectangle)
    draw_rect(15, 104, 70, 7.5, "Data Hygiene & Zero-Leak Partitioning", "SHA-256 / dHash Verified Deduplication\nTrain: 7,200 | Val: 800 (Clean) | Test: 2,000 | Holdout: 200")
    draw_arrow(50, 104, 50, 97.5)

    # 4. Model Training Process (Rectangle)
    draw_rect(12, 89, 76, 8.5, "Stochastic Compression-Aware Training", "Models: Baseline (EfficientNet-B3) & Proposed (Coord-EfficientNet-B3)\np_jpeg = 0.50, QF in {20, 40, 50, 60, 80}, Adam (lr=1e-4), BCE Loss")
    draw_arrow(50, 89, 50, 82)

    # 5. Decision 1: Benchmark Regime Branching (Diamond)
    draw_diamond(30, 73, 40, 9, "Benchmark", "Regime?")
    
    # Left Branch: Multi-QF Benchmark
    draw_arrow(30, 77.5, 12, 77.5, "")
    draw_arrow(12, 77.5, 12, 69, "2,000 Test Set")
    draw_rect(3, 61, 38, 8, "Multi-QF Testing (Table I)", "6 Regimes: Clean, QF80, QF60, QF50, QF40, QF20\n12,000 Total Inference Runs")

    # Right Branch: Unseen Holdout Benchmark
    draw_arrow(70, 77.5, 88, 77.5, "")
    draw_arrow(88, 77.5, 88, 69, "200 Holdout")
    draw_rect(59, 61, 38, 8, "Unseen Holdout Testing (Table II)", "Pure Facial Crops (100 Real, 100 Fake)\n1,200 Inferences Across 6 Regimes")

    # Merge branches into Probability Inference
    draw_arrow(22, 61, 22, 54)
    draw_arrow(78, 61, 78, 54)
    ax.plot([22, 78], [54, 54], color=c_line, lw=1.5)
    draw_arrow(50, 54, 50, 48.5)

    # 6. Inference Computation (Rectangle)
    draw_rect(20, 42, 60, 6.5, "Model Inference & Posterior Probability", "Compute P(Fake | X) = sigma(z) for Both Models")
    draw_arrow(50, 42, 50, 36.5)

    # 7. Decision 2: Decision Thresholding (Diamond)
    draw_diamond(30, 27.5, 40, 9, "Classification Rule", "P(Fake | X) >= tau?")
    
    draw_arrow(70, 32, 85, 32)
    ax.text(75, 33.2, "Yes", fontsize=7.5, fontweight='bold', color='#B9770E')
    ax.text(86, 32, "Fake (1)", fontsize=8, fontweight='bold', va='center', color='#900C3F')

    draw_arrow(30, 32, 15, 32)
    ax.text(22, 33.2, "No", fontsize=7.5, fontweight='bold', color='#B9770E')
    ax.text(6, 32, "Real (0)", fontsize=8, fontweight='bold', va='center', color='#196F3D')

    draw_arrow(50, 27.5, 50, 20.5, "")

    # 8. Output Benchmark Artifacts (Parallelogram)
    draw_parallelogram(8, 12.5, 84, 8, "Statistical Validation & Tabular Results Generation", "Table I: Multi-QF Benchmark  |  Table II: Unseen Holdout (-50% FN)\nTable III: Threshold Optimization (tau in [0.15, 0.50]) | McNemar (p < 0.01)")
    draw_arrow(50, 12.5, 50, 6.5)

    # 9. End (Oval)
    draw_oval(32, 1, 36, 5.5, "End: Robust Detection Validated", "14.4% Error Reduction Confirmed")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated clean flowchart at {output_path}")

def generate_cleaned_accuracy_curve(output_path):
    conditions = ['Clean', 'QF80', 'QF60', 'QF50', 'QF40', 'QF20']
    qf_labels = ['Clean\n(None)', 'QF80\n(Mild)', 'QF60\n(Moderate)', 'QF50\n(Medium)', 'QF40\n(Heavy)', 'QF20\n(Severe)']
    
    base_acc = [97.40, 97.20, 97.40, 96.65, 96.25, 94.80]
    coord_acc = [97.80, 97.65, 97.55, 97.15, 96.65, 95.85]

    plt.figure(figsize=(9, 5.2), dpi=300)
    x = np.arange(len(conditions))

    plt.plot(x, base_acc, marker='o', linewidth=2.5, markersize=8, color='#d9534f', label='Baseline (EfficientNet-B3)')
    plt.plot(x, coord_acc, marker='s', linewidth=2.5, markersize=8, color='#0275d8', label='Proposed (Coord-EfficientNet-B3)')

    for i in range(len(conditions)):
        plt.annotate(f"{base_acc[i]:.2f}%", (x[i], base_acc[i]), textcoords="offset points", xytext=(0, -18), ha='center', fontsize=9, color='#d9534f', fontweight='bold')
        plt.annotate(f"{coord_acc[i]:.2f}%", (x[i], coord_acc[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#0275d8', fontweight='bold')

    plt.title('DeepFake Detection Accuracy Under JPEG Compression Degradation\nBaseline (EfficientNet-B3) vs. Proposed (Coord-EfficientNet-B3)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Evaluation Regime (Compression Quality Factor)', fontsize=11, fontweight='bold')
    plt.ylabel('Classification Accuracy (%)', fontsize=11, fontweight='bold')
    plt.xticks(x, qf_labels, fontsize=10)
    plt.ylim(93.5, 98.8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10.5, loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated clean accuracy curve at {output_path}")

def generate_cleaned_error_breakdown(output_path):
    conditions = ['Clean', 'QF80', 'QF60', 'QF50', 'QF40', 'QF20']
    qf_labels = ['Clean\n(None)', 'QF80\n(Mild)', 'QF60\n(Moderate)', 'QF50\n(Medium)', 'QF40\n(Heavy)', 'QF20\n(Severe)']

    base_fp = [14, 16, 20, 36, 18, 38]
    coord_fp = [12, 10, 14, 25, 16, 32]

    base_fn = [38, 40, 32, 31, 57, 66]
    coord_fn = [32, 37, 35, 32, 51, 51]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    bar_width = 0.35
    x = np.arange(len(conditions))

    # FP Subplot
    ax1.bar(x - bar_width/2, base_fp, bar_width, label='Baseline (EfficientNet-B3)', color='#f0ad4e', edgecolor='black', alpha=0.85)
    ax1.bar(x + bar_width/2, coord_fp, bar_width, label='Proposed (Coord-EfficientNet-B3)', color='#5cb85c', edgecolor='black', alpha=0.85)
    ax1.set_title('False Positives (False Alarms on Real Faces)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Compression Condition', fontsize=10, fontweight='bold')
    ax1.set_ylabel('False Positive Count (lower is better)', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(qf_labels, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.legend(fontsize=9.5)
    for i in range(len(x)):
        ax1.text(x[i] - bar_width/2, base_fp[i] + 0.8, str(base_fp[i]), ha='center', fontsize=9, fontweight='bold')
        ax1.text(x[i] + bar_width/2, coord_fp[i] + 0.8, str(coord_fp[i]), ha='center', fontsize=9, fontweight='bold')

    # FN Subplot
    ax2.bar(x - bar_width/2, base_fn, bar_width, label='Baseline (EfficientNet-B3)', color='#d9534f', edgecolor='black', alpha=0.85)
    ax2.bar(x + bar_width/2, coord_fn, bar_width, label='Proposed (Coord-EfficientNet-B3)', color='#0275d8', edgecolor='black', alpha=0.85)
    ax2.set_title('False Negatives (Missed Synthetic Faces)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Compression Condition', fontsize=10, fontweight='bold')
    ax2.set_ylabel('False Negative Count (lower is better)', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(qf_labels, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend(fontsize=9.5)
    for i in range(len(x)):
        ax2.text(x[i] - bar_width/2, base_fn[i] + 1.0, str(base_fn[i]), ha='center', fontsize=9, fontweight='bold')
        ax2.text(x[i] + bar_width/2, coord_fn[i] + 1.0, str(coord_fn[i]), ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Error Analysis: False Positives & False Negatives Across Compression Levels', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated clean error breakdown at {output_path}")

if __name__ == '__main__':
    generate_flowchart("paper/figures/experimental_workflow.png")
    generate_cleaned_accuracy_curve("paper/figures/accuracy_vs_qf.png")
    generate_cleaned_error_breakdown("paper/figures/error_breakdown.png")
