"""
generate_paper_figures.py
Generates publication-quality technical diagrams for the research paper:
1. architecture_block_diagram.png (Coord-EfficientNet-B3 Architecture)
2. experimental_workflow.png (End-to-end Experimental & Evaluation Workflow)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55)
    ax.axis('off')

    # Palette
    c_input = '#E8F0FE'
    c_backbone = '#D2E3FC'
    c_coord = '#CEEAD6'
    c_head = '#FCE8E6'
    c_output = '#FEF7E0'
    c_border_blue = '#1A73E8'
    c_border_green = '#1E8E3E'
    c_border_red = '#D93025'
    c_border_yellow = '#F9AB00'
    c_text = '#202124'

    def add_box(x, y, w, h, title, subtitle="", bg='#FFFFFF', border='#5F6368', radius=1.5):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
                             facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color=c_text)
            ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center',
                    fontsize=7.5, color='#3C4043')
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=c_text)

    def add_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color='#3C4043', lw=1.5, mutation_scale=12))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.2, label, ha='center', va='bottom',
                    fontsize=7, color='#5F6368', fontweight='bold')

    # Main workflow boxes (Left to Right)
    # 1. Input
    add_box(2, 22, 12, 14, "Input Image", "224 x 224 x 3\nFace Crop", bg=c_input, border=c_border_blue)
    add_arrow(14, 29, 18, 29)

    # 2. Backbone
    add_box(18, 19, 16, 20, "EfficientNet-B3\nBackbone", "ImageNet Pretrained\nStem + MBConv1-7\nExtracts $F \in \mathbb{R}^{1536 \\times 7 \\times 7}$",
            bg=c_backbone, border=c_border_blue)
    add_arrow(34, 29, 38, 29, "$F$")

    # 3. Coordinate Attention Group (Container Box)
    coord_group = FancyBboxPatch((38, 7), 36, 43, boxstyle="round,pad=0.5,rounding_size=2.0",
                                 facecolor='#F6FBF7', edgecolor=c_border_green, linewidth=1.8, linestyle='--')
    ax.add_patch(coord_group)
    ax.text(56, 47.5, "Proposed Coordinate Attention Module", ha='center', va='center',
            fontsize=10, fontweight='bold', color=c_border_green)

    # Sub-blocks inside Coordinate Attention
    # 1D Horizontal & Vertical Pooling
    add_box(40, 34, 14, 9, "X-AvgPool (1D)", "Pool $W \\to 1$\n$[1536 \\times 7 \\times 1]$", bg=c_coord, border=c_border_green)
    add_box(40, 15, 14, 9, "Y-AvgPool (1D)", "Pool $H \\to 1$\n$[1536 \\times 1 \\times 7]$", bg=c_coord, border=c_border_green)

    # Shared Conv Block
    add_box(58, 24, 14, 11, "Shared 1x1 Conv\nBN + Hard-Swish", "Reduction $r=32$\n$1536 \\to 48$ channels", bg=c_coord, border=c_border_green)

    # Arrows inside CoordAtt
    add_arrow(34, 29, 40, 38.5)
    add_arrow(34, 29, 40, 19.5)
    add_arrow(54, 38.5, 58, 31)
    add_arrow(54, 19.5, 58, 27)

    # Modulated Output inside CoordAtt
    add_box(58, 9, 14, 10, "Split 1x1 Convs\n& Sigmoid ($g^h, g^w$)", "$48 \\to 1536$ channels", bg=c_coord, border=c_border_green)
    add_arrow(65, 24, 65, 19)

    add_box(40, 9, 14, 5, "Feature Reweight", "$Y = F \\otimes g^h \\otimes g^w$", bg=c_coord, border=c_border_green)
    add_arrow(58, 11.5, 54, 11.5)

    add_arrow(40, 11.5, 36, 11.5)
    ax.plot([36, 36, 76, 76], [11.5, 3, 3, 29], color='#3C4043', lw=1.5)
    ax.annotate('', xy=(78, 29), xytext=(76, 29),
                arrowprops=dict(arrowstyle="-|>", color='#3C4043', lw=1.5, mutation_scale=12))
    ax.text(56, 4.5, "Modulated Representation $Y \\in \\mathbb{R}^{1536 \\times 7 \\times 7}$",
            ha='center', va='center', fontsize=7.5, color='#1E8E3E', fontweight='bold')

    # 4. Classifier Head
    add_box(78, 18, 10, 22, "Classifier Head", "Global Avg Pool\n↓\nFC 1536→128 (p=0.3)\n↓\nFC 128→64 (p=0.2)\n↓\nFC 64→1 (Logit $z$)",
            bg=c_head, border=c_border_red)
    add_arrow(88, 29, 91, 29)

    # 5. Output / Decision
    add_box(91, 23, 8, 12, "Decision", "Sigmoid\n$P(\\text{Fake}) = \\sigma(z)$\nThreshold $\\tau$",
            bg=c_output, border=c_border_yellow)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved architecture block diagram to {output_path}")

def create_workflow_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    c_blue = '#E8F0FE'
    c_green = '#CEEAD6'
    c_orange = '#FEF7E0'
    c_purple = '#F3E8FD'
    b_blue = '#1A73E8'
    b_green = '#1E8E3E'
    b_orange = '#F9AB00'
    b_purple = '#9334E6'
    c_text = '#202124'

    def add_card(x, y, w, h, title, items, bg='#FFFFFF', border='#5F6368'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                             facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 2.5, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=c_text)
        item_y = y + h - 5.5
        for it in items:
            ax.text(x + 1.2, item_y, f"• {it}", ha='left', va='center',
                    fontsize=7.3, color='#3C4043')
            item_y -= 2.6

    def add_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color='#3C4043', lw=1.8, mutation_scale=14))

    # Column 1: Dataset Partitioning & Hygiene
    add_card(2, 4, 21, 42, "1. Dataset Curation", [
        "140,000 Faces (Karki Benchmark)",
        "Real: FFHQ (authentic human)",
        "Fake: NVIDIA StyleGAN",
        "Strict Partitioning & Deduplication:",
        "Train Split: 7,200 (balanced)",
        "Validation: 800 (Clean faces)",
        "Test Split: 2,000 (1k Real, 1k Fake)",
        "Holdout Set: 200 Unseen Faces",
        "SHA-256 / dHash Verified Zero-Leak"
    ], bg=c_blue, border=b_blue)

    add_arrow(23, 25, 27, 25)

    # Column 2: Compression-Aware Training
    add_card(27, 4, 21, 42, "2. Robust Training", [
        "Backbone: EfficientNet-B3",
        "+ Coordinate Attention (CA)",
        "Compression-Aware Augmentation:",
        "p = 0.50 Stochastic JPEG",
        "QF in {20, 40, 50, 60, 80}",
        "2-Stage Transfer Learning:",
        "Ep 1-2: Warmup (Backbone frozen)",
        "Ep 3-10: End-to-end Fine-Tuning",
        "Loss: Binary Cross-Entropy",
        "Optimizer: Adam, lr = 1e-4, B = 16"
    ], bg=c_green, border=b_green)

    add_arrow(48, 25, 52, 25)

    # Column 3: Multi-Regime Evaluation
    add_card(52, 4, 21, 42, "3. Multi-Regime Testing", [
        "6 Evaluation Conditions:",
        "Clean, QF80, QF60, QF50, QF40, QF20",
        "Standard Multi-QF Benchmark:",
        "2,000 images x 6 regimes",
        "= 12,000 Total Inferences",
        "Unseen Facial Holdout Benchmark:",
        "200 images x 6 regimes",
        "= 1,200 Total Inferences",
        "Strict Frozen Baseline vs CoordAtt"
    ], bg=c_orange, border=b_orange)

    add_arrow(73, 25, 77, 25)

    # Column 4: Comprehensive Analysis & Validation
    add_card(77, 4, 21, 42, "4. Statistical Validation", [
        "Performance Metrics:",
        "Accuracy, Precision, Recall, F1",
        "Matthews Correlation Coeff (MCC)",
        "FP & FN Error Breakdown",
        "Significance Testing:",
        "Paired McNemar Test (p < 0.01)",
        "Decision Threshold Optimization:",
        "tau in [0.15, 0.50] for High Recall",
        "Probability Drift Characterization:",
        "P(Fake) Distribution Stability"
    ], bg=c_purple, border=b_purple)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved experimental workflow diagram to {output_path}")

if __name__ == '__main__':
    create_architecture_diagram("paper/figures/architecture_block_diagram.png")
    create_workflow_diagram("paper/figures/experimental_workflow.png")
