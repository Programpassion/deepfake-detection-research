import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm

from model import EfficientNetB3Base
from dataset import get_dataloaders

def evaluate_model(model, loader, device):
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating on Test Set", leave=False):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                probs = torch.sigmoid(logits)
                
            all_probs.extend(probs.squeeze().cpu().numpy().tolist())
            all_targets.extend(targets.squeeze().cpu().numpy().tolist())
            
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    
    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec = precision_score(all_targets, all_preds)
    rec = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    
    return acc, prec, rec, f1, auc, cm, all_targets, all_preds

def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline EfficientNet-B3 Model")
    parser.add_argument("--data_dir", type=str, default=".", help="Root dataset directory")
    parser.add_argument("--model_path", type=str, default="models/efficientnet_b3_base.pth", help="Path to trained model checkpoint")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save evaluation plots")
    parser.add_argument("--test_samples_per_class", type=int, default=1000, help="Test samples per class (Total 2000 matching Section 9.1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print("\n--- Loading Test Dataset ---")
    _, _, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        train_samples_per_class=100, # Dummy for test loader
        test_samples_per_class=args.test_samples_per_class,
        batch_size=16,
        num_workers=0
    )

    print(f"\n--- Loading Trained Checkpoint: {args.model_path} ---")
    model = EfficientNetB3Base(pretrained=False).to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    acc, prec, rec, f1, auc, cm, targets, preds = evaluate_model(model, test_loader, device)

    tn, fp, fn, tp = cm.ravel()

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - EfficientNet-B3 Base')
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
    cm_path = os.path.join(args.models_dir, "confusion_matrix_base.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix plot saved to: {cm_path}")

    # Print Detailed Report and Comparison Table
    print("\n" + "="*70)
    print("      DEEPFAKE DETECTION - BASELINE MODEL EVALUATION REPORT")
    print("="*70)
    print(f"Total Test Samples Evaluated : {len(targets)} (Real: {len(targets) - int(targets.sum())}, Fake: {int(targets.sum())})")
    print(f"True Positives (Fake -> Fake): {tp}")
    print(f"True Negatives (Real -> Real): {tn}")
    print(f"False Positives (Real-> Fake): {fp}")
    print(f"False Negatives (Fake-> Real): {fn}")
    print(f"Overall Test Accuracy        : {acc:.2f}%")
    print(f"Precision (Fake Class)       : {prec:.4f}")
    print(f"Recall (Fake Class)          : {rec:.4f}")
    print(f"F1-Score (Fake Class)        : {f1:.4f}")
    print(f"ROC-AUC Score                : {auc:.4f}")
    print("="*70)

    print("\n--- DIRECT COMPARISON AGAINST BASE RESEARCH PAPER (Deepa et al., 2026) ---")
    print(f"{'Metric':<25} | {'Base Paper Reported':<20} | {'Our Simulation (Replicated)':<20}")
    print("-" * 72)
    print(f"{'Overall Accuracy':<25} | {'97.95%':<20} | {f'{acc:.2f}%':<20}")
    print(f"{'Precision':<25} | {'0.9800':<20} | {f'{prec:.4f}':<20}")
    print(f"{'Recall':<25} | {'0.9800':<20} | {f'{rec:.4f}':<20}")
    print(f"{'F1-Score':<25} | {'0.9800':<20} | {f'{f1:.4f}':<20}")
    print(f"{'True Positives (Fake)':<25} | {'988 (out of 1009)':<20} | {f'{tp} (out of {int(targets.sum())})':<20}")
    print(f"{'True Negatives (Real)':<25} | {'971 (out of 991)':<20} | {f'{tn} (out of {len(targets)-int(targets.sum())})':<20}")
    print("="*72)

if __name__ == "__main__":
    main()
