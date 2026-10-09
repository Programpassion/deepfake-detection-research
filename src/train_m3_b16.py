"""
train_and_eval_m3_16_compression_aware.py
PHASE 3.2-A / M3-B16 — COMPRESSION-AWARE EFFICIENTNET-B3 WITH BATCH SIZE 16

Controlled experimental investigation:
Tests whether doubling the batch size from 8 to 16 during online stochastic
JPEG compression-aware training improves representation learning and compression
robustness for the EfficientNet-B3 backbone.

Key Fairness Constraints & Methodology:
  - Architecture: Native ImageNet-pretrained EfficientNet-B3 (No CBAM) + 3-Stage Head
      1536 -> Linear(128) -> ReLU -> Dropout(0.30) -> Linear(64) -> ReLU -> Dropout(0.20) -> Linear(1)
  - Optimizer: Adam, LR = 1e-4 throughout (NO 1e-5)
  - Batch Size = 16 (doubled from 8)
  - Epochs = 10, Seed = 42
  - Stage 1 (Epochs 1-2): Backbone FROZEN, Classifier TRAINABLE, LR = 1e-4
  - Stage 2 (Epochs 3-10): Backbone UNFROZEN, Full Network TRAINABLE, LR = 1e-4
  - Training Data: Online stochastic JPEG compression:
      Clean = 0.50, QF80 = 0.10, QF60 = 0.10, QF50 = 0.10, QF40 = 0.10, QF20 = 0.10
  - Validation: CLEAN only (val.csv, 800 images). Model selection: best val accuracy.
  - Test: Frozen test set (test.csv, 2,000 images) evaluated on Clean, QF80, QF60, QF50, QF40, QF20.
  - Output: Checkpoint, Condition JSONs, Prediction CSV, Stability CSV, Drift CSV, Comparison CSV, Visualizations.
"""

import os
import sys
import io
import time
import json
import csv
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, matthews_corrcoef,
    confusion_matrix
)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import EfficientNetB3Baseline, count_parameters
from dataset_loader import get_phase2_transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CompressionAwareDeepFakeDataset(Dataset):
    """
    Dataset implementing online stochastic JPEG compression augmentation.
    Probabilities:
      Clean = 0.50, QF80 = 0.10, QF60 = 0.10, QF50 = 0.10, QF40 = 0.10, QF20 = 0.10
    """
    def __init__(self, csv_path, is_training=True, transform=None, fixed_condition=None, fixed_qf=None):
        self.csv_path = csv_path
        self.is_training = is_training
        self.transform = transform
        self.fixed_condition = fixed_condition
        self.fixed_qf = fixed_qf
        self.records = []

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = 0.0 if row['class'] == 'Real' else 1.0
                self.records.append({
                    'image_id': row['image_id'],
                    'original_filename': row['original_filename'],
                    'path': row['original_path'],
                    'class': row['class'],
                    'label': label,
                    'split': row['split']
                })

        self.conditions = ['Clean', 'QF80', 'QF60', 'QF50', 'QF40', 'QF20']
        self.probabilities = [0.50, 0.10, 0.10, 0.10, 0.10, 0.10]
        self.qf_dict = {'Clean': None, 'QF80': 80, 'QF60': 60, 'QF50': 50, 'QF40': 40, 'QF20': 20}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image = Image.open(item['path']).convert("RGB")

        if self.is_training:
            chosen_cond = random.choices(self.conditions, weights=self.probabilities, k=1)[0]
            qf = self.qf_dict[chosen_cond]
        else:
            chosen_cond = self.fixed_condition if self.fixed_condition is not None else 'Clean'
            qf = self.fixed_qf

        if qf is not None:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=qf)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor([item['label']], dtype=torch.float32),
            'image_id': item['image_id'],
            'original_filename': item['original_filename'],
            'class': item['class'],
            'condition_applied': chosen_cond
        }

def evaluate_test_condition(model, dataset, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_probs = []
    all_targets = []
    all_preds = []
    individual_preds = []
    all_features = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            image_ids = batch['image_id']
            filenames = batch['original_filename']
            classes = batch['class']

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                feat = model.backbone(images) # [B, 1536]
                logits = model.classifier(feat) # [B, 1]
                probs = torch.sigmoid(logits)

            probs_np = probs.cpu().squeeze().tolist()
            if not isinstance(probs_np, list):
                probs_np = [probs_np]

            targets_np = labels.cpu().squeeze().tolist()
            if not isinstance(targets_np, list):
                targets_np = [targets_np]

            all_probs.extend(probs_np)
            all_targets.extend(targets_np)
            all_features.append(feat.cpu().numpy())

            for i in range(len(probs_np)):
                p = probs_np[i]
                pred_label = "Fake" if p >= 0.5 else "Real"
                all_preds.append(1.0 if p >= 0.5 else 0.0)
                individual_preds.append({
                    'image_id': image_ids[i],
                    'original_filename': filenames[i],
                    'true_label': classes[i],
                    'predicted_label': pred_label,
                    'probability_fake': round(p, 6),
                    'condition': dataset.fixed_condition,
                    'jpeg_quality': dataset.fixed_qf if dataset.fixed_qf is not None else 'NA'
                })

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_features = np.vstack(all_features) # [N, 1536]

    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, zero_division=0))
    rec = float(recall_score(all_targets, all_preds, zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, zero_division=0))
    roc_auc = float(roc_auc_score(all_targets, all_probs))

    precisions_curve, recalls_curve, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc = float(auc(recalls_curve, precisions_curve))
    mcc = float(matthews_corrcoef(all_targets, all_preds))
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics = {
        'condition': dataset.fixed_condition,
        'jpeg_quality': dataset.fixed_qf if dataset.fixed_qf is not None else 'NA',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'FPR': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'FNR': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'total_samples': len(all_targets)
    }

    return metrics, individual_preds, all_features

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("PHASE 3.2-A / M3-B16: COMPRESSION-AWARE EFFICIENTNET-B3 (BATCH SIZE = 16)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # Directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    reports_dir = os.path.join(base_dir, 'reports')
    viz_dir = os.path.join(base_dir, 'visualizations', 'm3_16_compression_aware')
    brain_dir = r"C:\Users\hdutt\.gemini\antigravity\brain\5a93e7ba-552e-46fb-93aa-afeae126c4e7"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    config = {
        'experiment_name': 'PHASE 3.2-A / M3-B16 — COMPRESSION-AWARE B3 (BATCH 16)',
        'model_architecture': 'EfficientNet-B3 (No Attention)',
        'backbone': 'efficientnet_b3 (ImageNet-pretrained)',
        'classifier': '1536 -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1)',
        'loss_function': 'BCEWithLogitsLoss',
        'optimizer': 'Adam',
        'learning_rate': 0.0001,
        'batch_size': 16,
        'num_epochs': 10,
        'stage1_epochs': 2,
        'stage2_epochs': 8,
        'random_seed': 42,
        'compression_sampling_probabilities': {
            'Clean': 0.50,
            'QF80': 0.10,
            'QF60': 0.10,
            'QF50': 0.10,
            'QF40': 0.10,
            'QF20': 0.10
        },
        'validation_protocol': 'Clean Standardized val.csv (800 images)',
        'checkpoint_selection': 'Best validation accuracy'
    }

    config_path = os.path.join(results_dir, 'm3_16_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to: {config_path}")

    # 1. Datasets & Transforms
    train_tx, eval_tx = get_phase2_transforms()
    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv = os.path.join(data_dir, 'val.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    train_dataset = CompressionAwareDeepFakeDataset(train_csv, is_training=True, transform=train_tx)
    val_dataset = CompressionAwareDeepFakeDataset(val_csv, is_training=False, transform=eval_tx, fixed_condition='Clean', fixed_qf=None)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    print(f"Loaded Train: {len(train_dataset)} images ({len(train_loader)} batches of 16) | Val: {len(val_dataset)} images")

    # 2. Model Initialization
    model = EfficientNetB3Baseline(pretrained=True).to(device)
    tot_params, tr_params, ntr_params = count_parameters(model)
    print(f"Model Initialized: EfficientNet-B3 (Total Parameters: {tot_params:,})")

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    num_epochs = config['num_epochs']
    frozen_epochs = config['stage1_epochs']
    lr = config['learning_rate']

    history = []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = os.path.join(models_dir, 'efficientnet_b3_compression_aware_m3_16_best.pth')

    print("\n" + "=" * 80)
    print("STARTING M3-B16 TRAINING (10 EPOCHS, BATCH SIZE = 16)")
    print("=" * 80)

    start_training_time = time.time()

    # 3. Training Loop
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        if epoch <= frozen_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = Adam(trainable_params, lr=lr)
            stage_str = "STAGE 1: BACKBONE FROZEN (Classifier Only)"
            backbone_frozen = True
        else:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=lr)
            stage_str = "STAGE 2: FULL NETWORK FINE-TUNING (Backbone Unfrozen)"
            backbone_frozen = False

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        cond_counts = {c: 0 for c in train_dataset.conditions}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs:02d} [{stage_str}]", leave=False):
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            for c in batch['condition_applied']:
                cond_counts[c] += 1

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Clean Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = batch['label'].to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start

        epoch_record = {
            'epoch': epoch,
            'stage': stage_str,
            'backbone_frozen': backbone_frozen,
            'train_loss': round(epoch_train_loss, 6),
            'train_acc': round(epoch_train_acc, 6),
            'val_loss': round(epoch_val_loss, 6),
            'val_acc': round(epoch_val_acc, 6),
            'learning_rate': lr,
            'epoch_time_seconds': round(epoch_time, 2),
            'train_clean_count': cond_counts['Clean'],
            'train_qf80_count': cond_counts['QF80'],
            'train_qf60_count': cond_counts['QF60'],
            'train_qf50_count': cond_counts['QF50'],
            'train_qf40_count': cond_counts['QF40'],
            'train_qf20_count': cond_counts['QF20']
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) | {stage_str}")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"  Val Loss  : {epoch_val_loss:.4f} | Val Acc  : {epoch_val_acc*100:.2f}%")
        print(f"  Augment Exposures: Clean={cond_counts['Clean']}, QF80={cond_counts['QF80']}, QF60={cond_counts['QF60']}, QF50={cond_counts['QF50']}, QF40={cond_counts['QF40']}, QF20={cond_counts['QF20']}")

        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': best_val_loss,
                'train_acc': epoch_train_acc,
                'train_loss': epoch_train_loss,
                'config': config
            }, best_model_path)
            print(f"  >>> Best Checkpoint Saved at Epoch {epoch:02d} (Val Acc: {best_val_acc*100:.2f}%) <<<")

    total_training_time = time.time() - start_training_time
    print(f"\nTraining Complete in {total_training_time/60:.2f} minutes.")
    print(f"Best Epoch: {best_epoch:02d} with Validation Accuracy: {best_val_acc*100:.2f}%")

    # Save training history CSV
    history_csv_path = os.path.join(results_dir, 'm3_16_training_history.csv')
    with open(history_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training history to: {history_csv_path}")

    # Plot Training & Validation Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], 'b-o', label='Train Loss')
    plt.plot([h['epoch'] for h in history], [h['val_loss'] for h in history], 'r-s', label='Val Loss')
    plt.axvline(best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('M3-B16 (Batch 16): Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot([h['epoch'] for h in history], [h['train_acc']*100 for h in history], 'b-o', label='Train Acc')
    plt.plot([h['epoch'] for h in history], [h['val_acc']*100 for h in history], 'r-s', label='Val Acc')
    plt.axvline(best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('M3-B16 (Batch 16): Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    hist_plot_path = os.path.join(viz_dir, 'm3_16_training_validation_history.png')
    plt.savefig(hist_plot_path, dpi=300)
    plt.close()

    # 4. Multi-QF Evaluation on Frozen Test Set
    print("\n" + "=" * 80)
    print("STARTING MULTI-QF EVALUATION ON FROZEN TEST SET (2,000 IMAGES)")
    print("=" * 80)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_conditions = [
        {'name': 'Clean', 'qf': None},
        {'name': 'QF80',  'qf': 80},
        {'name': 'QF60',  'qf': 60},
        {'name': 'QF50',  'qf': 50},
        {'name': 'QF40',  'qf': 40},
        {'name': 'QF20',  'qf': 20}
    ]

    master_metrics = []
    all_predictions = []
    condition_features = {}
    condition_probs = {}

    for cond in test_conditions:
        c_name = cond['name']
        c_qf = cond['qf']
        print(f"Evaluating Condition: {c_name} (JPEG QF={c_qf if c_qf else 'Original'})...")

        test_ds = CompressionAwareDeepFakeDataset(
            test_csv, is_training=False, transform=eval_tx,
            fixed_condition=c_name, fixed_qf=c_qf
        )

        metrics, preds, feats = evaluate_test_condition(model, test_ds, device, batch_size=16)
        master_metrics.append(metrics)
        all_predictions.extend(preds)
        condition_features[c_name] = feats
        condition_probs[c_name] = np.array([p['probability_fake'] for p in preds])

        json_filename = f"m3_16_{c_name.lower()}_metrics.json"
        with open(os.path.join(results_dir, json_filename), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)

        print(f"  Acc: {metrics['accuracy']*100:.2f}% | Rec: {metrics['recall']:.4f} | Prec: {metrics['precision']:.4f} | F1: {metrics['f1_score']:.4f} | FP: {metrics['FP']}, FN: {metrics['FN']}")

    # Save Predictions CSV
    pred_csv_path = os.path.join(results_dir, 'm3_16_predictions.csv')
    with open(pred_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0].keys()))
        writer.writeheader()
        writer.writerows(all_predictions)
    print(f"Saved test predictions to: {pred_csv_path}")

    # 5. Direct Comparison against M3-B8 and Phase 2 Baseline (M1)
    # Load M3-B8 metrics from phase3_2_compression_aware_2R_*_metrics.json
    m3_b8_metrics = {}
    for cond in test_conditions:
        c_name = cond['name']
        c_lower = c_name.lower()
        m3_b8_json = os.path.join(results_dir, f"phase3_2_compression_aware_2R_{c_lower}_metrics.json")
        with open(m3_b8_json, 'r', encoding='utf-8') as f:
            m3_b8_metrics[c_name] = json.load(f)

    # Load Phase 2 baseline metrics
    p2_metrics = {}
    with open(os.path.join(results_dir, 'multi_qf_master_metrics.csv'), 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p2_metrics[r['condition']] = {
                'accuracy': float(r['accuracy']),
                'precision': float(r['precision']),
                'recall': float(r['recall']),
                'f1_score': float(r['f1']),
                'roc_auc': float(r['roc_auc']),
                'pr_auc': float(r['pr_auc']),
                'mcc': float(r['mcc']),
                'TP': int(r['TP']),
                'TN': int(r['TN']),
                'FP': int(r['FP']),
                'FN': int(r['FN']),
                'FPR': float(r['FPR']),
                'FNR': float(r['FNR'])
            }

    comparison_records = []
    for m in master_metrics:
        c = m['condition']
        b8 = m3_b8_metrics[c]
        p2 = p2_metrics[c]

        delta_acc_vs_b8 = (m['accuracy'] - b8['accuracy']) * 100
        delta_f1_vs_b8 = (m['f1_score'] - b8['f1_score']) * 100
        delta_fp_vs_b8 = m['FP'] - b8['FP']
        delta_fn_vs_b8 = m['FN'] - b8['FN']

        delta_acc_vs_p2 = (m['accuracy'] - p2['accuracy']) * 100
        delta_fp_vs_p2 = m['FP'] - p2['FP']
        delta_fn_vs_p2 = m['FN'] - p2['FN']

        comparison_records.append({
            'condition': c,
            'jpeg_quality': m['jpeg_quality'],
            'm3_16_accuracy': round(m['accuracy'], 6),
            'm3_b8_accuracy': round(b8['accuracy'], 6),
            'p2_baseline_accuracy': round(p2['accuracy'], 6),
            'delta_acc_vs_m3_b8_pp': round(delta_acc_vs_b8, 2),
            'delta_acc_vs_p2_pp': round(delta_acc_vs_p2, 2),
            'm3_16_precision': round(m['precision'], 6),
            'm3_b8_precision': round(b8['precision'], 6),
            'p2_precision': round(p2['precision'], 6),
            'm3_16_recall': round(m['recall'], 6),
            'm3_b8_recall': round(b8['recall'], 6),
            'p2_recall': round(p2['recall'], 6),
            'm3_16_f1': round(m['f1_score'], 6),
            'm3_b8_f1': round(b8['f1_score'], 6),
            'p2_f1': round(p2['f1_score'], 6),
            'delta_f1_vs_m3_b8_pp': round(delta_f1_vs_b8, 2),
            'm3_16_roc_auc': round(m['roc_auc'], 6),
            'm3_b8_roc_auc': round(b8['roc_auc'], 6),
            'p2_roc_auc': round(p2['roc_auc'], 6),
            'm3_16_pr_auc': round(m['pr_auc'], 6),
            'm3_b8_pr_auc': round(b8['pr_auc'], 6),
            'p2_pr_auc': round(p2['pr_auc'], 6),
            'm3_16_mcc': round(m['mcc'], 6),
            'm3_b8_mcc': round(b8['mcc'], 6),
            'p2_mcc': round(p2['mcc'], 6),
            'm3_16_TP': m['TP'],
            'm3_b8_TP': b8['TP'],
            'p2_TP': p2['TP'],
            'm3_16_TN': m['TN'],
            'm3_b8_TN': b8['TN'],
            'p2_TN': p2['TN'],
            'm3_16_FP': m['FP'],
            'm3_b8_FP': b8['FP'],
            'delta_FP_vs_m3_b8': delta_fp_vs_b8,
            'p2_FP': p2['FP'],
            'delta_FP_vs_p2': delta_fp_vs_p2,
            'm3_16_FN': m['FN'],
            'm3_b8_FN': b8['FN'],
            'delta_FN_vs_m3_b8': delta_fn_vs_b8,
            'p2_FN': p2['FN'],
            'delta_FN_vs_p2': delta_fn_vs_p2
        })

    comp_csv_path = os.path.join(results_dir, 'm3_16_master_comparison.csv')
    with open(comp_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_records[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_records)
    print(f"Saved master comparison to: {comp_csv_path}")

    # 6. Probability Stability Analysis
    prob_stab_records = []
    clean_p = condition_probs['Clean']

    for c in ['QF80', 'QF60', 'QF50', 'QF40', 'QF20']:
        qf_p = condition_probs[c]
        diffs = np.abs(qf_p - clean_p)
        mean_d = float(np.mean(diffs))
        median_d = float(np.median(diffs))
        p95_d = float(np.percentile(diffs, 95))
        max_d = float(np.max(diffs))

        prob_stab_records.append({
            'qf_condition': c,
            'mean_abs_prob_change': round(mean_d, 6),
            'median_abs_prob_change': round(median_d, 6),
            'p95_abs_prob_change': round(p95_d, 6),
            'max_abs_prob_change': round(max_d, 6)
        })

    prob_stab_csv = os.path.join(results_dir, 'm3_16_probability_stability.csv')
    with open(prob_stab_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(prob_stab_records[0].keys()))
        writer.writeheader()
        writer.writerows(prob_stab_records)
    print(f"Saved probability stability to: {prob_stab_csv}")

    # 7. Representation-Level Feature Drift Analysis
    print("\nComputing Representation-Level Backbone Drift...")
    # Load M3-B8 model to extract its representations for head-to-head comparison
    m3_b8_ckpt_path = os.path.join(models_dir, 'efficientnet_b3_compression_aware_2R_best.pth')
    m3_b8_model = EfficientNetB3Baseline(pretrained=False).to(device)
    m3_b8_ckpt = torch.load(m3_b8_ckpt_path, map_location=device, weights_only=False)
    m3_b8_model.load_state_dict(m3_b8_ckpt['model_state_dict'])
    m3_b8_model.eval()

    # Load Phase 2 baseline model
    p2_ckpt_path = os.path.join(models_dir, 'efficientnet_b3_baseline_best.pth')
    p2_model = EfficientNetB3Baseline(pretrained=False).to(device)
    p2_ckpt = torch.load(p2_ckpt_path, map_location=device, weights_only=False)
    p2_model.load_state_dict(p2_ckpt['model_state_dict'])
    p2_model.eval()

    p2_clean_feats, p2_qf_feats = None, {}
    b8_clean_feats, b8_qf_feats = None, {}

    with torch.no_grad():
        for cond in test_conditions:
            c_name = cond['name']
            c_qf = cond['qf']
            test_ds = CompressionAwareDeepFakeDataset(
                test_csv, is_training=False, transform=eval_tx,
                fixed_condition=c_name, fixed_qf=c_qf
            )
            loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
            p2_fl, b8_fl = [], []
            for batch in loader:
                imgs = batch['image'].to(device)
                p2_fl.append(p2_model.backbone(imgs).cpu().numpy())
                b8_fl.append(m3_b8_model.backbone(imgs).cpu().numpy())
            p2_arr = np.vstack(p2_fl)
            b8_arr = np.vstack(b8_fl)

            if c_name == 'Clean':
                p2_clean_feats = p2_arr
                b8_clean_feats = b8_arr
            else:
                p2_qf_feats[c_name] = p2_arr
                b8_qf_feats[c_name] = b8_arr

    drift_records = []
    m3_16_clean_f = condition_features['Clean']

    for c in ['QF80', 'QF60', 'QF50', 'QF40', 'QF20']:
        # M3-B16 drifts
        m3_16_qf_f = condition_features[c]
        m3_16_l2 = np.linalg.norm(m3_16_qf_f - m3_16_clean_f, axis=1)
        m3_16_clean_norms = np.linalg.norm(m3_16_clean_f, axis=1)
        m3_16_rel_l2 = m3_16_l2 / (m3_16_clean_norms + 1e-8)
        m3_16_cos = np.sum(m3_16_qf_f * m3_16_clean_f, axis=1) / (np.linalg.norm(m3_16_qf_f, axis=1) * m3_16_clean_norms + 1e-8)

        # M3-B8 drifts
        b8_qf_f = b8_qf_feats[c]
        b8_l2 = np.linalg.norm(b8_qf_f - b8_clean_feats, axis=1)
        b8_clean_norms = np.linalg.norm(b8_clean_feats, axis=1)
        b8_rel_l2 = b8_l2 / (b8_clean_norms + 1e-8)
        b8_cos = np.sum(b8_qf_f * b8_clean_feats, axis=1) / (np.linalg.norm(b8_qf_f, axis=1) * b8_clean_norms + 1e-8)

        # Phase 2 drifts
        p2_qf_f = p2_qf_feats[c]
        p2_l2 = np.linalg.norm(p2_qf_f - p2_clean_feats, axis=1)
        p2_clean_norms = np.linalg.norm(p2_clean_feats, axis=1)
        p2_rel_l2 = p2_l2 / (p2_clean_norms + 1e-8)
        p2_cos = np.sum(p2_qf_f * p2_clean_feats, axis=1) / (np.linalg.norm(p2_qf_f, axis=1) * p2_clean_norms + 1e-8)

        drift_records.append({
            'qf_condition': c,
            'm3_16_mean_l2_drift': float(np.mean(m3_16_l2)),
            'm3_b8_mean_l2_drift': float(np.mean(b8_l2)),
            'p2_mean_l2_drift': float(np.mean(p2_l2)),
            'l2_reduction_vs_b8_pct': float((np.mean(b8_l2) - np.mean(m3_16_l2)) / np.mean(b8_l2) * 100),
            'l2_reduction_vs_p2_pct': float((np.mean(p2_l2) - np.mean(m3_16_l2)) / np.mean(p2_l2) * 100),
            'm3_16_rel_l2_drift': float(np.mean(m3_16_rel_l2)),
            'm3_b8_rel_l2_drift': float(np.mean(b8_rel_l2)),
            'p2_rel_l2_drift': float(np.mean(p2_rel_l2)),
            'm3_16_cosine_similarity': float(np.mean(m3_16_cos)),
            'm3_b8_cosine_similarity': float(np.mean(b8_cos)),
            'p2_cosine_similarity': float(np.mean(p2_cos))
        })

    drift_csv = os.path.join(results_dir, 'm3_16_representation_drift.csv')
    with open(drift_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(drift_records[0].keys()))
        writer.writeheader()
        writer.writerows(drift_records)
    print(f"Saved representation drift to: {drift_csv}")

    # 8. Visualizations
    print("\nGenerating Diagnostic Visualizations...")
    qf_labels = ['Clean', 'QF80', 'QF60', 'QF50', 'QF40', 'QF20']

    # Plot 1: Accuracy vs Quality Factor (M1 vs M3-B8 vs M3-B16)
    m3_16_accs = [r['m3_16_accuracy']*100 for r in comparison_records]
    m3_b8_accs = [r['m3_b8_accuracy']*100 for r in comparison_records]
    p2_accs = [r['p2_baseline_accuracy']*100 for r in comparison_records]

    plt.figure(figsize=(9, 5.5))
    plt.plot(qf_labels, p2_accs, 'gray', linestyle=':', marker='o', label='M1 Baseline (Clean B3)', lw=1.8)
    plt.plot(qf_labels, m3_b8_accs, 'blue', linestyle='-.', marker='s', label='M3-B8 (CA-B3, Batch 8)', lw=2.2)
    plt.plot(qf_labels, m3_16_accs, '#9467bd', linestyle='-', marker='D', label='M3-B16 (CA-B3, Batch 16)', lw=2.5)
    plt.title('Accuracy vs Compression: M1 vs M3-B8 vs M3-B16', fontsize=12, fontweight='bold')
    plt.xlabel('Condition / Quality Factor', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.ylim(93, 99)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    for i, (b8_a, b16_a) in enumerate(zip(m3_b8_accs, m3_16_accs)):
        diff = b16_a - b8_a
        diff_str = f"{diff:+.2f}%"
        color = 'darkgreen' if diff >= 0 else 'firebrick'
        plt.annotate(f"{b16_a:.2f}%\n({diff_str})", (i, b16_a + 0.25), ha='center', fontsize=9, color=color, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'm3_16_accuracy_vs_qf_comparison.png'), dpi=300)
    plt.close()

    # Plot 2: FP and FN vs QF
    plt.figure(figsize=(10, 5))
    x = np.arange(len(qf_labels))
    width = 0.25
    m3_16_fps = [r['m3_16_FP'] for r in comparison_records]
    m3_b8_fps = [r['m3_b8_FP'] for r in comparison_records]
    p2_fps = [r['p2_FP'] for r in comparison_records]

    m3_16_fns = [r['m3_16_FN'] for r in comparison_records]
    m3_b8_fns = [r['m3_b8_FN'] for r in comparison_records]
    p2_fns = [r['p2_FN'] for r in comparison_records]

    plt.subplot(1, 2, 1)
    plt.bar(x - width, p2_fps, width, label='M1 Baseline', color='gray', alpha=0.7)
    plt.bar(x, m3_b8_fps, width, label='M3-B8', color='blue', alpha=0.7)
    plt.bar(x + width, m3_16_fps, width, label='M3-B16', color='#9467bd', alpha=0.85)
    plt.title('False Positives (Real Misclassified as Fake)', fontsize=11, fontweight='bold')
    plt.xticks(x, qf_labels)
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.bar(x - width, p2_fns, width, label='M1 Baseline', color='gray', alpha=0.7)
    plt.bar(x, m3_b8_fns, width, label='M3-B8', color='blue', alpha=0.7)
    plt.bar(x + width, m3_16_fns, width, label='M3-B16', color='#9467bd', alpha=0.85)
    plt.title('False Negatives (Missed Deepfakes)', fontsize=11, fontweight='bold')
    plt.xticks(x, qf_labels)
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'm3_16_fp_fn_vs_qf.png'), dpi=300)
    plt.close()

    # Plot 3: Representation Drift Comparison
    qfs_only = ['QF80', 'QF60', 'QF50', 'QF40', 'QF20']
    p2_drifts = [r['p2_mean_l2_drift'] for r in drift_records]
    m3_b8_drifts = [r['m3_b8_mean_l2_drift'] for r in drift_records]
    m3_16_drifts = [r['m3_16_mean_l2_drift'] for r in drift_records]

    x_d = np.arange(len(qfs_only))
    w_d = 0.25
    plt.figure(figsize=(9, 5.5))
    plt.bar(x_d - w_d, p2_drifts, w_d, label='M1 Baseline (Clean-trained)', color='gray', alpha=0.7)
    plt.bar(x_d, m3_b8_drifts, w_d, label='M3-B8 (Batch 8)', color='blue', alpha=0.7)
    plt.bar(x_d + w_d, m3_16_drifts, w_d, label='M3-B16 (Batch 16)', color='#9467bd', alpha=0.85)
    plt.title('Backbone Representation Drift ($\mathbb{E}[||X_{QF} - X_{clean}||_2$])', fontsize=12, fontweight='bold')
    plt.xlabel('JPEG Quality Factor', fontsize=11)
    plt.ylabel('Mean L2 Feature Distance', fontsize=11)
    plt.xticks(x_d, qfs_only)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'm3_16_representation_drift_comparison.png'), dpi=300)
    plt.close()

    # Copy plots to brain dir
    for img_f in os.listdir(viz_dir):
        if img_f.endswith('.png'):
            src_f = os.path.join(viz_dir, img_f)
            dst_f = os.path.join(brain_dir, img_f)
            with open(src_f, 'rb') as sf, open(dst_f, 'wb') as df:
                df.write(sf.read())

    print("\n" + "=" * 90)
    print("PHASE 3.2-A / M3-B16 MASTER COMPARISON SUMMARY")
    print("=" * 90)
    for r in comparison_records:
        print(f"[{r['condition']:<5}] M3-B16 Acc: {r['m3_16_accuracy']*100:.2f}% | M3-B8: {r['m3_b8_accuracy']*100:.2f}% (Delta: {r['delta_acc_vs_m3_b8_pp']:+6.2f} pp) | FP: {r['m3_16_FP']} (B8: {r['m3_b8_FP']}), FN: {r['m3_16_FN']} (B8: {r['m3_b8_FN']}) | ROC-AUC: {r['m3_16_roc_auc']:.4f} (B8: {r['m3_b8_roc_auc']:.4f})")
    print("=" * 90)

if __name__ == '__main__':
    main()
