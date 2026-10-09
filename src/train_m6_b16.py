import os
import sys
import io
import time
import random
import json
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# Import custom architecture
from model_coordatt import EfficientNetB3CoordAtt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable

def get_phase2_transforms():
    train_tx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_tx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tx, eval_tx

class CompressionAwareDeepFakeDataset(Dataset):
    def __init__(self, csv_path, is_training=False, transform=None, fixed_condition=None, fixed_qf=None):
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

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 85)
    print("PHASE 5.B: M6-B16 (COMPRESSION-AWARE EFFICIENTNET-B3 + COORDINATE ATTENTION, BATCH 16)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    reports_dir = os.path.join(base_dir, 'reports')
    viz_dir = os.path.join(base_dir, 'visualizations', 'm6_16')
    brain_dir = r"C:\Users\hdutt\.gemini\antigravity\brain\5a93e7ba-552e-46fb-93aa-afeae126c4e7"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    config = {
        'experiment_name': 'Phase 5.B: M6-B16 (Compression-Aware EfficientNet-B3 + Coordinate Attention, Batch 16)',
        'model_name': 'M6-B16',
        'attention_mechanism': 'Coordinate Attention (Hou et al., CVPR 2021: 1D H-pool + 1D W-pool)',
        'reduction_ratio': 32,
        'classifier': 'Linear(1536,128) -> ReLU -> Dropout(0.3) -> Linear(128,64) -> ReLU -> Dropout(0.2) -> Linear(64,1)',
        'loss_function': 'BCEWithLogitsLoss',
        'optimizer': 'Adam',
        'backbone_lr': 0.0001,
        'coordatt_lr': 0.0001,
        'classifier_lr': 0.0001,
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
        'checkpoint_selection': 'Best clean validation accuracy',
        'decision_threshold': 0.50
    }

    config_path = os.path.join(results_dir, 'm6_16_config.json')
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
    model = EfficientNetB3CoordAtt(pretrained=True, reduction=32).to(device)
    tot_params, tr_params, ntr_params = count_parameters(model)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    coordatt_params = sum(p.numel() for p in model.coord_att.parameters())
    head_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"Model Initialized: EfficientNet-B3 + Coordinate Attention (Total: {tot_params:,})")
    print(f"  Backbone (lr={config['backbone_lr']}) : {backbone_params:,} params")
    print(f"  CoordAtt (lr={config['coordatt_lr']}) : {coordatt_params:,} params [1D H/W Direction-Aware, r=32]")
    print(f"  Head     (lr={config['classifier_lr']}) : {head_params:,} params")

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    num_epochs = config['num_epochs']
    frozen_epochs = config['stage1_epochs']

    history = []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = os.path.join(models_dir, 'efficientnet_b3_coordatt_compression_aware_m6_16_best.pth')

    print("\n" + "=" * 85)
    print("STARTING M6-B16 TRAINING (10 EPOCHS, BATCH SIZE = 16, COORDINATE ATTENTION)")
    print("=" * 85)

    start_training_time = time.time()

    # 3. Training Loop
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        if epoch <= frozen_epochs:
            # Stage 1: Freeze Backbone
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.coord_att.parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True

            optimizer = Adam([
                {'params': model.coord_att.parameters(), 'lr': config['coordatt_lr']},       # 1e-4
                {'params': model.classifier.parameters(), 'lr': config['classifier_lr']}    # 1e-4
            ])
            stage_str = "STAGE 1: BACKBONE FROZEN (CoordAtt lr=1e-4, Head lr=1e-4)"
            backbone_frozen = True
        else:
            # Stage 2: Full Network Trainable
            for param in model.parameters():
                param.requires_grad = True

            optimizer = Adam([
                {'params': model.backbone.parameters(), 'lr': config['backbone_lr']},       # 1e-4
                {'params': model.coord_att.parameters(), 'lr': config['coordatt_lr']},     # 1e-4
                {'params': model.classifier.parameters(), 'lr': config['classifier_lr']}    # 1e-4
            ])
            stage_str = "STAGE 2: FULL NETWORK FINE-TUNING (Backbone lr=1e-4, CoordAtt lr=1e-4, Head lr=1e-4)"
            backbone_frozen = False

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        cond_counts = {c: 0 for c in train_dataset.conditions}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs:02d} [{stage_str[:7]}]", leave=False):
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

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation (Clean standard set)
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

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start

        is_best = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, best_model_path)
            star = " [*BEST]"
        else:
            star = ""

        history.append({
            'epoch': epoch,
            'stage': stage_str,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch_time': epoch_time,
            'is_best': is_best,
            'condition_samples': cond_counts
        })

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%{star}")

    total_training_time = time.time() - start_training_time
    print("=" * 85)
    print(f"TRAINING COMPLETE in {total_training_time/60:.2f} mins. Best Model at Epoch {best_epoch} (Val Acc: {best_val_acc*100:.2f}%)")
    print(f"Best Checkpoint: {best_model_path}")
    print("=" * 85)

    # Save training history
    history_path = os.path.join(results_dir, 'm6_16_training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    print(f"Saved history to: {history_path}")

    # Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], 'b-o', label='Train Loss')
    plt.plot([h['epoch'] for h in history], [h['val_loss'] for h in history], 'r-s', label='Val Loss')
    plt.axvline(best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('M6-B16 (Coordinate Attention): Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot([h['epoch'] for h in history], [h['train_acc']*100 for h in history], 'b-o', label='Train Acc')
    plt.plot([h['epoch'] for h in history], [h['val_acc']*100 for h in history], 'r-s', label='Val Acc')
    plt.axvline(best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('M6-B16 (Coordinate Attention): Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'm6_16_training_history.png'), dpi=300)
    plt.close()

    # 4. Multi-QF Evaluation on Frozen Test Set
    print("\n" + "=" * 85)
    print("STARTING MULTI-QF EVALUATION ON FROZEN TEST SET (2,000 IMAGES PER CONDITION)")
    print("=" * 85)

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

    for cond in test_conditions:
        c_name = cond['name']
        c_qf = cond['qf']
        print(f"Evaluating Condition: {c_name} (JPEG QF={c_qf if c_qf else 'Original'})...")

        test_ds = CompressionAwareDeepFakeDataset(
            test_csv, is_training=False, transform=eval_tx,
            fixed_condition=c_name, fixed_qf=c_qf
        )
        loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

        all_probs = []
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                image_ids = batch['image_id']
                filenames = batch['original_filename']
                classes = batch['class']

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(images)
                    probs = torch.sigmoid(logits)

                probs_np = probs.cpu().squeeze().tolist()
                if not isinstance(probs_np, list): probs_np = [probs_np]
                targets_np = labels.cpu().squeeze().tolist()
                if not isinstance(targets_np, list): targets_np = [targets_np]

                all_probs.extend(probs_np)
                all_targets.extend(targets_np)

                for i in range(len(probs_np)):
                    p = probs_np[i]
                    pred_label = "Fake" if p >= 0.5 else "Real"
                    all_preds.append(1.0 if p >= 0.5 else 0.0)
                    all_predictions.append({
                        'image_id': image_ids[i],
                        'original_filename': filenames[i],
                        'true_label': classes[i],
                        'predicted_label': pred_label,
                        'probability_fake': round(p, 6),
                        'condition': c_name,
                        'jpeg_quality': c_qf if c_qf is not None else 'NA'
                    })

        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

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

        m_dict = {
            'condition': c_name,
            'jpeg_quality': c_qf if c_qf is not None else 'NA',
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
        master_metrics.append(m_dict)

        # Save individual JSON metric file
        json_name = f"m6_16_{c_name.lower()}_metrics.json"
        with open(os.path.join(results_dir, json_name), 'w', encoding='utf-8') as f:
            json.dump(m_dict, f, indent=4)

        print(f"  Acc: {acc*100:.2f}% | Rec: {rec:.4f} | Prec: {prec:.4f} | F1: {f1:.4f} | FP: {fp}, FN: {fn}")

    # Save Predictions CSV
    pred_csv_path = os.path.join(results_dir, 'm6_16_predictions.csv')
    with open(pred_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_predictions[0].keys()))
        writer.writeheader()
        writer.writerows(all_predictions)
    print(f"Saved test predictions to: {pred_csv_path}")

    # 5. Direct Comparison: M6-B16 vs M5-B16 vs M4-B16 vs M3-B16
    print("\nComparing M6-B16 vs Baselines M5-B16 (Triplet), M4-B16 (CBAM), and M3-B16 (No Attention)...")
    m3_16_metrics = {}
    m4_16_metrics = {}
    m5_16_metrics = {}
    m6_b8_metrics = {}
    for cond in test_conditions:
        c_name = cond['name']
        c_lower = c_name.lower()
        with open(os.path.join(results_dir, f"m3_16_{c_lower}_metrics.json"), 'r', encoding='utf-8') as f:
            m3_16_metrics[c_name] = json.load(f)
        with open(os.path.join(results_dir, f"m4_16_{c_lower}_metrics.json"), 'r', encoding='utf-8') as f:
            m4_16_metrics[c_name] = json.load(f)
        with open(os.path.join(results_dir, f"m5_16_{c_lower}_metrics.json"), 'r', encoding='utf-8') as f:
            m5_16_metrics[c_name] = json.load(f)
        with open(os.path.join(results_dir, f"m6_b8_{c_lower}_metrics.json"), 'r', encoding='utf-8') as f:
            m6_b8_metrics[c_name] = json.load(f)

    comparison_records = []
    for m in master_metrics:
        c = m['condition']
        m3 = m3_16_metrics[c]
        m4 = m4_16_metrics[c]
        m5 = m5_16_metrics[c]
        m6_8 = m6_b8_metrics[c]

        comparison_records.append({
            'condition': c,
            'jpeg_quality': m['jpeg_quality'],
            'm6_16_accuracy': round(m['accuracy'], 6),
            'm5_16_accuracy': round(m5['accuracy'], 6),
            'm4_16_accuracy': round(m4['accuracy'], 6),
            'm3_16_accuracy': round(m3['accuracy'], 6),
            'm6_b8_accuracy': round(m6_8['accuracy'], 6),

            'm6_16_precision': round(m['precision'], 6),
            'm5_16_precision': round(m5['precision'], 6),
            'm4_16_precision': round(m4['precision'], 6),
            'm3_16_precision': round(m3['precision'], 6),

            'm6_16_recall': round(m['recall'], 6),
            'm5_16_recall': round(m5['recall'], 6),
            'm4_16_recall': round(m4['recall'], 6),
            'm3_16_recall': round(m3['recall'], 6),

            'm6_16_f1': round(m['f1_score'], 6),
            'm5_16_f1': round(m5['f1_score'], 6),
            'm4_16_f1': round(m4['f1_score'], 6),
            'm3_16_f1': round(m3['f1_score'], 6),

            'm6_16_roc_auc': round(m['roc_auc'], 6),
            'm5_16_roc_auc': round(m5['roc_auc'], 6),
            'm4_16_roc_auc': round(m4['roc_auc'], 6),
            'm3_16_roc_auc': round(m3['roc_auc'], 6),

            'm6_16_FP': m['FP'],
            'm5_16_FP': m5['FP'],
            'm4_16_FP': m4['FP'],
            'm3_16_FP': m3['FP'],
            'm6_b8_FP': m6_8['FP'],

            'm6_16_FN': m['FN'],
            'm5_16_FN': m5['FN'],
            'm4_16_FN': m4['FN'],
            'm3_16_FN': m3['FN'],
            'm6_b8_FN': m6_8['FN'],

            'm6_16_TP': m['TP'],
            'm6_16_TN': m['TN']
        })

    comp_csv_path = os.path.join(results_dir, 'm6_16_master_comparison.csv')
    with open(comp_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_records[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_records)
    print(f"Saved master comparison to: {comp_csv_path}")

    # 6. Diagnostic Visualizations
    print("Generating Diagnostic Figures...")
    conds = [m['condition'] for m in master_metrics]
    m6_acc = [m['accuracy']*100 for m in master_metrics]
    m5_acc = [m5_16_metrics[c]['accuracy']*100 for c in conds]
    m4_acc = [m4_16_metrics[c]['accuracy']*100 for c in conds]
    m3_acc = [m3_16_metrics[c]['accuracy']*100 for c in conds]

    plt.figure(figsize=(10.5, 5.5))
    plt.plot(conds, m3_acc, 'blue', linestyle=':', marker='s', lw=1.8, label='M3-B16: CA-B3 (No Attention)')
    plt.plot(conds, m4_acc, 'green', linestyle='--', marker='D', lw=1.8, label='M4-B16: CA-CBAM (r=16, k=7)')
    plt.plot(conds, m5_acc, '#9467bd', linestyle='-.', marker='^', lw=1.8, label='M5-B16: CA-Triplet (Z-Pool)')
    plt.plot(conds, m6_acc, '#d62728', linestyle='-', marker='o', lw=2.5, label='M6-B16: CA-CoordAtt (1D H/W r=32)')
    plt.title('Accuracy vs Compression: M3-B16 vs M4-B16 vs M5-B16 vs M6-B16 (Batch 16)', fontsize=12, fontweight='bold')
    plt.xlabel('Condition / Quality Factor', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.ylim(93.8, 98.2)
    plt.legend(fontsize=9.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    for i, a in enumerate(m6_acc):
        plt.annotate(f"{a:.2f}%", (i, a + 0.15), ha='center', fontsize=9, fontweight='bold', color='#d62728')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'm6_16_accuracy_vs_qf.png'), dpi=300)
    plt.close()

    # Plot FP and FN comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x_idx = np.arange(len(conds))
    w = 0.20

    m6_fps = [m['FP'] for m in master_metrics]
    m5_fps = [m5_16_metrics[c]['FP'] for c in conds]
    m4_fps = [m4_16_metrics[c]['FP'] for c in conds]
    m3_fps = [m3_16_metrics[c]['FP'] for c in conds]

    m6_fns = [m['FN'] for m in master_metrics]
    m5_fns = [m5_16_metrics[c]['FN'] for c in conds]
    m4_fns = [m4_16_metrics[c]['FN'] for c in conds]
    m3_fns = [m3_16_metrics[c]['FN'] for c in conds]

    axes[0].bar(x_idx - 1.5*w, m3_fps, w, label='M3-B16 (No Attention)', color='#4285F4', alpha=0.85)
    axes[0].bar(x_idx - 0.5*w, m4_fps, w, label='M4-B16 (CBAM)', color='#34A853', alpha=0.85)
    axes[0].bar(x_idx + 0.5*w, m5_fps, w, label='M5-B16 (Triplet)', color='#9467bd', alpha=0.85)
    axes[0].bar(x_idx + 1.5*w, m6_fps, w, label='M6-B16 (CoordAtt)', color='#d62728', alpha=0.95)
    axes[0].set_title('False Positives: Real Misclassified as Fake (Lower is Better)', fontweight='bold')
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(conds)
    axes[0].set_ylabel('Count (out of 1,000 Real images)')
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    axes[1].bar(x_idx - 1.5*w, m3_fns, w, label='M3-B16 (No Attention)', color='#4285F4', alpha=0.85)
    axes[1].bar(x_idx - 0.5*w, m4_fns, w, label='M4-B16 (CBAM)', color='#34A853', alpha=0.85)
    axes[1].bar(x_idx + 0.5*w, m5_fns, w, label='M5-B16 (Triplet)', color='#9467bd', alpha=0.85)
    axes[1].bar(x_idx + 1.5*w, m6_fns, w, label='M6-B16 (CoordAtt)', color='#d62728', alpha=0.95)
    axes[1].set_title('False Negatives: Missed Deepfakes (Lower is Better)', fontweight='bold')
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(conds)
    axes[1].set_ylabel('Count (out of 1,000 Fake images)')
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'm6_16_fp_fn_vs_qf.png'), dpi=300)
    plt.close()

    # Copy plots to brain directory
    for img_f in os.listdir(viz_dir):
        if img_f.endswith('.png'):
            src_f = os.path.join(viz_dir, img_f)
            dst_f = os.path.join(brain_dir, img_f)
            with open(src_f, 'rb') as sf, open(dst_f, 'wb') as df:
                df.write(sf.read())

    print("\n" + "=" * 90)
    print("PHASE 5.B SUMMARY: M6-B16 (COORDINATE ATTENTION, BATCH 16) vs M5-B16 vs M4-B16 vs M3-B16")
    print("=" * 90)
    for r in comparison_records:
        print(f"[{r['condition']:<5}] M6 Acc: {r['m6_16_accuracy']*100:.2f}% (M5: {r['m5_16_accuracy']*100:.2f}%, M4: {r['m4_16_accuracy']*100:.2f}%, M3: {r['m3_16_accuracy']*100:.2f}%) | "
              f"M6 FP: {r['m6_16_FP']} (M5: {r['m5_16_FP']}, M4: {r['m4_16_FP']}, M3: {r['m3_16_FP']}) | "
              f"M6 FN: {r['m6_16_FN']} (M5: {r['m5_16_FN']}, M4: {r['m4_16_FN']}, M3: {r['m3_16_FN']})")
    print("=" * 90)
    print("[M6-B16 Coordinate Attention Script Completed Successfully]")

if __name__ == '__main__':
    main()
