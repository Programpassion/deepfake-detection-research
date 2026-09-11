import os
import sys
import time
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_cbam import EfficientNetB3CBAM

class InMemoryDeepFakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)

def build_80_20_loaders(data_dir, total_per_class=5000, batch_size=8, seed=42):
    train_dir = os.path.join(data_dir, "Train")
    real_dir = os.path.join(train_dir, "Real")
    fake_dir = os.path.join(train_dir, "Fake")

    valid_exts = ('.jpg', '.jpeg', '.png')
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(valid_exts)]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(valid_exts)]

    real_files.sort()
    fake_files.sort()

    rng = random.Random(seed)
    sampled_real = rng.sample(real_files, total_per_class)
    sampled_fake = rng.sample(fake_files, total_per_class)

    all_paths = sampled_real + sampled_fake
    all_labels = [0.0] * total_per_class + [1.0] * total_per_class

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=seed
    )

    train_samples = list(zip(train_paths, train_labels))
    test_samples = list(zip(test_paths, test_labels))

    print(f"Dataset Pool: {len(all_paths)} total ({total_per_class} Real, {total_per_class} Fake)")
    print(f"  -> 80% Training Set  : {len(train_samples)} images ({int(train_labels.count(0.0))} Real, {int(train_labels.count(1.0))} Fake)")
    print(f"  -> 20% Holdout Set   : {len(test_samples)} images ({int(test_labels.count(0.0))} Real, {int(test_labels.count(1.0))} Fake)")

    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.244, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_loader = DataLoader(InMemoryDeepFakeDataset(train_samples, train_transform), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(InMemoryDeepFakeDataset(test_samples, test_transform), batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in tqdm(loader, desc="Training CBAM", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, (correct / total) * 100.0

def evaluate_holdout(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_targets, all_probs = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.squeeze().cpu().numpy().tolist())
            all_targets.extend(targets.squeeze().cpu().numpy().tolist())
            preds = (probs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = (correct / total) * 100.0
    return running_loss / total, acc, np.array(all_targets), np.array(all_probs)

def main():
    data_dir = "C:\\Users\\hdutt\\Desktop\\Deep Fake Detection Research"
    models_dir = os.path.join(data_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print("\n--- Creating 80:20 Internal Stratified Split ---")
    train_loader, test_loader = build_80_20_loaders(data_dir, total_per_class=5000, batch_size=8, seed=42)

    print("\n--- Initializing EfficientNet-B3 + CBAM Model ---")
    model = EfficientNetB3CBAM(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    epochs = 10
    best_acc = 0.0
    best_model_path = os.path.join(models_dir, "efficientnet_b3_cbam_8020.pth")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Starting 10-Epoch Training for EfficientNet-B3 + CBAM ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = evaluate_holdout(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Holdout Loss: {val_loss:.4f}, Holdout Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path} (Acc: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nCBAM Training Complete in {total_time/60:.2f} minutes. Best Accuracy: {best_acc:.2f}%")

    # Final Evaluation of the Saved Best Checkpoint
    print(f"\n--- Final Evaluation of Best Checkpoint ({best_model_path}) ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    _, final_acc, targets, probs = evaluate_holdout(model, test_loader, criterion, device)
    preds = (probs >= 0.5).astype(int)

    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    auc = roc_auc_score(targets, probs)
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()

    # Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - EfficientNet-B3 + CBAM (80:20 Split)')
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
    cm_path = os.path.join(models_dir, "confusion_matrix_cbam_8020.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save Curves Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Holdout Loss', marker='s')
    plt.title('Loss Curves - EfficientNet-B3 + CBAM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Acc', marker='o')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Holdout Acc', marker='s')
    plt.title('Accuracy Curves - EfficientNet-B3 + CBAM')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    curves_path = os.path.join(models_dir, "training_curves_cbam_8020.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()

    print("\n" + "="*76)
    print("      MODEL 2: EFFICIENTNET-B3 + CBAM - FINAL EVALUATION REPORT")
    print("="*76)
    print(f"Total Test/Holdout Samples    : {len(targets)} (Real: 1000, Fake: 1000)")
    print(f"True Positives (Fake -> Fake) : {tp}")
    print(f"True Negatives (Real -> Real) : {tn}")
    print(f"False Positives (Real -> Fake): {fp}")
    print(f"False Negatives (Fake -> Real): {fn}")
    print(f"Overall Accuracy              : {final_acc:.2f}%")
    print(f"Precision                     : {prec:.4f}")
    print(f"Recall                        : {rec:.4f}")
    print(f"F1-Score                      : {f1:.4f}")
    print(f"ROC-AUC Score                 : {auc:.4f}")
    print("="*76)

    print("\n--- DIRECT COMPARISON: BASE MODEL vs. EFFICIENTNET-B3 + CBAM ---")
    print(f"{'Metric':<25} | {'Base Model (80:20 Replicated)':<30} | {'EfficientNet-B3 + CBAM':<22}")
    print("-" * 82)
    print(f"{'Overall Accuracy':<25} | {'98.00%':<30} | {f'{final_acc:.2f}%':<22}")
    print(f"{'Precision':<25} | {'0.9829':<30} | {f'{prec:.4f}':<22}")
    print(f"{'Recall':<25} | {'0.9770':<30} | {f'{rec:.4f}':<22}")
    print(f"{'F1-Score':<25} | {'0.9799':<30} | {f'{f1:.4f}':<22}")
    print(f"{'True Positives (Fake)':<25} | {'977 (out of 1000)':<30} | {f'{tp} (out of 1000)':<22}")
    print(f"{'True Negatives (Real)':<25} | {'983 (out of 1000)':<30} | {f'{tn} (out of 1000)':<22}")
    print(f"{'ROC-AUC Score':<25} | {'0.9961':<30} | {f'{auc:.4f}':<22}")
    print("="*82)

if __name__ == "__main__":
    main()
