import os
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import EfficientNetB3Base
from dataset import get_dataloaders

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in tqdm(loader, desc="Training", leave=False):
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
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", leave=False):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train Baseline EfficientNet-B3 on DeepFake Dataset")
    parser.add_argument("--data_dir", type=str, default=".", help="Root dataset directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--samples_per_class", type=int, default=3000, help="Number of training samples per class")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs matching base paper (default: 10)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size matching base paper (default: 8)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate matching Section 8.4 (default: 0.0001)")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print("\n--- Loading Dataset ---")
    train_loader, val_loader, _ = get_dataloaders(
        root_dir=args.data_dir,
        train_samples_per_class=args.samples_per_class,
        test_samples_per_class=1000,
        batch_size=args.batch_size,
        num_workers=0 # Set to 0 for Windows compatibility
    )

    print("\n--- Initializing Model ---")
    model = EfficientNetB3Base(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params / 1e6:.2f}M")

    # Training Configuration matching base paper
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = os.path.join(args.models_dir, "efficientnet_b3_base.pth")

    print(f"\n--- Starting 10-Epoch Training (Batch Size: {args.batch_size}, LR: {args.lr}) ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val Acc: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes. Best Val Acc: {best_val_acc:.2f}%")

    # Plot & Save Training Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, args.epochs + 1), history['val_loss'], label='Val Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), history['train_acc'], label='Train Acc', marker='o')
    plt.plot(range(1, args.epochs + 1), history['val_acc'], label='Val Acc', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    curves_path = os.path.join(args.models_dir, "training_curves_base.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {curves_path}")

if __name__ == "__main__":
    main()
