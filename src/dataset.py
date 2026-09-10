import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DeepFakeDataset(Dataset):
    """
    DeepFake & Real Images Dataset Loader.
    Labels: Real = 0.0, Fake = 1.0 (Matching Section 7, Step 2 & Section 8.3 of base paper).
    """
    def __init__(self, root_dir, split="Train", samples_per_class=None, transform=None, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        split_dir = os.path.join(root_dir, split)
        real_dir = os.path.join(split_dir, "Real")
        fake_dir = os.path.join(split_dir, "Fake")
        
        # Supported image extensions
        valid_exts = ('.jpg', '.jpeg', '.png')
        
        # Collect file paths
        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(valid_exts)]
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(valid_exts)]
        
        # Sort for reproducibility
        real_files.sort()
        fake_files.sort()
        
        # Apply balanced stratified sampling if specified
        if samples_per_class is not None and samples_per_class > 0:
            rng = random.Random(seed)
            if len(real_files) > samples_per_class:
                real_files = rng.sample(real_files, samples_per_class)
            if len(fake_files) > samples_per_class:
                fake_files = rng.sample(fake_files, samples_per_class)
                
        self.samples = [(p, 0.0) for p in real_files] + [(p, 1.0) for p in fake_files]
        # Shuffle across real and fake
        random.Random(seed).shuffle(self.samples)
        
        print(f"[{split}] Loaded {len(real_files)} Real, {len(fake_files)} Fake (Total: {len(self.samples)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)

def get_transforms():
    """
    Data Transformations matching Section 4 & 8.1:
    - Resize to 224x224
    - Random Horizontal Flip (p=0.5)
    - Random Rotation (+-30 degrees)
    - Normalize with ImageNet mean & std
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.244, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(root_dir, train_samples_per_class=3000, test_samples_per_class=1000, batch_size=8, num_workers=2):
    train_transform, eval_transform = get_transforms()
    
    train_dataset = DeepFakeDataset(root_dir, split="Train", samples_per_class=train_samples_per_class, transform=train_transform)
    val_dataset = DeepFakeDataset(root_dir, split="Validation", samples_per_class=500, transform=eval_transform)
    test_dataset = DeepFakeDataset(root_dir, split="Test", samples_per_class=test_samples_per_class, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
