import os
import csv
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Phase2DeepFakeDataset(Dataset):
    """
    Phase 2 Dataset Class reading strictly from train.csv, val.csv, or test.csv.
    
    Conditions supported:
      - 'clean': Original source image -> Resize to 224x224 -> Normalized.
      - 'qf50' : Original source image -> JPEG QF50 in-memory buffer -> Resize to 224x224 -> Normalized.
    """
    def __init__(self, csv_path, condition='clean', qf=50, transform=None):
        self.csv_path = csv_path
        self.condition = condition
        self.qf = qf
        self.transform = transform
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

        real_count = sum(1 for r in self.records if r['class'] == 'Real')
        fake_count = sum(1 for r in self.records if r['class'] == 'Fake')
        split_name = self.records[0]['split'] if self.records else "Empty"
        print(f"[{split_name} Dataset ({condition.upper()})] Loaded {len(self.records)} images "
              f"({real_count} Real, {fake_count} Fake) from {os.path.basename(csv_path)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image = Image.open(item['path']).convert("RGB")

        if self.condition == 'qf50' or (isinstance(self.qf, int) and self.qf < 100 and self.condition != 'clean'):
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=self.qf)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor([item['label']], dtype=torch.float32),
            'image_id': item['image_id'],
            'original_filename': item['original_filename'],
            'class': item['class']
        }

def get_phase2_transforms():
    """
    Standard preprocessing matching base paper:
      - Resize to 224x224
      - Normalization: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      - Training Augmentation: Random horizontal flip (p=0.5), Random rotation (+-30 deg)
      - Eval Augmentation: None (deterministic resize + norm)
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform

if __name__ == "__main__":
    base_dir = r"C:\Users\hdutt\Desktop\compressed images deepfake detection research\data"
    tr_tx, ev_tx = get_phase2_transforms()
    ds_train = Phase2DeepFakeDataset(os.path.join(base_dir, "train.csv"), condition='clean', transform=tr_tx)
    ds_val = Phase2DeepFakeDataset(os.path.join(base_dir, "val.csv"), condition='clean', transform=ev_tx)
    ds_test_clean = Phase2DeepFakeDataset(os.path.join(base_dir, "test.csv"), condition='clean', transform=ev_tx)
    ds_test_qf50 = Phase2DeepFakeDataset(os.path.join(base_dir, "test.csv"), condition='qf50', qf=50, transform=ev_tx)

    b = ds_train[0]
    print(f"Sample Train Item: Image={b['image'].shape}, Label={b['label'].item()}, ID={b['image_id']}")
    b_q = ds_test_qf50[0]
    print(f"Sample QF50 Test Item: Image={b_q['image'].shape}, Label={b_q['label'].item()}, ID={b_q['image_id']}")
    print("Phase 2 dataset loaders successfully verified!")
