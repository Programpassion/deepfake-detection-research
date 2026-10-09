"""
infer.py
Inference Script for Coord-EfficientNet-B3 (M6-B16) and Base EfficientNet-B3 (M3-B16).
Usage:
    python src/infer.py --image path/to/image.jpg --model m6
"""

import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import EfficientNetB3Baseline
from model_coordatt import EfficientNetB3CoordAtt
from dataset_loader import get_phase2_transforms

def load_model(model_type='m6', checkpoint_path=None, device='cpu'):
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_type.lower() == 'm6':
        model = EfficientNetB3CoordAtt(pretrained=False)
        default_ckpt = os.path.join(repo_dir, "models", "efficientnet_b3_coordatt_compression_aware_m6_16_best.pth")
    else:
        model = EfficientNetB3Baseline(pretrained=False)
        default_ckpt = os.path.join(repo_dir, "models", "efficientnet_b3_baseline_m3_16_best.pth")

    ckpt_path = checkpoint_path if checkpoint_path else default_ckpt
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device='cpu'):
    _, eval_tx = get_phase2_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = eval_tx(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    prediction = "Fake" if prob >= 0.50 else "Real"
    confidence = prob if prob >= 0.50 else (1.0 - prob)

    return {
        'prediction': prediction,
        'probability_fake': prob,
        'confidence': confidence
    }

def main():
    parser = argparse.ArgumentParser(description="DeepFake Inference with M3 / M6 models")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--model', type=str, default='m6', choices=['m3', 'm6'], help="Model type: m6 (proposed) or m3 (baseline)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Optional custom checkpoint path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, args.checkpoint, device)
    res = predict_image(args.image, model, device)

    print("=" * 50)
    print(f"Model: {args.model.upper()} | Input: {args.image}")
    print(f"Prediction : {res['prediction'].upper()}")
    print(f"Fake Prob  : {res['probability_fake']*100:.2f}%")
    print(f"Confidence : {res['confidence']*100:.2f}%")
    print("=" * 50)

if __name__ == '__main__':
    main()
