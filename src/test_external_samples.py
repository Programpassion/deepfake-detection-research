"""
test_external_samples.py
Download independent, external real and fake face samples from different datasets
(FaceForensics++, HiDF, OpenFace, FaceRecognition) not used during training or evaluation.
Run manual cross-dataset compression robustness test comparing M3-B16 vs. M6-B16.
NO Grad-CAM is added as per explicit user instructions.
"""

import os
import sys
import json
import csv
import urllib.request
from io import BytesIO
from PIL import Image
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_baseline import EfficientNetB3Baseline
from model_coordatt import EfficientNetB3CoordAtt
from dataset_loader import get_phase2_transforms

EXTERNAL_SAMPLES = [
    # REAL SAMPLES (Independent external datasets: OpenFace, FaceForensics, FaceRecognition)
    {
        'id': 'EXT_REAL_01',
        'name': 'biden.jpg',
        'source_dataset': 'FaceRecognition_Benchmark',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg'
    },
    {
        'id': 'EXT_REAL_02',
        'name': 'obama.jpg',
        'source_dataset': 'FaceRecognition_Benchmark',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg'
    },
    {
        'id': 'EXT_REAL_03',
        'name': 'carell.jpg',
        'source_dataset': 'OpenFace_Celebrity_Set',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/carell.jpg'
    },
    {
        'id': 'EXT_REAL_04',
        'name': 'adams.jpg',
        'source_dataset': 'OpenFace_Celebrity_Set',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/adams.jpg'
    },
    {
        'id': 'EXT_REAL_05',
        'name': 'clapton.jpg',
        'source_dataset': 'OpenFace_Celebrity_Set',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/clapton-1.jpg'
    },
    {
        'id': 'EXT_REAL_06',
        'name': 'lennon.jpg',
        'source_dataset': 'OpenFace_Celebrity_Set',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/cmusatyalab/openface/master/images/examples/lennon-1.jpg'
    },
    {
        'id': 'EXT_REAL_07',
        'name': 'ff_original_frame.png',
        'source_dataset': 'FaceForensics++_Pristine',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_original.png'
    },
    {
        'id': 'EXT_REAL_08',
        'name': 'ff_actors_frame.png',
        'source_dataset': 'FaceForensics++_Pristine',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_original_actors.png'
    },
    {
        'id': 'EXT_REAL_09',
        'name': 'alex_lacamoire.png',
        'source_dataset': 'FaceRecognition_Benchmark',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/alex-lacamoire.png'
    },
    {
        'id': 'EXT_REAL_10',
        'name': 'lin_miranda.png',
        'source_dataset': 'FaceRecognition_Benchmark',
        'ground_truth': 'Real',
        'url': 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/lin-manuel-miranda.png'
    },

    # FAKE SAMPLES (Independent external datasets: FaceForensics++, HiDF SKKU)
    {
        'id': 'EXT_FAKE_01',
        'name': 'ff_deepfakes.png',
        'source_dataset': 'FaceForensics++_Deepfakes',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_deepfakes.png'
    },
    {
        'id': 'EXT_FAKE_02',
        'name': 'ff_deepfakedetection.png',
        'source_dataset': 'FaceForensics++_DFD',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_deepfakedetection.png'
    },
    {
        'id': 'EXT_FAKE_03',
        'name': 'ff_neuraltextures.png',
        'source_dataset': 'FaceForensics++_NeuralTextures',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_neuraltextures.png'
    },
    {
        'id': 'EXT_FAKE_04',
        'name': 'hidf_c01276.jpg',
        'source_dataset': 'HiDF_HighQuality_Deepfake',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/samples/images/c01276_c01230.jpg'
    },
    {
        'id': 'EXT_FAKE_05',
        'name': 'hidf_c01280.jpg',
        'source_dataset': 'HiDF_HighQuality_Deepfake',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/samples/images/c01280_c01245.jpg'
    },
    {
        'id': 'EXT_FAKE_06',
        'name': 'hidf_c01282.jpg',
        'source_dataset': 'HiDF_HighQuality_Deepfake',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/samples/images/c01282_c01268.jpg'
    },
    {
        'id': 'EXT_FAKE_07',
        'name': 'hidf_c01283.jpg',
        'source_dataset': 'HiDF_HighQuality_Deepfake',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/samples/images/c01283_c01242.jpg'
    },
    {
        'id': 'EXT_FAKE_08',
        'name': 'hidf_c01284.jpg',
        'source_dataset': 'HiDF_HighQuality_Deepfake',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/samples/images/c01284_c01252.jpg'
    },
    {
        'id': 'EXT_FAKE_09',
        'name': 'hidf_c01285.jpg',
        'source_dataset': 'HiDF_HighQuality_Deepfake',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/samples/images/c01285_c01249.jpg'
    },
    {
        'id': 'EXT_FAKE_10',
        'name': 'hidf_c01287.jpg',
        'source_dataset': 'HiDF_HighQuality_Deepfake',
        'ground_truth': 'Fake',
        'url': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/samples/images/c01287_c01260.jpg'
    }
]

def download_samples(sample_dir):
    os.makedirs(sample_dir, exist_ok=True)
    downloaded = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for s in EXTERNAL_SAMPLES:
        sub = 'real' if s['ground_truth'] == 'Real' else 'fake'
        target_folder = os.path.join(sample_dir, sub)
        os.makedirs(target_folder, exist_ok=True)
        local_path = os.path.join(target_folder, s['name'])
        
        if not os.path.exists(local_path):
            print(f"Downloading {s['id']}: {s['name']} from {s['source_dataset']}...")
            try:
                req = urllib.request.Request(s['url'], headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                with open(local_path, 'wb') as f:
                    f.write(data)
            except Exception as e:
                print(f"Failed to download {s['url']}: {e}")
                continue
        s_copy = dict(s)
        s_copy['local_path'] = local_path
        downloaded.append(s_copy)
    return downloaded

def predict_single_image(model, img, qf, transform, device):
    img = img.convert("RGB")
    # Apply JPEG if qf is specified
    if qf is not None:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=qf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")


    tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logit = model(tensor)
            prob = float(torch.sigmoid(logit).cpu().numpy().flatten()[0])

    pred_label = "Fake" if prob >= 0.50 else "Real"
    conf = prob if pred_label == "Fake" else (1.0 - prob)
    return {
        'fake_prob': prob,
        'pred_label': pred_label,
        'confidence': conf
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 95)
    print("OUT-OF-DISTRIBUTION CROSS-DATASET EVALUATION: M3-B16 BASELINE VS. M6-B16 COORDATT")
    print(f"Device: {device} | Autocast: {device.type == 'cuda'}")
    print("=" * 95)

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(repo_dir, "data", "external_samples")
    out_dir = os.path.join(repo_dir, "results", "external_testing")
    os.makedirs(out_dir, exist_ok=True)

    m3_path = os.path.join(repo_dir, "models", "efficientnet_b3_baseline_m3_16_best.pth")
    m6_path = os.path.join(repo_dir, "models", "efficientnet_b3_coordatt_compression_aware_m6_16_best.pth")

    _, eval_tx = get_phase2_transforms()

    # Load models
    print(f"Loading M3 Baseline from {os.path.basename(m3_path)}...")
    m3 = EfficientNetB3Baseline(pretrained=False)
    m3.load_state_dict(torch.load(m3_path, map_location=device)['model_state_dict'])
    m3.to(device)

    print(f"Loading M6 CoordAtt from {os.path.basename(m6_path)}...")
    m6 = EfficientNetB3CoordAtt(pretrained=False)
    m6.load_state_dict(torch.load(m6_path, map_location=device)['model_state_dict'])
    m6.to(device)

    # Download external test samples
    samples = download_samples(samples_dir)
    print(f"\nTotal external test samples verified: {len(samples)} (Real: {sum(1 for s in samples if s['ground_truth']=='Real')}, Fake: {sum(1 for s in samples if s['ground_truth']=='Fake')})")

    conditions = [
        ('Clean', None),
        ('QF80', 80),
        ('QF60', 60),
        ('QF50', 50),
        ('QF40', 40),
        ('QF20', 20)
    ]

    detailed_results = []
    
    print("\n" + "=" * 95)
    print(f"{'Sample ID':<11} | {'Source Dataset':<25} | {'GT':<5} | {'QF':<5} | {'M3 Pred':<7} | {'M3 P(Fake)':<10} | {'M6 Pred':<7} | {'M6 P(Fake)':<10} | {'Status'}")
    print("-" * 95)

    for s in samples:
        orig_img = Image.open(s['local_path'])
        gt = s['ground_truth']

        for c_name, qf in conditions:
            # Predict M3
            m3_out = predict_single_image(m3, orig_img, qf, eval_tx, device)
            # Predict M6
            m6_out = predict_single_image(m6, orig_img, qf, eval_tx, device)

            m3_correct = (m3_out['pred_label'] == gt)
            m6_correct = (m6_out['pred_label'] == gt)

            status = ""
            if m6_correct and not m3_correct:
                status = "M6 WINS (M3 Failed)"
            elif m3_correct and not m6_correct:
                status = "M3 WINS (M6 Failed)"
            elif m6_correct and m3_correct:
                status = "BOTH CORRECT"
            else:
                status = "BOTH FAILED"

            row = {
                'sample_id': s['id'],
                'image_name': s['name'],
                'source_dataset': s['source_dataset'],
                'ground_truth': gt,
                'condition': c_name,
                'qf': qf if qf is not None else 100,
                'm3_pred': m3_out['pred_label'],
                'm3_fake_prob_pct': f"{m3_out['fake_prob']*100:.2f}%",
                'm3_confidence_pct': f"{m3_out['confidence']*100:.2f}%",
                'm3_correct': m3_correct,
                'm6_pred': m6_out['pred_label'],
                'm6_fake_prob_pct': f"{m6_out['fake_prob']*100:.2f}%",
                'm6_confidence_pct': f"{m6_out['confidence']*100:.2f}%",
                'm6_correct': m6_correct,
                'status': status
            }
            detailed_results.append(row)

            # Print concise summary line for Clean, QF50, QF20 to keep console legible
            if c_name in ['Clean', 'QF50', 'QF20']:
                print(f"{s['id']:<11} | {s['source_dataset']:<25} | {gt:<5} | {c_name:<5} | {m3_out['pred_label']:<7} | {m3_out['fake_prob']*100:6.2f}%    | {m6_out['pred_label']:<7} | {m6_out['fake_prob']*100:6.2f}%    | {status}")

    # Compute summary across conditions
    print("\n" + "=" * 95)
    print("CROSS-DATASET ACCURACY BREAKDOWN BY COMPRESSION QUALITY FACTOR")
    print("=" * 95)
    print(f"{'Condition':<10} | {'M3 Baseline Acc':<18} | {'M6 CoordAtt Acc':<18} | {'M3 Correct':<12} | {'M6 Correct':<12} | {'Advantage'}")
    print("-" * 95)

    condition_summary = []
    for c_name, _ in conditions:
        sub = [r for r in detailed_results if r['condition'] == c_name]
        m3_acc = sum(1 for r in sub if r['m3_correct']) / len(sub) * 100.0
        m6_acc = sum(1 for r in sub if r['m6_correct']) / len(sub) * 100.0
        m3_c = sum(1 for r in sub if r['m3_correct'])
        m6_c = sum(1 for r in sub if r['m6_correct'])

        diff = m6_acc - m3_acc
        adv = f"M6 +{diff:.1f} pp" if diff > 0 else (f"M3 +{-diff:.1f} pp" if diff < 0 else "TIED")
        print(f"{c_name:<10} | {m3_acc:6.1f}% ({m3_c}/{len(sub)})    | {m6_acc:6.1f}% ({m6_c}/{len(sub)})    | {m3_c:>4d}/{len(sub)}     | {m6_c:>4d}/{len(sub)}     | {adv}")

        # FN breakdown on Fakes
        fake_sub = [r for r in sub if r['ground_truth'] == 'Fake']
        m3_fn = sum(1 for r in fake_sub if not r['m3_correct'])
        m6_fn = sum(1 for r in fake_sub if not r['m6_correct'])
        # FP breakdown on Reals
        real_sub = [r for r in sub if r['ground_truth'] == 'Real']
        m3_fp = sum(1 for r in real_sub if not r['m3_correct'])
        m6_fp = sum(1 for r in real_sub if not r['m6_correct'])

        condition_summary.append({
            'condition': c_name,
            'total_samples': len(sub),
            'm3_accuracy': f"{m3_acc:.1f}%",
            'm6_accuracy': f"{m6_acc:.1f}%",
            'm3_fn': m3_fn,
            'm6_fn': m6_fn,
            'm3_fp': m3_fp,
            'm6_fp': m6_fp
        })

    # Save detailed CSV
    csv_path = os.path.join(out_dir, "external_samples_m3_vs_m6_predictions.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_results[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_results)
    print(f"\nSaved detailed sample predictions to: {csv_path}")

    # Save summary JSON
    summary_path = os.path.join(out_dir, "external_samples_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'condition_summary': condition_summary,
            'total_samples_evaluated': len(samples),
            'real_count': sum(1 for s in samples if s['ground_truth']=='Real'),
            'fake_count': sum(1 for s in samples if s['ground_truth']=='Fake')
        }, f, indent=2)
    print(f"Saved summary JSON to: {summary_path}")

    print("\n" + "=" * 95)
    print("MANUAL EXTERNAL TEST COMPLETE! WAITING FOR YOUR COMMAND.")
    print("=" * 95)

if __name__ == '__main__':
    main()
