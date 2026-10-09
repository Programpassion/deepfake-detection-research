import torch
import torch.nn as nn
import timm

class EfficientNetB3Baseline(nn.Module):
    """
    Phase 2 Baseline Model: Vanilla EfficientNet-B3 with Custom 3-Stage Classification Head.
    Strictly reproducing the base paper (Deepa et al., Discover Computing 2026).
    
    Architectural Rules:
      - Native ImageNet-pretrained EfficientNet-B3 backbone (Untampered SiLU/Swish internal activations).
      - Replaces ONLY default classifier with 3-stage custom MLP head.
      - Global Average Pooling -> 1536 features.
      - Linear(1536, 128) -> ReLU -> Dropout(0.3)
      - Linear(128, 64) -> ReLU -> Dropout(0.2)
      - Linear(64, 1) -> Raw logit (Sigmoid for inference).
      
    No CBAM, No ReZero, No DCT, No extra attention or branches.
    """
    def __init__(self, pretrained=True):
        super(EfficientNetB3Baseline, self).__init__()
        # Load ImageNet-pretrained backbone with num_classes=0 to get pooled 1536 features
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features # 1536

        # 3-Stage Custom MLP Classification Head matching Deepa et al. Section 6 & 8.2
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1) # Outputs raw binary logit
        )

    def forward(self, x):
        feat = self.backbone(x) # [B, 1536] (global average pooled)
        logits = self.classifier(feat) # [B, 1]
        return logits

    def predict_proba(self, x):
        """Returns binary probability p in [0, 1] that image is Fake (Class 1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable

if __name__ == "__main__":
    model = EfficientNetB3Baseline(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    probs = model.predict_proba(dummy)
    tot, tr, ntr = count_parameters(model)
    print("=" * 60)
    print("EfficientNet-B3 Baseline Architecture Verification:")
    print(f"  Output logit shape: {out.shape}")
    print(f"  Output proba shape: {probs.shape} (Range: [{probs.min():.4f}, {probs.max():.4f}])")
    print(f"  Total parameters        : {tot:,}")
    print(f"  Trainable parameters    : {tr:,}")
    print(f"  Non-trainable parameters: {ntr:,}")
    print("=" * 60)
