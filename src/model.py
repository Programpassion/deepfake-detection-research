import torch
import torch.nn as nn
import timm

class EfficientNetB3Base(nn.Module):
    """
    Base DeepFake Detection Model matching Deepa et al. (Discover Computing, 2026).
    
    Backbone: EfficientNet-B3 pretrained on ImageNet (via timm)
    Classifier Head (Section 8.2 & Page 14):
        Linear Layer 1: 1536 -> 128 with ReLU
        Dropout (rate = 0.3)
        Linear Layer 2: 128 -> 64 with ReLU
        Dropout (rate = 0.2)
        Linear Layer 3: 64 -> 1 (Binary Real vs Fake logit)
    """
    def __init__(self, pretrained=True):
        super(EfficientNetB3Base, self).__init__()
        # Load EfficientNet-B3 backbone without classifier (num_classes=0 gives pooled 1536-D feature vector)
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features # 1536 for B3
        
        # Custom classification head strictly matching Section 8.2 & Page 14
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1) # Raw logit for BCEWithLogitsLoss
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x):
        """Returns probability p in [0, 1] that the image is Fake (class 1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

if __name__ == "__main__":
    model = EfficientNetB3Base(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Model forward pass successful!")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
