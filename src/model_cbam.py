import torch
import torch.nn as nn
import timm

class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module with SiLU activation matching EfficientNet-B3.
    Squeezes spatial dimensions via AdaptiveAvgPool and AdaptiveMaxPool, passes through a shared MLP
    with reduction ratio r, and applies Sigmoid gating.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_planes = max(8, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.SiLU(inplace=True), # Native EfficientNet-B3 activation
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module with BatchNorm stabilization.
    Applies AvgPool and MaxPool along channel dimension, concatenates them (2 channels),
    and applies a 7x7 convolution + BatchNorm followed by Sigmoid gating.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.bn(self.conv(x_cat)))
        return out

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) with Residual Attention Formulation (Wang et al., CVPR 2017).
    
    Prevents the 73% signal suppression defect by preserving the ImageNet feature trunk via identity branch:
        out = x * (1.0 + ChannelAttn(x) * SpatialAttn(x))
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        att = self.ca(x) * self.sa(x)
        # Residual gating: Identity flows at 1.0, attention amplifies discriminative manipulation cues
        out = x * (1.0 + att)
        return out

class EfficientNetB3CBAM(nn.Module):
    """
    Model 2: EfficientNet-B3 + Residual CBAM Architecture.
    
    Backbone: EfficientNet-B3 (pretrained on ImageNet via timm)
    Attention: Residual CBAM applied to final 1536-dimensional feature map (7x7 spatial resolution)
    Classifier Head (strictly matching Base Paper 3-Stage MLP):
        Linear Layer 1: 1536 -> 128 with ReLU
        Dropout (rate = 0.3)
        Linear Layer 2: 128 -> 64 with ReLU
        Dropout (rate = 0.2)
        Linear Layer 3: 64 -> 1 (Binary Real vs Fake logit)
    """
    def __init__(self, pretrained=True):
        super(EfficientNetB3CBAM, self).__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features # 1536 for B3
        
        # Insert Residual CBAM at the final feature map bottleneck
        self.cbam = CBAM(in_planes=in_features, ratio=16, kernel_size=7)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 3-Stage Custom MLP Classifier Head matching base paper
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
        # Extract 2D feature map from backbone: Shape (B, 1536, 7, 7)
        feat_map = self.backbone.forward_features(x)
        # Refine feature map using Residual Channel + Spatial attention
        refined_map = self.cbam(feat_map)
        # Pool to 1D vector: (B, 1536)
        pooled = self.global_pool(refined_map).flatten(1)
        # Classify through 3-stage custom MLP
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, x):
        """Returns probability p in [0, 1] that the image is Fake (class 1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

if __name__ == "__main__":
    model = EfficientNetB3CBAM(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Residual EfficientNet-B3 + CBAM forward pass successful!")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
