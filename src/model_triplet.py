import torch
import torch.nn as nn
import timm

class BasicConv(nn.Module):
    """
    Standard Convolution + BatchNorm layer for Triplet Attention.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        return self.bn(self.conv(x))

class ZPool(nn.Module):
    """
    Z-Pool layer: concatenates max-pooling and average-pooling across channel dimension (dim=1).
    Transforms tensor from (B, C, H, W) to (B, 2, H, W).
    """
    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, avg_pool), dim=1)

class AttentionGate(nn.Module):
    """
    Attention Gate applying Z-pooling followed by a 7x7 standard convolution and Sigmoid gating.
    """
    def __init__(self, kernel_size=7):
        super(AttentionGate, self).__init__()
        padding = (kernel_size - 1) // 2
        self.compress = ZPool()
        self.conv = BasicConv(in_planes=2, out_planes=1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        scale = self.sigmoid(self.conv(x_compress))
        return x * scale

class TripletAttention(nn.Module):
    """
    Triplet Attention Module (Misra et al., WACV 2021: 'Rotate to Attend: Convolutional Triplet Attention Module').
    
    Captures cross-dimension interactions using 3 parallel rotation branches:
      1. Branch 1: Channel & Width (C, W) -> Permute to (B, H, C, W), attend, permute back
      2. Branch 2: Channel & Height (C, H) -> Permute to (B, W, H, C), attend, permute back
      3. Branch 3: Spatial (H, W) -> Attend on (B, C, H, W)
      
    Final output is the element-wise average of all three branches:
      Y = (Y_cw + Y_hc + Y_hw) / 3.0
    """
    def __init__(self, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate(kernel_size)
        self.hc = AttentionGate(kernel_size)
        self.hw = AttentionGate(kernel_size)

    def forward(self, x):
        # Branch 1: Channel & Width interaction
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        out_cw = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()

        # Branch 2: Channel & Height interaction
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        out_hc = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()

        # Branch 3: Spatial (H, W) interaction
        out_hw = self.hw(x)

        return (out_cw + out_hc + out_hw) / 3.0

class EfficientNetB3Triplet(nn.Module):
    """
    Model 3: EfficientNet-B3 + Triplet Attention Architecture.
    
    Backbone: EfficientNet-B3 (pretrained on ImageNet via timm)
    Attention: Triplet Attention applied to final 1536-dimensional feature map (7x7 resolution)
    Classifier Head (Base Paper 3-Stage MLP):
        Linear Layer 1: 1536 -> 128 with ReLU
        Dropout (rate = 0.3)
        Linear Layer 2: 128 -> 64 with ReLU
        Dropout (rate = 0.2)
        Linear Layer 3: 64 -> 1 (Binary Real vs Fake logit)
    """
    def __init__(self, pretrained=True):
        super(EfficientNetB3Triplet, self).__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features # 1536
        
        # Insert Triplet Attention
        self.triplet = TripletAttention(kernel_size=7)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 3-Stage Custom MLP Head matching base paper
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1) # Output raw logit
        )

    def forward(self, x):
        feat_map = self.backbone.forward_features(x)
        refined_map = self.triplet(feat_map)
        pooled = self.global_pool(refined_map).flatten(1)
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, x):
        """Returns probability p in [0, 1] that the image is Fake (class 1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

if __name__ == "__main__":
    model = EfficientNetB3Triplet(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("EfficientNet-B3 + Triplet Attention forward pass successful!")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
