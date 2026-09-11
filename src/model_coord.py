import torch
import torch.nn as nn
import timm

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design (Hou et al., CVPR 2021).
    
    Encodes spatial coordinate information into channel representations by factorizing
    2D global pooling into two 1D spatial pooling operations:
      - 1D Horizontal pooling: captures long-range dependencies along X-axis
      - 1D Vertical pooling: captures long-range dependencies along Y-axis
      
    This enables the model to accurately locate boundary manipulations and micro-blending
    artifacts even under aggressive spatial degradation.
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 1D Directional Pooling
        x_h = self.pool_h(x)                      # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # Spatial Concatenation along coordinate axis
        y = torch.cat([x_h, x_w], dim=2)          # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)                           # (B, mip, H+W, 1)
        
        # Split back into Directional Attention Vectors
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)             # (B, mip, 1, W)

        # Coordinate Attention Weights via Sigmoid
        a_h = torch.sigmoid(self.conv_h(x_h))     # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w))     # (B, C, 1, W)

        # Apply Coordinate Attention
        out = identity * a_h * a_w
        return out

class CoordEfficientNetB3(nn.Module):
    """
    Model 4: Proposed Coordinate Attention EfficientNet-B3 (Coord-EfficientNet-B3).
    
    Backbone: EfficientNet-B3 (pretrained on ImageNet via timm)
    Attention: Coordinate Attention module applied to 1536-dimensional feature map (7x7 resolution)
    Classifier Head (Base Paper 3-Stage MLP):
        Linear Layer 1: 1536 -> 128 with ReLU
        Dropout (rate = 0.3)
        Linear Layer 2: 128 -> 64 with ReLU
        Dropout (rate = 0.2)
        Linear Layer 3: 64 -> 1 (Binary Real vs Fake logit)
    """
    def __init__(self, pretrained=True, reduction=32):
        super(CoordEfficientNetB3, self).__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features # 1536
        
        # Insert Coordinate Attention
        self.coord_att = CoordAttention(in_channels=in_features, out_channels=in_features, reduction=reduction)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 3-Stage Custom MLP Head strictly matching base paper
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1) # Raw logit
        )

    def forward(self, x):
        feat_map = self.backbone.forward_features(x)
        refined_map = self.coord_att(feat_map)
        pooled = self.global_pool(refined_map).flatten(1)
        logits = self.classifier(pooled)
        return logits

    def predict_proba(self, x):
        """Returns probability p in [0, 1] that the image is Fake (class 1)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

if __name__ == "__main__":
    model = CoordEfficientNetB3(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Coord-EfficientNet-B3 forward pass successful!")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    base_params = 10908449
    diff = total_params - base_params
    print(f"Added parameters over baseline: +{diff:,} (+{diff / base_params * 100:.2f}%)")
