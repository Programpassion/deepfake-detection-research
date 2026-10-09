"""
model_coordatt.py
Standard PyTorch implementation of Coordinate Attention (Hou et al., CVPR 2021)
and integration with ImageNet-pretrained EfficientNet-B3 for Phase 5.A (M6-B8).

Reference:
  "Coordinate Attention for Efficient Mobile Network Design"
  Qibin Hou, Daquan Zhou, Jiashi Feng (CVPR 2021)
  Official Repo: https://github.com/Andrew-Qibin/CoordAttention
"""

import torch
import torch.nn as nn
import timm

class h_sigmoid(nn.Module):
    """
    Hard-Sigmoid non-linear activation: ReLU6(x + 3) / 6
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    """
    Hard-Swish non-linear activation: x * h_sigmoid(x)
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    """
    Coordinate Attention Module (Hou et al., CVPR 2021)
    
    Decomposes channel attention into two 1D feature encoding processes
    aggregating features along the horizontal and vertical directions respectively:
    - pool_h: AdaptiveAvgPool2d((None, 1)) -> [B, C, H, 1]
    - pool_w: AdaptiveAvgPool2d((1, None)) -> [B, C, 1, W]
    
    Shared 1x1 conv with reduction ratio r=32 captures spatial-coordinate dependencies,
    followed by split conv_h and conv_w yielding direction-aware attention maps.
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class EfficientNetB3CoordAtt(nn.Module):
    """
    M6 Proposed Architecture:
    ImageNet-pretrained EfficientNet-B3 with Coordinate Attention inserted
    immediately before Global Average Pooling (GAP), followed by the locked
    canonical 3-stage custom classifier head.

    Data Flow:
      Input [B, 3, 224, 224]
        -> EfficientNet-B3 Backbone (forward_features)
        -> Feature Map [B, 1536, 7, 7]
        -> Coordinate Attention (1D H + 1D W pools, r=32) [B, 1536, 7, 7]
        -> Global Average Pooling [B, 1536]
        -> Classifier Head: Linear(1536->128) -> ReLU -> Dropout(0.30)
                            -> Linear(128->64) -> ReLU -> Dropout(0.20)
                            -> Linear(64->1) -> Logit
    """
    def __init__(self, pretrained=True, reduction=32, dropout1=0.3, dropout2=0.2):
        super(EfficientNetB3CoordAtt, self).__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features  # 1536

        # Standard Coordinate Attention Module before GAP
        self.coord_att = CoordAtt(inp=in_features, oup=in_features, reduction=reduction)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Locked Canonical Classifier Head (0.30 / 0.20 dropouts)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        feat_attn = self.coord_att(feat)
        pooled = self.gap(feat_attn).flatten(1)
        logits = self.classifier(pooled)
        return logits
