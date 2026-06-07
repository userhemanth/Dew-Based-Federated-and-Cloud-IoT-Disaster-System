# src/train_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =====================================================================
# Option A (NEW): MobileNetV3-Small (Dew Layer)
#   - High efficiency, tiny parameter count
#   - Perfect for True Edge inference and Federated Learning sync
# =====================================================================
class DisasterMobileNet(nn.Module):
    """
    MobileNetV3-Small tailored for Dew Layer (mobile devices).
    """
    def __init__(self, num_classes=9, pretrained=True):
        super(DisasterMobileNet, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v3_small(weights=weights)

        # Replace classification head
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# =====================================================================
# Option B: Improved custom CNN (lightweight alternative)
#   - Use this if pretrained weights are unavailable (offline env)
#   - Added BatchNorm + Dropout for regularization
# =====================================================================
class DisasterCNN_Lite(nn.Module):
    """
    Improved lightweight CNN with BatchNorm and Dropout.
    Use when pretrained ResNet is not available (fully offline).
    """
    def __init__(self, num_classes=9):
        super(DisasterCNN_Lite, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

        # After 4 pooling ops: 128 → 64 → 32 → 16 → 8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 128 → 64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 64  → 32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 32  → 16
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # 16  → 8
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# =====================================================================
# Option C: Tri-Model ViT Ensemble + Meta Classifier
#   - EfficientNet-B4 + ConvNeXt-Tiny + ViT-B/32
#   - Learned Meta-Classifier instead of soft-voting
# =====================================================================
class DisasterEnsemble(nn.Module):
    """
    Ensemble of EfficientNet-B4, ConvNeXt-Tiny, and ViT-B/32.
    Uses a Meta-Classifier to learn the optimal weighted combination of logits.
    """
    def __init__(self, num_classes=9, pretrained=True):
        super(DisasterEnsemble, self).__init__()
        
        # 1. EfficientNet-B4
        eff_weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.eff_net = models.efficientnet_b4(weights=eff_weights)
        in_features_eff = self.eff_net.classifier[1].in_features
        self.eff_net.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features_eff, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        # 2. ConvNeXt-Tiny
        convnext_weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.convnext = models.convnext_tiny(weights=convnext_weights)
        in_features_conv = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features_conv, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        # 3. Vision Transformer (ViT-B/32)
        vit_weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
        self.vit = models.vit_b_32(weights=vit_weights)
        in_features_vit = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Linear(in_features_vit, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        # Meta Classifier (Learned Weighted Ensemble)
        self.meta_classifier = nn.Linear(3 * num_classes, num_classes)
        
    def forward(self, x):
        logits1 = self.eff_net(x)
        logits2 = self.convnext(x)
        logits3 = self.vit(x)
        
        # Concatenate logits from all 3 models
        combined = torch.cat((logits1, logits2, logits3), dim=1)
        
        # Meta-Classifier outputs final prediction
        return self.meta_classifier(combined)

