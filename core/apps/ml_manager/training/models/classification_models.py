"""
Classification models based on segmentation architectures
Adapts existing U-Net architectures for classification tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet as MonaiUNet

try:
    from .resunet_model import ResUNet, DeepResUNet
except ImportError:
    from resunet_model import ResUNet, DeepResUNet


class UNetClassifier(nn.Module):
    """U-Net based classifier - uses U-Net encoder with classification head"""
    
    def __init__(self, input_channels=3, num_classes=2, n_channels=None, n_classes=None, use_monai=True, **kwargs):
        super(UNetClassifier, self).__init__()
        # Support both naming conventions for backward compatibility
        self.n_channels = n_channels if n_channels is not None else input_channels
        self.n_classes = n_classes if n_classes is not None else num_classes
        self.use_monai = use_monai
        
        if use_monai:
            # Use MONAI U-Net as feature extractor
            self.backbone = MonaiUNet(
                spatial_dims=kwargs.get('spatial_dims', 2),
                in_channels=self.n_channels,
                out_channels=64,  # Feature channels instead of class channels
                channels=kwargs.get('channels', (16, 32, 64, 128, 256)),
                strides=kwargs.get('strides', (2, 2, 2, 2)),
                num_res_units=kwargs.get('num_res_units', 2),
            )
        else:
            # Fallback to custom implementation
            self.backbone = self._create_encoder(self.n_channels)
        
        # Global Average Pooling to convert from [B, 64, H, W] to [B, 64]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head: [B, 64] -> [B, n_classes]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, self.n_classes)
        )
        
    def _create_encoder(self, n_channels):
        """Create a simple encoder if MONAI is not available"""
        return nn.Sequential(
            # Initial conv block
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsampling blocks
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Final feature layer
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Extract features using backbone (either MONAI UNet or custom encoder)
        features = self.backbone(x)  # [B, 64, H, W]
        
        # Global pooling to get [B, 64, 1, 1]
        pooled = self.global_pool(features)  # [B, 64, 1, 1]
        
        # Classification head to get [B, n_classes]
        logits = self.classifier(pooled)  # [B, n_classes]
        
        return logits


class ResUNetClassifier(nn.Module):
    """ResUNet based classifier - uses ResUNet encoder with classification head"""
    
    def __init__(self, input_channels=3, num_classes=2, n_channels=None, n_classes=None, deep=False, use_attention=False, **kwargs):
        super(ResUNetClassifier, self).__init__()
        # Support both naming conventions for backward compatibility
        self.n_channels = n_channels if n_channels is not None else input_channels
        self.n_classes = n_classes if n_classes is not None else num_classes
        
        # Create ResUNet as feature extractor (use segmentation model but extract features)
        if deep:
            self.backbone = DeepResUNet(self.n_channels, out_channels=64, 
                                      bilinear=kwargs.get('bilinear', False),
                                      use_attention=use_attention)
        else:
            self.backbone = ResUNet(self.n_channels, out_channels=64,
                                  bilinear=kwargs.get('bilinear', False), 
                                  use_attention=use_attention)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, self.n_classes)
        )
        
    def forward(self, x):
        # Extract features using ResUNet backbone
        features = self.backbone(x)  # [B, 64, H, W]
        
        # Global pooling
        pooled = self.global_pool(features)  # [B, 64, 1, 1]
        
        # Classification
        logits = self.classifier(pooled)  # [B, n_classes]
        
        return logits


# Convenience functions for creating classification models
def create_unet_classifier(n_channels=3, n_classes=2, use_monai=True, **kwargs):
    """Create a U-Net based classifier"""
    return UNetClassifier(n_channels, n_classes, use_monai, **kwargs)


def create_resunet_classifier(n_channels=3, n_classes=2, deep=False, use_attention=False, **kwargs):
    """Create a ResUNet based classifier"""
    return ResUNetClassifier(n_channels, n_classes, deep, use_attention, **kwargs)


def create_deep_resunet_classifier(n_channels=3, n_classes=2, use_attention=False, **kwargs):
    """Create a Deep ResUNet based classifier"""
    return ResUNetClassifier(n_channels, n_classes, deep=True, use_attention=use_attention, **kwargs)


def create_attention_resunet_classifier(n_channels=3, n_classes=2, deep=False, **kwargs):
    """Create a ResUNet based classifier with attention"""
    return ResUNetClassifier(n_channels, n_classes, deep=deep, use_attention=True, **kwargs)
