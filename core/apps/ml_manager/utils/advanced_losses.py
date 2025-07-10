"""
Advanced Loss Functions for Binary Segmentation
===============================================

This module implements advanced loss functions from pywick and other sources,
specifically optimized for binary segmentation tasks like coronary artery segmentation.

Based on analysis of pywick/losses.py and best practices for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math


class TverskyLoss(nn.Module):
    """
    Tversky Loss - excellent for handling class imbalance in segmentation.
    
    Tversky coefficient is a generalization of Dice coefficient.
    When alpha = beta = 0.5, it becomes Dice coefficient.
    When alpha = beta = 1, it becomes Tanimoto coefficient.
    
    Args:
        alpha: Weight for false positives (controls penalty for false positives)
        beta: Weight for false negatives (controls penalty for false negatives)
        smooth: Smoothing factor to avoid division by zero
        
    For coronary segmentation:
        - Use alpha < 0.5, beta > 0.5 to penalize false negatives more (missing arteries)
        - Use alpha > 0.5, beta < 0.5 to penalize false positives more (over-segmentation)
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W] with values in [0, 1]
        """
        # Apply sigmoid to convert logits to probabilities
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        
        # Calculate True Positives, False Positives, False Negatives
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)
        
        # Tversky coefficient
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Return loss (1 - coefficient)
        return (1 - tversky).mean()
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Return individual loss components for monitoring."""
        with torch.no_grad():
            loss_value = self.forward(predictions, targets)
            
        return {
            'tversky_loss': loss_value.item(),
            'total_loss': loss_value.item(),
            'alpha': self.alpha,
            'beta': self.beta
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation - addresses class imbalance by down-weighting easy examples.
    
    Args:
        alpha: Weight for positive class (None for no weighting)
        gamma: Focusing parameter (higher gamma = more focus on hard examples)
        reduction: Loss reduction method
    """
    
    def __init__(self, alpha: Optional[float] = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W] with values in [0, 1]
        """
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Calculate pt (probability of correct class)
        pt = torch.exp(-bce_loss)
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Return individual loss components for monitoring."""
        with torch.no_grad():
            loss_value = self.forward(predictions, targets)
            
        return {
            'focal_loss': loss_value.item(),
            'total_loss': loss_value.item(),
            'alpha': self.alpha,
            'gamma': self.gamma
        }


class ComboDiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss with configurable weighting.
    This is one of the most effective losses for binary segmentation.
    
    Args:
        dice_weight: Weight for Dice loss component
        bce_weight: Weight for BCE loss component
        smooth: Smoothing factor for Dice loss
        use_focal: Whether to use Focal BCE instead of regular BCE
        focal_gamma: Gamma parameter for focal loss (only used if use_focal=True)
        focal_alpha: Alpha parameter for focal loss (only used if use_focal=True)
    """
    
    def __init__(self, 
                 dice_weight: float = 0.7, 
                 bce_weight: float = 0.3,
                 smooth: float = 1e-6,
                 use_focal: bool = False,
                 focal_gamma: float = 2.0,
                 focal_alpha: Optional[float] = 0.25):
        super(ComboDiceBCELoss, self).__init__()
        
        # Normalize weights
        total_weight = dice_weight + bce_weight
        self.dice_weight = dice_weight / total_weight
        self.bce_weight = bce_weight / total_weight
        
        self.smooth = smooth
        self.use_focal = use_focal
        
        if use_focal:
            self.bce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss component."""
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)
        
        return (1 - dice).mean()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W] with values in [0, 1]
        """
        dice_component = self.dice_loss(predictions, targets)
        bce_component = self.bce_loss(predictions, targets)
        
        return self.dice_weight * dice_component + self.bce_weight * bce_component
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Return individual loss components for monitoring."""
        with torch.no_grad():
            dice_component = self.dice_loss(predictions, targets)
            bce_component = self.bce_loss(predictions, targets)
            total_loss = self.dice_weight * dice_component + self.bce_weight * bce_component
            
        return {
            'dice_loss': dice_component.item(),
            'bce_loss': bce_component.item(),
            'total_loss': total_loss.item(),
            'dice_weight': self.dice_weight,
            'bce_weight': self.bce_weight
        }


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss - optimized version of standard Dice loss.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W] with values in [0, 1]
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        
        # Soft Dice coefficient
        dice = (2. * intersection + self.smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)
        
        return (1 - dice).mean()
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Return individual loss components for monitoring."""
        with torch.no_grad():
            loss_value = self.forward(predictions, targets)
            
        return {
            'soft_dice_loss': loss_value.item(),
            'total_loss': loss_value.item(),
            'smooth': self.smooth
        }


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for handling class imbalance.
    
    Args:
        pos_weight: Weight for positive class (if None, calculated from targets)
        adaptive: Whether to calculate pos_weight adaptively from each batch
    """
    
    def __init__(self, pos_weight: Optional[float] = None, adaptive: bool = False):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.adaptive = adaptive
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W] with values in [0, 1]
        """
        if self.adaptive or self.pos_weight is None:
            # Calculate adaptive pos_weight from current batch
            pos_count = targets.sum()
            neg_count = (1 - targets).sum()
            if pos_count > 0:
                pos_weight = neg_count / pos_count
            else:
                pos_weight = 1.0
        else:
            pos_weight = self.pos_weight
            
        return F.binary_cross_entropy_with_logits(
            predictions, targets, 
            pos_weight=torch.tensor(pos_weight, device=predictions.device)
        )
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Return individual loss components for monitoring."""
        with torch.no_grad():
            loss_value = self.forward(predictions, targets)
            pos_count = targets.sum()
            neg_count = (1 - targets).sum()
            current_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            
        return {
            'weighted_bce_loss': loss_value.item(),
            'total_loss': loss_value.item(),
            'pos_weight': self.pos_weight,
            'current_pos_weight': current_pos_weight.item()
        }


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for improved boundary segmentation.
    Based on "Boundary loss for highly unbalanced segmentation" paper.
    
    Args:
        theta0: Parameter for boundary thickness
        theta: Parameter for extended boundary
    """
    
    def __init__(self, theta0: int = 3, theta: int = 5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W] with values in [0, 1]
        """
        predictions = torch.sigmoid(predictions)
        
        # Create boundary maps
        gt_boundary = F.max_pool2d(
            1 - targets, 
            kernel_size=self.theta0, 
            stride=1, 
            padding=self.theta0 // 2
        )
        gt_boundary = gt_boundary - (1 - targets)
        
        pred_boundary = F.max_pool2d(
            1 - predictions, 
            kernel_size=self.theta0, 
            stride=1, 
            padding=self.theta0 // 2
        )
        pred_boundary = pred_boundary - (1 - predictions)
        
        # Extended boundary maps
        gt_boundary_ext = F.max_pool2d(
            gt_boundary, 
            kernel_size=self.theta, 
            stride=1, 
            padding=self.theta // 2
        )
        
        pred_boundary_ext = F.max_pool2d(
            pred_boundary, 
            kernel_size=self.theta, 
            stride=1, 
            padding=self.theta // 2
        )
        
        # Calculate Precision and Recall
        eps = 1e-7
        P = (pred_boundary * gt_boundary_ext).sum() / (pred_boundary.sum() + eps)
        R = (pred_boundary_ext * gt_boundary).sum() / (gt_boundary.sum() + eps)
        
        # Boundary F1 Score
        BF1 = (2 * P * R + eps) / (P + R + eps)
        
        return 1 - BF1
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Return individual loss components for monitoring."""
        with torch.no_grad():
            loss_value = self.forward(predictions, targets)
            
        return {
            'boundary_loss': loss_value.item(),
            'total_loss': loss_value.item(),
            'theta0': self.theta0,
            'theta': self.theta
        }


class StableBCELoss(nn.Module):
    """
    Numerically stable BCE loss - useful for extreme predictions.
    """
    
    def __init__(self):
        super(StableBCELoss, self).__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W] with values in [0, 1]
        """
        neg_abs = -predictions.abs()
        loss = predictions.clamp(min=0) - predictions * targets + (1 + neg_abs.exp()).log()
        return loss.mean()
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Return individual loss components for monitoring."""
        with torch.no_grad():
            loss_value = self.forward(predictions, targets)
            
        return {
            'stable_bce_loss': loss_value.item(),
            'total_loss': loss_value.item()
        }


# Factory function for easy loss selection
def create_advanced_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create advanced loss functions.
    
    Args:
        loss_type: Type of loss function to create
        **kwargs: Arguments to pass to the loss function
        
    Returns:
        Initialized loss function
        
    Available loss types:
        - 'tversky': TverskyLoss
        - 'focal': FocalLoss  
        - 'combo_dice_bce': ComboDiceBCELoss
        - 'soft_dice': SoftDiceLoss
        - 'weighted_bce': WeightedBCELoss
        - 'boundary': BoundaryLoss
        - 'stable_bce': StableBCELoss
    """
    loss_functions = {
        'tversky': TverskyLoss,
        'focal': FocalLoss,
        'combo_dice_bce': ComboDiceBCELoss,
        'soft_dice': SoftDiceLoss,
        'weighted_bce': WeightedBCELoss,
        'boundary': BoundaryLoss,
        'stable_bce': StableBCELoss,
        # Aliasy dla kompatybilności z systemem trenowania
        'tversky_precision': TverskyLoss,
        'tversky_recall': TverskyLoss,
    }
    # Obsługa aliasów z domyślnymi parametrami
    if loss_type == 'tversky_precision':
        kwargs.setdefault('alpha', 0.7)
        kwargs.setdefault('beta', 0.3)
    elif loss_type == 'tversky_recall':
        kwargs.setdefault('alpha', 0.3)
        kwargs.setdefault('beta', 0.7)
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_functions.keys())}")
        
    return loss_functions[loss_type](**kwargs)


# Recommended configurations for coronary segmentation
CORONARY_LOSS_CONFIGS = {
    'conservative': {
        # Good for initial training - balanced approach
        'loss_type': 'combo_dice_bce',
        'dice_weight': 0.7,
        'bce_weight': 0.3,
        'smooth': 1e-6
    },
    
    'recall_focused': {
        # Prioritizes not missing arteries (high recall)
        'loss_type': 'tversky',
        'alpha': 0.3,  # Low alpha = penalize false negatives more
        'beta': 0.7,   # High beta = less penalty for false positives
        'smooth': 1e-6
    },
    
    'precision_focused': {
        # Prioritizes clean segmentations (high precision)  
        'loss_type': 'tversky',
        'alpha': 0.7,  # High alpha = penalize false positives more
        'beta': 0.3,   # Low beta = less penalty for false negatives
        'smooth': 1e-6
    },
    
    'class_imbalanced': {
        # Good for datasets with severe class imbalance
        'loss_type': 'combo_dice_bce',
        'dice_weight': 0.6,
        'bce_weight': 0.4,
        'use_focal': True,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25
    },
    
    'boundary_aware': {
        # Focuses on boundary accuracy
        'loss_type': 'boundary',
        'theta0': 3,
        'theta': 5
    }
}


def get_recommended_loss(config_name: str) -> nn.Module:
    """Get a recommended loss configuration for coronary segmentation."""
    if config_name not in CORONARY_LOSS_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CORONARY_LOSS_CONFIGS.keys())}")
        
    config = CORONARY_LOSS_CONFIGS[config_name]
    loss_type = config.pop('loss_type')
    return create_advanced_loss(loss_type, **config)


if __name__ == "__main__":
    # Example usage
    print("Advanced Loss Functions for Binary Segmentation")
    print("=" * 50)
    
    # Create sample data
    batch_size, height, width = 4, 64, 64
    predictions = torch.randn(batch_size, 1, height, width)  # Logits
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test different loss functions
    losses_to_test = [
        ('Tversky (Recall-focused)', get_recommended_loss('recall_focused')),
        ('Tversky (Precision-focused)', get_recommended_loss('precision_focused')),
        ('Combo Dice+BCE', get_recommended_loss('conservative')),
        ('Focal BCE', create_advanced_loss('focal', gamma=2.0)),
        ('Boundary Loss', get_recommended_loss('boundary_aware')),
    ]
    
    print(f"Testing with data shapes: predictions={predictions.shape}, targets={targets.shape}")
    print()
    
    for name, loss_fn in losses_to_test:
        try:
            loss_value = loss_fn(predictions, targets)
            print(f"{name:25s}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"{name:25s}: ERROR - {e}")
    
    print("\n✅ All loss functions tested successfully!")
