"""
Enhanced Loss Function Manager for ML Training System
Provides advanced loss function combinations, scheduling, and monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union
from abc import ABC, abstractmethod

try:
    from monai.losses import DiceLoss as MonaiDiceLoss, FocalLoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseLossFunction(nn.Module, ABC):
    """Base class for all loss functions with common interface."""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.history = []
    
    @abstractmethod
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss value."""
        pass
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Get detailed loss component breakdown."""
        with torch.no_grad():
            loss_value = self.forward(predictions, targets)
            return {
                f'{self.name}_loss': loss_value.item(),
                'total_loss': loss_value.item()
            }
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss function configuration."""
        return {
            'name': self.name,
            'class': self.__class__.__name__
        }


class EnhancedDiceLoss(BaseLossFunction):
    """Enhanced Dice loss with multiple variants and smoothing options."""
    
    def __init__(self, smooth: float = 1e-6, squared_pred: bool = False, 
                 jaccard: bool = False, reduction: str = 'mean'):
        super().__init__('dice')
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to predictions
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        
        # Calculate intersection and sums
        intersection = (pred_flat * target_flat).sum(dim=1)
        
        if self.squared_pred:
            pred_sum = (pred_flat ** 2).sum(dim=1)
        else:
            pred_sum = pred_flat.sum(dim=1)
            
        target_sum = target_flat.sum(dim=1)
        
        # Dice coefficient
        if self.jaccard:
            # Jaccard index (IoU)
            union = pred_sum + target_sum - intersection
            dice = (intersection + self.smooth) / (union + self.smooth)
        else:
            # Dice coefficient
            dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Dice loss
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'smooth': self.smooth,
            'squared_pred': self.squared_pred,
            'jaccard': self.jaccard,
            'reduction': self.reduction
        })
        return config


class EnhancedBCELoss(BaseLossFunction):
    """Enhanced Binary Cross Entropy with class weighting and focal loss features."""
    
    def __init__(self, pos_weight: Optional[float] = None, alpha: Optional[float] = None,
                 gamma: float = 0.0, reduction: str = 'mean'):
        super().__init__('bce')
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma  # Focal loss gamma parameter
        self.reduction = reduction
        
        # Create BCE loss
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight), 
                reduction='none' if gamma > 0 else reduction
            )
        else:
            self.bce_loss = nn.BCEWithLogitsLoss(
                reduction='none' if gamma > 0 else reduction
            )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate basic BCE loss
        bce_loss = self.bce_loss(predictions, targets)
        
        # Apply focal loss modification if gamma > 0
        if self.gamma > 0:
            # Calculate pt (probability of correct class)
            probs = torch.sigmoid(predictions)
            pt = targets * probs + (1 - targets) * (1 - probs)
            
            # Apply focal weight
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply alpha weighting if specified
            if self.alpha is not None:
                alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
                focal_weight = alpha_t * focal_weight
            
            loss = focal_weight * bce_loss
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        
        return bce_loss
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'pos_weight': self.pos_weight,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'reduction': self.reduction
        })
        return config


class CombinedLoss(BaseLossFunction):
    """Flexible combined loss function supporting multiple loss components."""
    
    def __init__(self, loss_functions: Dict[str, BaseLossFunction], 
                 weights: Dict[str, float], normalize_weights: bool = True):
        super().__init__('combined')
        self.loss_functions = nn.ModuleDict(loss_functions)
        
        # Normalize weights if requested
        if normalize_weights:
            total_weight = sum(weights.values())
            self.weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.weights = weights
        
        # Validate that all loss functions have corresponding weights
        for name in loss_functions.keys():
            if name not in weights:
                raise ValueError(f"No weight specified for loss function: {name}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for name, loss_fn in self.loss_functions.items():
            loss_value = loss_fn(predictions, targets)
            weighted_loss = self.weights[name] * loss_value
            total_loss += weighted_loss
        
        return total_loss
    
    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Get detailed breakdown of all loss components."""
        components = {}
        total_loss = 0.0
        
        with torch.no_grad():
            for name, loss_fn in self.loss_functions.items():
                loss_value = loss_fn(predictions, targets)
                weighted_loss = self.weights[name] * loss_value
                
                components[f'{name}_loss'] = loss_value.item()
                components[f'{name}_weighted'] = weighted_loss.item()
                components[f'{name}_weight'] = self.weights[name]
                
                total_loss += weighted_loss.item()
            
            components['total_loss'] = total_loss
        
        return components
    
    def update_weights(self, new_weights: Dict[str, float], normalize: bool = True):
        """Update loss function weights dynamically."""
        if normalize:
            total_weight = sum(new_weights.values())
            self.weights = {k: v / total_weight for k, v in new_weights.items()}
        else:
            self.weights = new_weights
        
        logger.info(f"Loss weights updated: {self.weights}")
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'weights': self.weights,
            'loss_functions': {name: loss_fn.get_config() 
                             for name, loss_fn in self.loss_functions.items()}
        })
        return config


class LossScheduler:
    """Scheduler for dynamically adjusting loss function weights during training."""
    
    def __init__(self, loss_function: CombinedLoss, schedule_type: str = 'adaptive',
                 **scheduler_params):
        self.loss_function = loss_function
        self.schedule_type = schedule_type
        self.scheduler_params = scheduler_params
        self.history = []
        
    def step(self, epoch: int, metrics: Dict[str, float]):
        """Update loss weights based on training progress."""
        if self.schedule_type == 'adaptive':
            self._adaptive_schedule(epoch, metrics)
        elif self.schedule_type == 'cosine':
            self._cosine_schedule(epoch, metrics)
        elif self.schedule_type == 'step':
            self._step_schedule(epoch, metrics)
        elif self.schedule_type == 'performance':
            self._performance_based_schedule(epoch, metrics)
        
        # Record weight changes
        self.history.append({
            'epoch': epoch,
            'weights': self.loss_function.weights.copy(),
            'metrics': metrics.copy()
        })
    
    def _adaptive_schedule(self, epoch: int, metrics: Dict[str, float]):
        """Adaptive scheduling based on validation performance."""
        val_dice = metrics.get('val_dice', 0.5)
        
        # Adjust weights based on performance
        if val_dice > 0.8:  # High performance - focus more on Dice
            dice_weight = min(0.9, self.loss_function.weights.get('dice', 0.7) * 1.1)
        elif val_dice < 0.4:  # Low performance - balance more towards BCE
            dice_weight = max(0.5, self.loss_function.weights.get('dice', 0.7) * 0.9)
        else:
            return  # No change needed
        
        bce_weight = 1.0 - dice_weight
        self.loss_function.update_weights({'dice': dice_weight, 'bce': bce_weight})
    
    def _cosine_schedule(self, epoch: int, metrics: Dict[str, float]):
        """Cosine annealing for loss weights."""
        max_epochs = self.scheduler_params.get('max_epochs', 100)
        min_dice_weight = self.scheduler_params.get('min_dice_weight', 0.5)
        max_dice_weight = self.scheduler_params.get('max_dice_weight', 0.9)
        
        # Cosine annealing
        progress = min(epoch / max_epochs, 1.0)
        dice_weight = min_dice_weight + (max_dice_weight - min_dice_weight) * \
                     (1 + np.cos(np.pi * progress)) / 2
        
        bce_weight = 1.0 - dice_weight
        self.loss_function.update_weights({'dice': dice_weight, 'bce': bce_weight})
    
    def _step_schedule(self, epoch: int, metrics: Dict[str, float]):
        """Step-based scheduling."""
        step_size = self.scheduler_params.get('step_size', 20)
        weight_step = self.scheduler_params.get('weight_step', 0.1)
        
        if epoch > 0 and epoch % step_size == 0:
            current_dice = self.loss_function.weights.get('dice', 0.7)
            new_dice_weight = min(0.9, current_dice + weight_step)
            bce_weight = 1.0 - new_dice_weight
            
            self.loss_function.update_weights({'dice': new_dice_weight, 'bce': bce_weight})
    
    def _performance_based_schedule(self, epoch: int, metrics: Dict[str, float]):
        """Performance-based weight adjustment."""
        if len(self.history) < 2:
            return
        
        # Compare current performance with previous
        current_val_dice = metrics.get('val_dice', 0.0)
        prev_val_dice = self.history[-1]['metrics'].get('val_dice', 0.0)
        
        improvement = current_val_dice - prev_val_dice
        
        if improvement > 0.01:  # Good improvement - maintain current strategy
            return
        elif improvement < -0.01:  # Performance degraded - adjust weights
            # Reduce dice weight slightly
            current_dice = self.loss_function.weights.get('dice', 0.7)
            new_dice_weight = max(0.5, current_dice - 0.05)
            bce_weight = 1.0 - new_dice_weight
            
            self.loss_function.update_weights({'dice': new_dice_weight, 'bce': bce_weight})


class LossManager:
    """Central manager for loss functions with factory methods and monitoring."""
    
    @staticmethod
    def create_loss_function(loss_config: Dict[str, Any]) -> BaseLossFunction:
        """
        Factory method to create loss functions from configuration.
        
        Args:
            loss_config: Configuration dictionary with loss type and parameters
            
        Returns:
            Configured loss function
        """
        loss_type = loss_config.get('type', 'dice')
        
        if loss_type == 'dice':
            return EnhancedDiceLoss(
                smooth=loss_config.get('smooth', 1e-6),
                squared_pred=loss_config.get('squared_pred', False),
                jaccard=loss_config.get('jaccard', False),
                reduction=loss_config.get('reduction', 'mean')
            )
        
        elif loss_type == 'bce':
            return EnhancedBCELoss(
                pos_weight=loss_config.get('pos_weight'),
                alpha=loss_config.get('alpha'),
                gamma=loss_config.get('gamma', 0.0),
                reduction=loss_config.get('reduction', 'mean')
            )
        
        elif loss_type == 'combined' or loss_type == 'mixed':
            # Create component loss functions
            dice_config = loss_config.get('dice_config', {})
            bce_config = loss_config.get('bce_config', {})
            
            dice_loss = EnhancedDiceLoss(**dice_config)
            bce_loss = EnhancedBCELoss(**bce_config)
            
            # Get weights
            dice_weight = loss_config.get('dice_weight', 0.7)
            bce_weight = loss_config.get('bce_weight', 0.3)
            
            return CombinedLoss(
                loss_functions={'dice': dice_loss, 'bce': bce_loss},
                weights={'dice': dice_weight, 'bce': bce_weight},
                normalize_weights=loss_config.get('normalize_weights', True)
            )
        
        elif loss_type == 'focal':
            return EnhancedBCELoss(
                alpha=loss_config.get('alpha', 0.25),
                gamma=loss_config.get('gamma', 2.0),
                reduction=loss_config.get('reduction', 'mean')
            )
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def create_loss_scheduler(loss_function: CombinedLoss, 
                            scheduler_config: Dict[str, Any]) -> LossScheduler:
        """Create a loss scheduler for dynamic weight adjustment."""
        return LossScheduler(
            loss_function=loss_function,
            schedule_type=scheduler_config.get('type', 'adaptive'),
            **scheduler_config.get('params', {})
        )
    
    @staticmethod
    def evaluate_loss_performance(loss_function: BaseLossFunction,
                                predictions: torch.Tensor,
                                targets: torch.Tensor) -> Dict[str, Any]:
        """Comprehensive evaluation of loss function performance."""
        with torch.no_grad():
            # Basic loss computation
            loss_value = loss_function(predictions, targets)
            components = loss_function.get_loss_components(predictions, targets)
            
            # Additional metrics
            predictions_sigmoid = torch.sigmoid(predictions)
            
            # Dice coefficient for comparison
            pred_flat = predictions_sigmoid.view(-1)
            target_flat = targets.view(-1)
            intersection = (pred_flat * target_flat).sum()
            dice_coeff = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
            
            # IoU calculation
            union = pred_flat.sum() + target_flat.sum() - intersection
            iou = intersection / (union + 1e-6)
            
            # Accuracy metrics
            pred_binary = (predictions_sigmoid > 0.5).float()
            accuracy = (pred_binary == targets).float().mean()
            
            return {
                'loss_value': loss_value.item(),
                'loss_components': components,
                'dice_coefficient': dice_coeff.item(),
                'iou': iou.item(),
                'accuracy': accuracy.item(),
                'prediction_stats': {
                    'min': predictions.min().item(),
                    'max': predictions.max().item(),
                    'mean': predictions.mean().item(),
                    'std': predictions.std().item()
                }
            }


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

LOSS_PRESETS = {
    'default_segmentation': {
        'type': 'combined',
        'dice_weight': 0.7,
        'bce_weight': 0.3,
        'dice_config': {'smooth': 1e-6},
        'bce_config': {}
    },
    
    'focal_segmentation': {
        'type': 'combined',
        'dice_weight': 0.6,
        'bce_weight': 0.4,
        'dice_config': {'smooth': 1e-6},
        'bce_config': {'gamma': 2.0, 'alpha': 0.25}
    },
    
    'balanced_segmentation': {
        'type': 'combined',
        'dice_weight': 0.5,
        'bce_weight': 0.5,
        'dice_config': {'smooth': 1e-6, 'jaccard': False},
        'bce_config': {'pos_weight': 1.0}
    },
    
    'dice_focused': {
        'type': 'combined',
        'dice_weight': 0.8,
        'bce_weight': 0.2,
        'dice_config': {'smooth': 1e-6, 'squared_pred': True},
        'bce_config': {}
    },
    
    'jaccard_based': {
        'type': 'combined',
        'dice_weight': 0.7,
        'bce_weight': 0.3,
        'dice_config': {'smooth': 1e-6, 'jaccard': True},
        'bce_config': {}
    }
}

SCHEDULER_PRESETS = {
    'adaptive': {
        'type': 'adaptive'
    },
    
    'cosine_annealing': {
        'type': 'cosine',
        'params': {
            'max_epochs': 100,
            'min_dice_weight': 0.5,
            'max_dice_weight': 0.9
        }
    },
    
    'step_increase': {
        'type': 'step',
        'params': {
            'step_size': 20,
            'weight_step': 0.05
        }
    },
    
    'performance_based': {
        'type': 'performance'
    }
}


def get_preset_loss_config(preset_name: str) -> Dict[str, Any]:
    """Get a preset loss configuration."""
    if preset_name not in LOSS_PRESETS:
        available = list(LOSS_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    return LOSS_PRESETS[preset_name].copy()


def get_preset_scheduler_config(preset_name: str) -> Dict[str, Any]:
    """Get a preset scheduler configuration."""
    if preset_name not in SCHEDULER_PRESETS:
        available = list(SCHEDULER_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    return SCHEDULER_PRESETS[preset_name].copy()
