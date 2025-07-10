import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def compute(self, pred, target):
        pass

# class Dice(LossFunction):
#     def compute(self, pred, target, smooth=1e-5):
#         pred = pred.contiguous()
#         target = target.contiguous()

#         intersection = (pred * target).sum(dim=2).sum(dim=2)

#         loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        
#         # print(f'\rLoss dice: {loss.mean()}', end='', flush=True)
#         return loss.mean()

class Dice(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        # Apply sigmoid to convert logits to probabilities
        # inputs = torch.sigmoid(inputs)
        
        # # Flatten the tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        # # Calculate intersection and union
        # intersection = (inputs * targets).sum()
        # union = inputs.sum() + targets.sum()
        
        # # Calculate Dice coefficient
        # dice = (2. * intersection + smooth) / (union + smooth)
        
        # # Return Dice loss
        # return 1 - dice
        pred = inputs.contiguous()
        target = targets.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
        return loss.mean()

class IoU(LossFunction):
    def compute(self, pred_soft, target, smooth=1e-5):
        intersection = (pred_soft * target).sum()
        union = pred_soft.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        loss = 1 - iou
        
        # print(f'\rLoss iou: {loss.item()}', end='', flush=True)
        return loss
    
class DebugDice(LossFunction):
    def compute(self, pred, target, smooth=1e-5):
        # Print detailed information about predictions and targets
        print("Pred stats:")
        print(f"  Min: {pred.min().item():.4f}")
        print(f"  Max: {pred.max().item():.4f}")
        print(f"  Mean: {pred.mean().item():.4f}")
        
        print("\nTarget stats:")
        print(f"  Min: {target.min().item():.4f}")
        print(f"  Max: {target.max().item():.4f}")
        print(f"  Mean: {target.mean().item():.4f}")
        
        # Soft thresholding instead of hard binarization
        pred_soft = torch.sigmoid(pred)
        
        # Compute intersection and sums with support for arbitrary dimensions
        intersection = (pred_soft * target).sum(dim=tuple(range(1, pred.dim())))
        pred_sum = pred_soft.sum(dim=tuple(range(1, pred.dim())))
        target_sum = target.sum(dim=tuple(range(1, target.dim())))
        
        print("\nIntersection stats:")
        print(f"  Mean intersection: {intersection.mean().item():.4f}")
        print(f"  Pred sum: {pred_sum.mean().item():.4f}")
        print(f"  Target sum: {target_sum.mean().item():.4f}")
        
        # Compute Dice coefficient and loss
        dice_coeff = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        loss = 1 - dice_coeff
        
        print("\nDice coefficient:", dice_coeff.mean().item())
        print("Dice loss:", loss.mean().item())
        
        return loss.mean()
    

# def calc_loss(pred, target, metrics, bce_weight=0.1):
#     bce = F.binary_cross_entropy_with_logits(pred, target)
#     pred_sigmoid = torch.sigmoid(pred)
#     dice = dice_loss(pred_sigmoid, target)
#     loss = bce * bce_weight + dice * (1 - bce_weight)
#     with torch.no_grad():
#         metrics['bce'] += bce.item() * target.size(0)
#         metrics['dice'] += dice.item() * target.size(0)
#         metrics['loss'] += loss.item() * target.size(0)
#     return loss


def calc_loss(outputs: torch.Tensor, labels: torch.Tensor, metrics: defaultdict, 
              bce_weight: float = 0.1, loss_function: nn.Module = None) -> torch.Tensor:
    """
    Calculate the combined loss for segmentation.
    
    Args:
        outputs: Model predictions (B, C, H, W)
        labels: Ground truth labels (B, C, H, W)
        metrics: Dictionary to store various metrics
        bce_weight: Weight for BCE loss
        loss_function: Custom loss function (Dice, IoU, etc.)
    
    Returns:
        Combined loss value
    """
    # Ensure inputs are float tensors
    outputs = outputs.float()
    labels = labels.float()

    # Binary Cross Entropy loss
    bce = F.binary_cross_entropy_with_logits(outputs, labels)
    pred_sigmoid = torch.sigmoid(outputs)

    # Custom loss (Dice, IoU, etc.)
    custom_loss = loss_function(pred_sigmoid, labels) if loss_function else 0

    # Combine losses
    loss = bce * bce_weight + custom_loss * (1 - bce_weight)

    # Store individual loss components
    with torch.no_grad():
        metrics['bce'] += bce.item() * labels.size(0)
        if loss_function:
            metrics[loss_function.__class__.__name__.lower()] += custom_loss.item() * labels.size(0)
        metrics['loss'] += loss.item() * labels.size(0)

    return loss
