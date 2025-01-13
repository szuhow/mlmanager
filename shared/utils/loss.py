import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def compute(self, pred, target):
        pass

class Dice(LossFunction):
    def compute(self, pred, target, smooth=1e-5):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        
        # print(f'\rLoss dice: {loss.mean()}', end='', flush=True)
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
    
# class calcLoss:
#     def __init__(self, pred, target, metrics, bce_weight=0.5, loss_function=Dice()):
#         self.pred = pred
#         self.target = target
#         self.metrics = metrics
#         self.bce_weight = bce_weight
#         self.loss_function = loss_function

# Funkcja do obliczania straty łączonej BCE i Dice
def calc_loss(pred, target, metrics, bce_weight=0.5, loss_function=Dice()):
    # Apply sigmoid to get probabilities
    pred_soft = torch.sigmoid(pred)
    
    # Calculate BCE loss
    bce = F.binary_cross_entropy_with_logits(pred, target)
    
    # Calculate Dice loss
    dice_loss = loss_function.compute(pred_soft, target)
    
    # Combine losses with equal weights by default
    loss = bce * bce_weight + dice_loss * (1 - bce_weight)
    
    # Update metrics
    with torch.no_grad():
        batch_size = target.size(0)
        metrics['bce'] += bce.item() * batch_size
        metrics[loss_function.__class__.__name__.lower()] += dice_loss.item() * batch_size
        metrics['loss'] += loss.item() * batch_size

    return loss