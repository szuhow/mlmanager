"""
Dynamic Learning Rate Scheduler for Training
Allows adjustment of learning rate during training based on various strategies
"""

import os
import json
import logging
import time
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class DynamicLearningRateScheduler:
    """Manages dynamic learning rate adjustments during training"""
    
    def __init__(self, initial_lr: float, model_id: int = None, scheduler_type: str = 'plateau', 
                 patience: int = 5, factor: float = 0.5, min_lr: float = 1e-7,
                 step_size: int = 10, gamma: float = 0.1):
        """
        Initialize the dynamic learning rate scheduler
        
        Args:
            initial_lr: Initial learning rate
            model_id: Model ID for file-based adjustments
            scheduler_type: Type of scheduler ('plateau', 'step', 'exponential', 'cosine', 'adaptive')
            patience: Epochs to wait before reducing LR (for plateau scheduler)
            factor: Factor to reduce LR by (for plateau scheduler)
            min_lr: Minimum learning rate
            step_size: Step size for step scheduler
            gamma: Gamma for step scheduler
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.model_id = model_id
        self.lr_history = []
        self.adjustment_file = None
        
        # Scheduler configuration
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.step_size = step_size
        self.gamma = gamma
        
        # State tracking
        self.best_metric = None
        self.epochs_without_improvement = 0
        self.last_adjustment_epoch = -1
        
        if model_id:
            # Create file path for learning rate adjustments
            self.adjustment_file = f"/tmp/lr_adjustment_{model_id}.json"
    
    def check_and_adjust(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if learning rate should be adjusted based on current metrics
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary containing current metrics (val_dice, val_loss, etc.)
            
        Returns:
            Dictionary with adjustment information
        """
        adjustment_info = {
            'adjusted': False,
            'old_lr': self.current_lr,
            'new_lr': self.current_lr,
            'reason': 'No adjustment needed'
        }
        
        # Check for manual adjustment first
        manual_lr = self.check_for_lr_adjustment()
        if manual_lr is not None:
            adjustment_info.update({
                'adjusted': True,
                'new_lr': manual_lr,
                'reason': 'Manual adjustment'
            })
            self.current_lr = manual_lr
            self._record_adjustment(epoch, manual_lr, 'manual')
            return adjustment_info
        
        # Apply automatic scheduling based on scheduler type
        new_lr = self._calculate_new_lr(epoch, metrics)
        
        if new_lr != self.current_lr and new_lr >= self.min_lr:
            adjustment_info.update({
                'adjusted': True,
                'new_lr': new_lr,
                'reason': f'{self.scheduler_type} scheduler adjustment'
            })
            self.current_lr = new_lr
            self._record_adjustment(epoch, new_lr, self.scheduler_type)
        
        return adjustment_info
    
    def _calculate_new_lr(self, epoch: int, metrics: Dict[str, float]) -> float:
        """Calculate new learning rate based on scheduler type and metrics"""
        
        if self.scheduler_type == 'plateau':
            return self._plateau_scheduler(metrics)
        elif self.scheduler_type == 'step':
            return self._step_scheduler(epoch)
        elif self.scheduler_type == 'exponential':
            return self._exponential_scheduler(epoch)
        elif self.scheduler_type == 'cosine':
            return self._cosine_scheduler(epoch)
        elif self.scheduler_type == 'adaptive':
            return self._adaptive_scheduler(metrics)
        else:
            return self.current_lr
    
    def _plateau_scheduler(self, metrics: Dict[str, float]) -> float:
        """ReduceLROnPlateau scheduler implementation"""
        # Use validation dice as the primary metric for improvement
        current_metric = metrics.get('val_dice', 0.0)
        
        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            self.epochs_without_improvement = 0  # Reset counter
            return new_lr
        
        return self.current_lr
    
    def _step_scheduler(self, epoch: int) -> float:
        """StepLR scheduler implementation"""
        if epoch > 0 and epoch % self.step_size == 0:
            return max(self.current_lr * self.gamma, self.min_lr)
        return self.current_lr
    
    def _exponential_scheduler(self, epoch: int) -> float:
        """ExponentialLR scheduler implementation"""
        return max(self.initial_lr * (self.gamma ** epoch), self.min_lr)
    
    def _cosine_scheduler(self, epoch: int, max_epochs: int = 100) -> float:
        """CosineAnnealingLR scheduler implementation"""
        import math
        if epoch == 0:
            return self.initial_lr
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
        return max(self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor, self.min_lr)
    
    def _adaptive_scheduler(self, metrics: Dict[str, float]) -> float:
        """Adaptive scheduler based on multiple metrics"""
        val_dice = metrics.get('val_dice', 0.0)
        val_loss = metrics.get('val_loss', float('inf'))
        train_loss = metrics.get('train_loss', float('inf'))
        
        # Adaptive logic based on training progress
        if self.best_metric is None:
            self.best_metric = val_dice
            return self.current_lr
        
        # Check for rapid improvement - increase LR slightly
        if val_dice > self.best_metric * 1.05:  # 5% improvement
            self.best_metric = val_dice
            self.epochs_without_improvement = 0
            return min(self.current_lr * 1.1, self.initial_lr * 2)  # Cap at 2x initial
        
        # Check for stagnation - reduce LR
        elif val_dice <= self.best_metric:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= 3:
                self.epochs_without_improvement = 0
                return max(self.current_lr * 0.5, self.min_lr)
        
        return self.current_lr
    
    def _record_adjustment(self, epoch: int, new_lr: float, reason: str):
        """Record learning rate adjustment in history"""
        self.lr_history.append({
            'epoch': epoch,
            'timestamp': time.time(),
            'old_lr': self.current_lr,
            'new_lr': new_lr,
            'reason': reason
        })
        
        logger.info(f"[LR_SCHEDULER] Learning rate adjusted at epoch {epoch}: {self.current_lr:.6f} → {new_lr:.6f} ({reason})")

    def check_for_lr_adjustment(self) -> Optional[float]:
        """Check if there's a pending learning rate adjustment"""
        if not self.adjustment_file or not os.path.exists(self.adjustment_file):
            return None
        
        try:
            with open(self.adjustment_file, 'r') as f:
                data = json.load(f)
            
            new_lr = data.get('learning_rate')
            if new_lr and new_lr != self.current_lr:
                # Remove the file after reading
                os.remove(self.adjustment_file)
                return float(new_lr)
                
        except Exception as e:
            logger.warning(f"[LR_SCHEDULER] Failed to read adjustment file: {e}")
        
        return None
    
    def adjust_learning_rate(self, optimizer, new_lr: float):
        """Adjust learning rate in optimizer"""
        old_lr = self.current_lr
        self.current_lr = new_lr
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.lr_history.append({
            'timestamp': time.time(),
            'old_lr': old_lr,
            'new_lr': new_lr
        })
        
        logger.info(f"[LR_SCHEDULER] Learning rate adjusted: {old_lr:.6f} → {new_lr:.6f}")
        return True
    
    def adaptive_lr_adjustment(self, current_loss: float, best_loss: float, patience_count: int):
        """Adaptive learning rate adjustment based on training progress"""
        adjustment_factor = 1.0
        
        # Reduce LR if loss hasn't improved for several epochs
        if patience_count >= 3:
            adjustment_factor = 0.5
            logger.info(f"[LR_SCHEDULER] Reducing LR due to plateau (patience: {patience_count})")
        
        # Increase LR slightly if loss is improving rapidly
        elif current_loss < best_loss * 0.9:
            adjustment_factor = 1.1
            logger.info(f"[LR_SCHEDULER] Increasing LR due to rapid improvement")
        
        if adjustment_factor != 1.0:
            new_lr = max(self.current_lr * adjustment_factor, 1e-7)  # Don't go below 1e-7
            return new_lr
        
        return None
    
    def get_lr_history(self):
        """Get learning rate adjustment history"""
        return self.lr_history.copy()
    
    def save_lr_state(self, filepath: str):
        """Save learning rate state to file"""
        state = {
            'initial_lr': self.initial_lr,
            'current_lr': self.current_lr,
            'history': self.lr_history
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"[LR_SCHEDULER] Saved LR state to {filepath}")
        except Exception as e:
            logger.error(f"[LR_SCHEDULER] Failed to save LR state: {e}")

def create_lr_adjustment_file(model_id: int, new_lr: float):
    """Create a learning rate adjustment file for a running training"""
    adjustment_file = f"/tmp/lr_adjustment_{model_id}.json"
    
    data = {
        'learning_rate': new_lr,
        'timestamp': time.time(),
        'requested_by': 'user'
    }
    
    try:
        with open(adjustment_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"[LR_ADJUSTMENT] Created adjustment file for model {model_id}: LR = {new_lr}")
        return True
        
    except Exception as e:
        logger.error(f"[LR_ADJUSTMENT] Failed to create adjustment file: {e}")
        return False

def get_current_lr_info(model_id: int) -> Dict[str, Any]:
    """Get current learning rate information for a model"""
    lr_state_file = f"/tmp/lr_state_{model_id}.json"
    
    if os.path.exists(lr_state_file):
        try:
            with open(lr_state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[LR_INFO] Failed to read LR state: {e}")
    
    return {'current_lr': 'unknown', 'history': []}
