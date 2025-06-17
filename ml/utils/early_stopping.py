"""
Early Stopping Implementation for Training
Automatically stops training when validation performance stops improving
"""

import logging
import time
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early Stopping utility to halt training when validation performance stops improving
    
    This class monitors validation metrics and stops training when:
    - Validation performance hasn't improved for `patience` epochs
    - Minimum number of epochs have been completed
    - Optional: Validation loss starts increasing significantly
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_epochs: int = 20,
                 min_delta: float = 1e-4,
                 monitor_metric: str = 'val_dice',
                 mode: str = 'max',
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        """
        Initialize Early Stopping
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_epochs: Minimum number of epochs to train before early stopping can occur
            min_delta: Minimum change in monitored metric to qualify as improvement
            monitor_metric: Metric to monitor ('val_dice', 'val_loss', etc.)
            mode: 'max' for metrics where higher is better, 'min' for lower is better
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to log early stopping decisions
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # State tracking
        self.best_metric = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.should_stop = False
        self.best_weights = None
        self.stopped_epoch = None
        
        # History tracking
        self.metric_history = []
        
        # Determine comparison function based on mode
        if mode == 'max':
            self.is_better = lambda current, best: current > best + self.min_delta
            self.best_metric = -float('inf')
        elif mode == 'min':
            self.is_better = lambda current, best: current < best - self.min_delta
            self.best_metric = float('inf')
        else:
            raise ValueError(f"Mode must be 'max' or 'min', got {mode}")
            
        if self.verbose:
            logger.info(f"[EARLY_STOPPING] Initialized with patience={patience}, "
                       f"min_epochs={min_epochs}, monitor='{monitor_metric}', mode='{mode}'")
    
    def __call__(self, epoch: int, metrics: Dict[str, float], model=None) -> Dict[str, Any]:
        """
        Check if training should stop based on current metrics
        
        Args:
            epoch: Current epoch number (0-based)
            metrics: Dictionary containing current metrics
            model: Optional model instance to save best weights
            
        Returns:
            Dictionary with early stopping information
        """
        current_metric = metrics.get(self.monitor_metric)
        
        if current_metric is None:
            logger.warning(f"[EARLY_STOPPING] Metric '{self.monitor_metric}' not found in metrics")
            return self._create_result(should_stop=False, reason="metric_not_found")
        
        # Track metric history
        self.metric_history.append({
            'epoch': epoch,
            'metric': current_metric,
            'timestamp': time.time()
        })
        
        # Don't apply early stopping before minimum epochs
        if epoch < self.min_epochs:
            if self.verbose and epoch == 0:
                logger.info(f"[EARLY_STOPPING] Will monitor after epoch {self.min_epochs}")
            return self._create_result(should_stop=False, reason="min_epochs_not_reached")
        
        # Check for improvement
        if self.is_better(current_metric, self.best_metric):
            # Improvement found
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            
            # Save best weights if model provided
            if model is not None and self.restore_best_weights:
                try:
                    import copy
                    self.best_weights = copy.deepcopy(model.state_dict())
                    if self.verbose:
                        logger.info(f"[EARLY_STOPPING] New best {self.monitor_metric}: {current_metric:.4f} "
                                   f"at epoch {epoch + 1} (saved weights)")
                except Exception as e:
                    logger.warning(f"[EARLY_STOPPING] Failed to save best weights: {e}")
            elif self.verbose:
                logger.info(f"[EARLY_STOPPING] New best {self.monitor_metric}: {current_metric:.4f} "
                           f"at epoch {epoch + 1}")
            
            return self._create_result(should_stop=False, reason="improvement_found")
        
        else:
            # No improvement
            self.epochs_without_improvement += 1
            
            if self.verbose:
                logger.info(f"[EARLY_STOPPING] No improvement for {self.epochs_without_improvement}/{self.patience} epochs "
                           f"(current: {current_metric:.4f}, best: {self.best_metric:.4f})")
            
            # Check if we should stop
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                
                logger.info(f"[EARLY_STOPPING] 🛑 Early stopping triggered! "
                           f"No improvement for {self.patience} epochs. "
                           f"Best {self.monitor_metric}: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")
                
                # Restore best weights if available
                if model is not None and self.best_weights is not None:
                    try:
                        model.load_state_dict(self.best_weights)
                        logger.info(f"[EARLY_STOPPING] ✅ Restored best weights from epoch {self.best_epoch + 1}")
                    except Exception as e:
                        logger.warning(f"[EARLY_STOPPING] Failed to restore best weights: {e}")
                
                return self._create_result(should_stop=True, reason="patience_exceeded")
            
            return self._create_result(should_stop=False, reason="waiting_for_improvement")
    
    def _create_result(self, should_stop: bool, reason: str) -> Dict[str, Any]:
        """Create standardized result dictionary"""
        return {
            'should_stop': should_stop,
            'reason': reason,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'stopped_epoch': self.stopped_epoch,
            'patience': self.patience
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of early stopping state"""
        return {
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'patience': self.patience,
            'min_epochs': self.min_epochs,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'should_stop': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'total_epochs_tracked': len(self.metric_history)
        }
    
    def reset(self):
        """Reset early stopping state for new training"""
        self.best_metric = -float('inf') if self.mode == 'max' else float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.should_stop = False
        self.best_weights = None
        self.stopped_epoch = None
        self.metric_history = []
        
        if self.verbose:
            logger.info("[EARLY_STOPPING] State reset for new training")


def create_early_stopping_from_args(args) -> Optional[EarlyStopping]:
    """
    Create EarlyStopping instance from training arguments
    
    Args:
        args: Training arguments object
        
    Returns:
        EarlyStopping instance or None if disabled
    """
    # Check if early stopping is enabled
    if not getattr(args, 'use_early_stopping', False):
        return None
    
    patience = getattr(args, 'early_stopping_patience', 10)
    min_epochs = getattr(args, 'early_stopping_min_epochs', 20)
    min_delta = getattr(args, 'early_stopping_min_delta', 1e-4)
    monitor_metric = getattr(args, 'early_stopping_metric', 'val_dice')
    
    # Determine mode based on metric
    if 'loss' in monitor_metric.lower():
        mode = 'min'  # Lower loss is better
    else:
        mode = 'max'  # Higher metrics (dice, accuracy) are better
    
    return EarlyStopping(
        patience=patience,
        min_epochs=min_epochs,
        min_delta=min_delta,
        monitor_metric=monitor_metric,
        mode=mode,
        restore_best_weights=True,
        verbose=True
    )


def add_early_stopping_args(parser):
    """Add early stopping arguments to argument parser"""
    early_stopping_group = parser.add_argument_group('Early Stopping')
    
    early_stopping_group.add_argument('--use-early-stopping', action='store_true',
                                     help='Enable early stopping during training')
    early_stopping_group.add_argument('--early-stopping-patience', type=int, default=10,
                                     help='Number of epochs to wait for improvement before stopping')
    early_stopping_group.add_argument('--early-stopping-min-epochs', type=int, default=20,
                                     help='Minimum number of epochs before early stopping can occur')
    early_stopping_group.add_argument('--early-stopping-min-delta', type=float, default=1e-4,
                                     help='Minimum improvement required to reset patience counter')
    early_stopping_group.add_argument('--early-stopping-metric', type=str, default='val_dice',
                                     choices=['val_dice', 'val_loss', 'val_accuracy'],
                                     help='Metric to monitor for early stopping')
