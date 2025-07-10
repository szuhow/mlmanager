"""
Enhanced Checkpointing Manager for ML Training System
Provides advanced model checkpointing with automatic saving, versioning, and metadata.
"""

import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Advanced checkpoint manager with support for:
    - Automatic best model detection
    - Multiple checkpoint strategies
    - Comprehensive metadata storage
    - Model state restoration
    - Training resume capability
    """
    
    def __init__(self, checkpoint_dir: str, model_name: str, 
                 save_strategy: str = 'best', max_checkpoints: int = 5,
                 monitor_metric: str = 'val_loss', mode: str = 'min',
                 save_optimizer: bool = True, save_scheduler: bool = True):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name prefix for checkpoint files
            save_strategy: 'best', 'epoch', 'interval', or 'all'
            max_checkpoints: Maximum number of checkpoints to keep
            monitor_metric: Metric to monitor for best model selection
            mode: 'min' or 'max' for the monitored metric
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.save_strategy = save_strategy
        self.max_checkpoints = max_checkpoints
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        # Tracking variables
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.current_epoch = 0
        
        # Create subdirectories
        self.weights_dir = self.checkpoint_dir / 'weights'
        self.metadata_dir = self.checkpoint_dir / 'metadata'
        self.config_dir = self.checkpoint_dir / 'config'
        
        for dir_path in [self.weights_dir, self.metadata_dir, self.config_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"CheckpointManager initialized:")
        logger.info(f"  Directory: {self.checkpoint_dir}")
        logger.info(f"  Strategy: {self.save_strategy}")
        logger.info(f"  Monitor: {self.monitor_metric} ({self.mode})")
        logger.info(f"  Max checkpoints: {self.max_checkpoints}")
    
    def should_save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> tuple[bool, str]:
        """
        Determine if a checkpoint should be saved based on the strategy.
        
        Args:
            epoch: Current epoch number
            metrics: Training metrics dictionary
            
        Returns:
            Tuple of (should_save, checkpoint_type)
        """
        current_metric = metrics.get(self.monitor_metric)
        
        if self.save_strategy == 'best':
            if current_metric is None:
                return False, 'none'
                
            is_better = (
                (self.mode == 'min' and current_metric < self.best_metric) or
                (self.mode == 'max' and current_metric > self.best_metric)
            )
            
            if is_better:
                self.best_metric = current_metric
                return True, 'best'
            return False, 'none'
            
        elif self.save_strategy == 'epoch':
            return True, 'epoch'
            
        elif self.save_strategy == 'interval':
            # Default to every 5 epochs for interval
            interval = getattr(self, 'save_interval', 5)
            return epoch % interval == 0, 'interval'
            
        elif self.save_strategy == 'all':
            is_best = (
                current_metric is not None and 
                ((self.mode == 'min' and current_metric < self.best_metric) or
                 (self.mode == 'max' and current_metric > self.best_metric))
            )
            if is_best:
                self.best_metric = current_metric
                return True, 'best'
            return True, 'regular'
        
        return False, 'none'
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None, epoch: int = 0, 
                       metrics: Dict[str, float] = None, 
                       model_metadata: Dict[str, Any] = None,
                       training_args: Dict[str, Any] = None,
                       loss_function_info: Dict[str, Any] = None) -> Optional[str]:
        """
        Save a model checkpoint with comprehensive metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer instance (optional)
            scheduler: Learning rate scheduler (optional)
            epoch: Current epoch number
            metrics: Training metrics dictionary
            model_metadata: Additional model metadata
            training_args: Training configuration
            loss_function_info: Loss function configuration
            
        Returns:
            Path to saved checkpoint or None if not saved
        """
        if metrics is None:
            metrics = {}
            
        should_save, checkpoint_type = self.should_save_checkpoint(epoch, metrics)
        
        if not should_save:
            return None
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if checkpoint_type == 'best':
            filename = f"{self.model_name}_best_epoch_{epoch+1:03d}_{timestamp}.pth"
        else:
            filename = f"{self.model_name}_{checkpoint_type}_epoch_{epoch+1:03d}_{timestamp}.pth"
        
        checkpoint_path = self.weights_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'monitor_metric': self.monitor_metric,
            'checkpoint_type': checkpoint_type,
            'timestamp': timestamp,
            'pytorch_version': torch.__version__,
        }
        
        # Add optimizer state if requested and available
        if self.save_optimizer and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state if requested and available
        if self.save_scheduler and scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add additional metadata
        if model_metadata:
            checkpoint_data['model_metadata'] = model_metadata
        if training_args:
            checkpoint_data['training_args'] = training_args
        if loss_function_info:
            checkpoint_data['loss_function_info'] = loss_function_info
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update checkpoint history
        checkpoint_info = {
            'epoch': epoch + 1,
            'path': str(checkpoint_path),
            'filename': filename,
            'checkpoint_type': checkpoint_type,
            'metric_value': metrics.get(self.monitor_metric),
            'timestamp': timestamp,
            'metrics': metrics.copy()
        }
        
        self.checkpoint_history.append(checkpoint_info)
        
        # Save checkpoint metadata separately
        self._save_checkpoint_metadata(checkpoint_info, epoch, metrics, model_metadata)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        # Update current epoch
        self.current_epoch = epoch + 1
        
        logger.info(f"Checkpoint saved: {filename}")
        logger.info(f"Type: {checkpoint_type}, Epoch: {epoch+1}")
        if metrics.get(self.monitor_metric) is not None:
            logger.info(f"{self.monitor_metric}: {metrics[self.monitor_metric]:.6f}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       strict: bool = True) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model to load state into
            optimizer: Optimizer to restore state into (optional)
            scheduler: Scheduler to restore state into (optional)
            strict: Whether to strictly enforce state dict keys match
            
        Returns:
            Dictionary with checkpoint information and metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        device = next(model.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Restore optimizer state if available
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state restored")
            except Exception as e:
                logger.warning(f"Failed to restore optimizer state: {e}")
        
        # Restore scheduler state if available
        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Scheduler state restored")
            except Exception as e:
                logger.warning(f"Failed to restore scheduler state: {e}")
        
        # Update manager state
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.current_epoch = checkpoint.get('epoch', 0)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path.name}")
        logger.info(f"Epoch: {self.current_epoch}")
        logger.info(f"Best {self.monitor_metric}: {self.best_metric:.6f}")
        
        return {
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'metrics': checkpoint.get('metrics', {}),
            'model_metadata': checkpoint.get('model_metadata', {}),
            'training_args': checkpoint.get('training_args', {}),
            'loss_function_info': checkpoint.get('loss_function_info', {}),
            'checkpoint_type': checkpoint.get('checkpoint_type', 'unknown'),
            'timestamp': checkpoint.get('timestamp', '')
        }
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get the path to the best checkpoint."""
        best_checkpoints = [cp for cp in self.checkpoint_history 
                          if cp['checkpoint_type'] == 'best']
        
        if not best_checkpoints:
            return None
        
        # Return the most recent best checkpoint
        return best_checkpoints[-1]['path']
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        if not self.checkpoint_history:
            return None
        
        # Sort by epoch and return the latest
        sorted_checkpoints = sorted(self.checkpoint_history, 
                                  key=lambda x: x['epoch'], reverse=True)
        return sorted_checkpoints[0]['path']
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with their information."""
        return sorted(self.checkpoint_history, key=lambda x: x['epoch'])
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get a summary of all checkpoints and training progress."""
        if not self.checkpoint_history:
            return {
                'total_checkpoints': 0,
                'best_checkpoint': None,
                'latest_checkpoint': None,
                'training_progress': {}
            }
        
        best_checkpoint = self.get_best_checkpoint_path()
        latest_checkpoint = self.get_latest_checkpoint_path()
        
        # Calculate training progress
        epochs = [cp['epoch'] for cp in self.checkpoint_history]
        metrics_history = []
        
        for cp in self.checkpoint_history:
            if cp['metrics']:
                metrics_history.append({
                    'epoch': cp['epoch'],
                    'metrics': cp['metrics']
                })
        
        return {
            'total_checkpoints': len(self.checkpoint_history),
            'best_checkpoint': best_checkpoint,
            'latest_checkpoint': latest_checkpoint,
            'best_metric_value': self.best_metric,
            'monitor_metric': self.monitor_metric,
            'epoch_range': [min(epochs), max(epochs)] if epochs else [0, 0],
            'checkpoint_types': list(set(cp['checkpoint_type'] for cp in self.checkpoint_history)),
            'metrics_history': metrics_history,
            'training_progress': {
                'current_epoch': self.current_epoch,
                'total_epochs_trained': len(set(epochs))
            }
        }
    
    def _save_checkpoint_metadata(self, checkpoint_info: Dict[str, Any], 
                                 epoch: int, metrics: Dict[str, float],
                                 model_metadata: Dict[str, Any] = None):
        """Save detailed checkpoint metadata separately."""
        metadata_filename = f"checkpoint_metadata_epoch_{epoch+1:03d}.json"
        metadata_path = self.metadata_dir / metadata_filename
        
        metadata = {
            'checkpoint_info': checkpoint_info,
            'training_metrics': metrics,
            'model_metadata': model_metadata or {},
            'manager_state': {
                'best_metric': self.best_metric,
                'current_epoch': self.current_epoch,
                'strategy': self.save_strategy,
                'monitor_metric': self.monitor_metric
            },
            'system_info': {
                'pytorch_version': torch.__version__,
                'timestamp': datetime.now().isoformat(),
                'checkpoint_dir': str(self.checkpoint_dir)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the maximum limit."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Separate best and regular checkpoints
        best_checkpoints = [cp for cp in self.checkpoint_history 
                          if cp['checkpoint_type'] == 'best']
        regular_checkpoints = [cp for cp in self.checkpoint_history 
                             if cp['checkpoint_type'] != 'best']
        
        # Always keep all best checkpoints
        to_keep = best_checkpoints.copy()
        
        # Keep the most recent regular checkpoints
        regular_sorted = sorted(regular_checkpoints, 
                              key=lambda x: x['epoch'], reverse=True)
        
        keep_count = max(0, self.max_checkpoints - len(best_checkpoints))
        to_keep.extend(regular_sorted[:keep_count])
        
        # Remove old checkpoints
        to_remove = [cp for cp in self.checkpoint_history if cp not in to_keep]
        
        for checkpoint in to_remove:
            try:
                checkpoint_path = Path(checkpoint['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path.name}")
                
                # Remove corresponding metadata
                epoch = checkpoint['epoch']
                metadata_path = self.metadata_dir / f"checkpoint_metadata_epoch_{epoch:03d}.json"
                if metadata_path.exists():
                    metadata_path.unlink()
                    
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")
        
        # Update checkpoint history
        self.checkpoint_history = to_keep
    
    def export_training_history(self, output_path: str):
        """Export complete training history to JSON file."""
        history_data = {
            'checkpoint_manager_config': {
                'model_name': self.model_name,
                'save_strategy': self.save_strategy,
                'monitor_metric': self.monitor_metric,
                'mode': self.mode,
                'max_checkpoints': self.max_checkpoints
            },
            'training_summary': self.get_checkpoint_summary(),
            'complete_history': self.checkpoint_history,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Training history exported to: {output_path}")


class AutoCheckpointManager(CheckpointManager):
    """
    Automatic checkpoint manager with intelligent strategies and monitoring.
    """
    
    def __init__(self, *args, auto_strategy: str = 'adaptive', **kwargs):
        """
        Initialize with automatic checkpoint strategies.
        
        Args:
            auto_strategy: 'adaptive', 'performance_based', or 'time_based'
        """
        super().__init__(*args, **kwargs)
        self.auto_strategy = auto_strategy
        self.performance_history = []
        self.last_improvement_epoch = 0
        
    def should_save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> tuple[bool, str]:
        """Enhanced checkpoint decision with automatic strategies."""
        current_metric = metrics.get(self.monitor_metric)
        
        if self.auto_strategy == 'adaptive':
            return self._adaptive_checkpoint_strategy(epoch, metrics, current_metric)
        elif self.auto_strategy == 'performance_based':
            return self._performance_based_strategy(epoch, metrics, current_metric)
        elif self.auto_strategy == 'time_based':
            return self._time_based_strategy(epoch, metrics, current_metric)
        else:
            # Fall back to parent implementation
            return super().should_save_checkpoint(epoch, metrics)
    
    def _adaptive_checkpoint_strategy(self, epoch: int, metrics: Dict[str, float], 
                                    current_metric: Optional[float]) -> tuple[bool, str]:
        """Adaptive strategy that adjusts based on training progress."""
        if current_metric is None:
            return False, 'none'
        
        # Track performance history
        self.performance_history.append(current_metric)
        
        is_best = (
            (self.mode == 'min' and current_metric < self.best_metric) or
            (self.mode == 'max' and current_metric > self.best_metric)
        )
        
        if is_best:
            self.best_metric = current_metric
            self.last_improvement_epoch = epoch
            return True, 'best'
        
        # Save checkpoint if no improvement for a while (early plateau detection)
        epochs_since_improvement = epoch - self.last_improvement_epoch
        
        if epochs_since_improvement >= 10 and epoch % 5 == 0:
            return True, 'plateau'
        
        # Regular interval saving for long training
        if epoch > 50 and epoch % 20 == 0:
            return True, 'interval'
        
        return False, 'none'
    
    def _performance_based_strategy(self, epoch: int, metrics: Dict[str, float], 
                                  current_metric: Optional[float]) -> tuple[bool, str]:
        """Strategy based on performance improvement rate."""
        if current_metric is None:
            return False, 'none'
        
        # Always save best models
        is_best = (
            (self.mode == 'min' and current_metric < self.best_metric) or
            (self.mode == 'max' and current_metric > self.best_metric)
        )
        
        if is_best:
            self.best_metric = current_metric
            return True, 'best'
        
        # Save if performance is within 5% of best (good models)
        if self.mode == 'min':
            threshold = self.best_metric * 1.05
            if current_metric <= threshold:
                return True, 'good'
        else:
            threshold = self.best_metric * 0.95
            if current_metric >= threshold:
                return True, 'good'
        
        return False, 'none'
    
    def _time_based_strategy(self, epoch: int, metrics: Dict[str, float], 
                           current_metric: Optional[float]) -> tuple[bool, str]:
        """Time-based strategy with checkpointing at key intervals."""
        if current_metric is None:
            return False, 'none'
        
        # Always save best
        is_best = (
            (self.mode == 'min' and current_metric < self.best_metric) or
            (self.mode == 'max' and current_metric > self.best_metric)
        )
        
        if is_best:
            self.best_metric = current_metric
            return True, 'best'
        
        # Time-based intervals
        if epoch in [1, 5, 10, 25, 50, 100, 200] or epoch % 50 == 0:
            return True, 'milestone'
        
        return False, 'none'
