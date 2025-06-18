"""
Comprehensive Training Manager for ML Training System
Integrates checkpointing, loss management, and enhanced training features.
"""

import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable

# Import our enhanced utilities
from .checkpoint_manager import CheckpointManager, AutoCheckpointManager
from .loss_manager import LossManager, LossScheduler, get_preset_loss_config, get_preset_scheduler_config

try:
    from core.apps.dataset_manager.utils import EnhancedModelCheckpoint, MixedLoss, create_loss_function
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Comprehensive training manager that orchestrates all training components:
    - Checkpoint management
    - Loss function management and scheduling
    - Training progress monitoring
    - Enhanced logging and metrics
    """
    
    def __init__(self, model_dir: str, model_name: str, config: Dict[str, Any] = None):
        """
        Initialize the training manager.
        
        Args:
            model_dir: Directory for model outputs
            model_name: Name of the model
            config: Training configuration dictionary
        """
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.config = config or {}
        
        # Create directory structure
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.model_dir / 'weights'
        self.logs_dir = self.model_dir / 'logs'
        self.artifacts_dir = self.model_dir / 'artifacts'
        
        for dir_path in [self.weights_dir, self.logs_dir, self.artifacts_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.checkpoint_manager = None
        self.loss_function = None
        self.loss_scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = None
        self.training_history = []
        
        logger.info(f"TrainingManager initialized for {model_name}")
        logger.info(f"Model directory: {model_dir}")
    
    def setup_checkpointing(self, checkpoint_config: Dict[str, Any] = None) -> CheckpointManager:
        """
        Setup checkpoint management.
        
        Args:
            checkpoint_config: Checkpoint configuration
            
        Returns:
            Configured checkpoint manager
        """
        if checkpoint_config is None:
            checkpoint_config = {
                'save_strategy': 'best',
                'max_checkpoints': 5,
                'monitor_metric': 'val_dice',
                'mode': 'max'
            }
        
        # Choose checkpoint manager type
        manager_type = checkpoint_config.get('manager_type', 'standard')
        
        if manager_type == 'auto':
            self.checkpoint_manager = AutoCheckpointManager(
                checkpoint_dir=str(self.weights_dir / 'checkpoints'),
                model_name=self.model_name,
                save_strategy=checkpoint_config.get('save_strategy', 'best'),
                max_checkpoints=checkpoint_config.get('max_checkpoints', 5),
                monitor_metric=checkpoint_config.get('monitor_metric', 'val_dice'),
                mode=checkpoint_config.get('mode', 'max'),
                auto_strategy=checkpoint_config.get('auto_strategy', 'adaptive')
            )
        else:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(self.weights_dir / 'checkpoints'),
                model_name=self.model_name,
                save_strategy=checkpoint_config.get('save_strategy', 'best'),
                max_checkpoints=checkpoint_config.get('max_checkpoints', 5),
                monitor_metric=checkpoint_config.get('monitor_metric', 'val_dice'),
                mode=checkpoint_config.get('mode', 'max')
            )
        
        logger.info(f"Checkpoint manager setup complete ({manager_type})")
        return self.checkpoint_manager
    
    def setup_loss_function(self, loss_config: Dict[str, Any] = None):
        """
        Setup loss function and optional scheduling.
        
        Args:
            loss_config: Loss function configuration
            
        Returns:
            Configured loss function
        """
        if loss_config is None:
            loss_config = get_preset_loss_config('default_segmentation')
        
        # Check if it's a preset
        preset_name = loss_config.get('preset')
        if preset_name:
            loss_config = get_preset_loss_config(preset_name)
        
        # Create loss function
        self.loss_function = LossManager.create_loss_function(loss_config)
        
        # Setup loss scheduler if requested
        scheduler_config = loss_config.get('scheduler')
        if scheduler_config and hasattr(self.loss_function, 'update_weights'):
            self.loss_scheduler = LossManager.create_loss_scheduler(
                self.loss_function, scheduler_config
            )
            logger.info("Loss scheduler enabled")
        
        logger.info(f"Loss function setup complete: {self.loss_function.name}")
        return self.loss_function
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any] = None, epoch: int = 0,
                       metrics: Dict[str, float] = None,
                       additional_data: Dict[str, Any] = None) -> Optional[str]:
        """
        Save a training checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            epoch: Current epoch
            metrics: Training metrics
            additional_data: Additional data to save
            
        Returns:
            Path to saved checkpoint or None
        """
        if self.checkpoint_manager is None:
            logger.warning("Checkpoint manager not initialized")
            return None
        
        # Prepare model metadata
        model_metadata = {
            'model_name': self.model_name,
            'model_class': model.__class__.__name__,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': str(next(model.parameters()).device),
            'training_manager_version': '1.0.0'
        }
        
        if additional_data:
            model_metadata.update(additional_data)
        
        # Loss function info
        loss_function_info = {}
        if self.loss_function:
            loss_function_info = self.loss_function.get_config()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics or {},
            model_metadata=model_metadata,
            training_args=self.config,
            loss_function_info=loss_function_info
        )
        
        # Update training state
        self.current_epoch = epoch + 1
        if metrics:
            self._update_training_history(epoch, metrics)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """
        Load a checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model
            optimizer: Optimizer (optional)
            scheduler: Scheduler (optional)
            
        Returns:
            Checkpoint information
        """
        if self.checkpoint_manager is None:
            # Create temporary checkpoint manager
            self.setup_checkpointing()
        
        # Load checkpoint
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        
        # Update training state
        self.current_epoch = checkpoint_info['epoch']
        self.best_metric = checkpoint_info['best_metric']
        
        logger.info(f"Training resumed from epoch {self.current_epoch}")
        return checkpoint_info
    
    def train_step(self, model: torch.nn.Module, batch_data: tuple,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   step: int = None) -> Dict[str, float]:
        """
        Execute a single training step with enhanced logging.
        
        Args:
            model: PyTorch model
            batch_data: (inputs, targets) tuple
            optimizer: Optimizer
            device: Training device
            step: Current step number
            
        Returns:
            Dictionary with loss components and metrics
        """
        if self.loss_function is None:
            raise RuntimeError("Loss function not initialized. Call setup_loss_function() first.")
        
        inputs, targets = batch_data
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate loss
        loss = self.loss_function(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Get loss components
        loss_components = self.loss_function.get_loss_components(outputs, targets)
        
        # Calculate additional metrics
        with torch.no_grad():
            predictions = torch.sigmoid(outputs)
            
            # Dice coefficient
            pred_flat = predictions.view(-1)
            target_flat = targets.view(-1)
            intersection = (pred_flat * target_flat).sum()
            dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
            
            # Accuracy
            pred_binary = (predictions > 0.5).float()
            accuracy = (pred_binary == targets).float().mean()
        
        step_metrics = {
            **loss_components,
            'dice': dice.item(),
            'accuracy': accuracy.item()
        }
        
        return step_metrics
    
    def validation_step(self, model: torch.nn.Module, val_loader,
                       device: torch.device) -> Dict[str, float]:
        """
        Execute validation with comprehensive metrics.
        
        Args:
            model: PyTorch model
            val_loader: Validation data loader
            device: Device for computation
            
        Returns:
            Validation metrics
        """
        if self.loss_function is None:
            raise RuntimeError("Loss function not initialized.")
        
        model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        all_loss_components = {}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = self.loss_function(outputs, targets)
                
                # Accumulate loss components
                loss_components = self.loss_function.get_loss_components(outputs, targets)
                for key, value in loss_components.items():
                    if key not in all_loss_components:
                        all_loss_components[key] = 0.0
                    all_loss_components[key] += value
                
                # Calculate metrics
                predictions = torch.sigmoid(outputs)
                
                # Dice coefficient
                pred_flat = predictions.view(-1)
                target_flat = targets.view(-1)
                intersection = (pred_flat * target_flat).sum()
                dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
                
                # Accuracy
                pred_binary = (predictions > 0.5).float()
                accuracy = (pred_binary == targets).float().mean()
                
                total_loss += loss.item()
                total_dice += dice.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        # Average metrics
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_dice': total_dice / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
        
        # Add averaged loss components
        for key, value in all_loss_components.items():
            val_metrics[f'val_{key}'] = value / num_batches
        
        return val_metrics
    
    def end_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        End-of-epoch processing including loss scheduling and logging.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            Combined epoch summary
        """
        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        
        # Update loss scheduler if available
        if self.loss_scheduler:
            self.loss_scheduler.step(epoch, val_metrics)
        
        # Update training history
        self._update_training_history(epoch, epoch_metrics)
        
        # Log comprehensive epoch summary
        self._log_epoch_summary(epoch, epoch_metrics)
        
        return {
            'epoch': epoch,
            'metrics': epoch_metrics,
            'training_history': self.training_history[-10:],  # Last 10 epochs
            'checkpoint_summary': self.get_checkpoint_summary() if self.checkpoint_manager else None
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'model_info': {
                'model_name': self.model_name,
                'model_dir': str(self.model_dir),
                'current_epoch': self.current_epoch,
                'best_metric': self.best_metric
            },
            'loss_info': {},
            'checkpoint_info': {},
            'training_progress': {
                'total_epochs': len(self.training_history),
                'training_history': self.training_history
            }
        }
        
        # Loss function information
        if self.loss_function:
            summary['loss_info'] = {
                'config': self.loss_function.get_config(),
                'scheduler_active': self.loss_scheduler is not None
            }
            
            if self.loss_scheduler:
                summary['loss_info']['scheduler_history'] = getattr(
                    self.loss_scheduler, 'history', []
                )
        
        # Checkpoint information
        if self.checkpoint_manager:
            summary['checkpoint_info'] = self.checkpoint_manager.get_checkpoint_summary()
        
        return summary
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get checkpoint summary if manager is available."""
        if self.checkpoint_manager:
            return self.checkpoint_manager.get_checkpoint_summary()
        return {}
    
    def export_training_data(self, output_file: str = None):
        """Export complete training data to JSON."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.artifacts_dir / f"training_summary_{timestamp}.json"
        
        summary = self.get_training_summary()
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training data exported to: {output_file}")
        return output_file
    
    def _update_training_history(self, epoch: int, metrics: Dict[str, float]):
        """Update internal training history."""
        history_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.copy()
        }
        self.training_history.append(history_entry)
        
        # Update best metric
        monitor_metric = getattr(self.checkpoint_manager, 'monitor_metric', 'val_loss')
        if monitor_metric in metrics:
            current_value = metrics[monitor_metric]
            mode = getattr(self.checkpoint_manager, 'mode', 'min')
            
            if self.best_metric is None:
                self.best_metric = current_value
            elif ((mode == 'min' and current_value < self.best_metric) or
                  (mode == 'max' and current_value > self.best_metric)):
                self.best_metric = current_value
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """Log comprehensive epoch summary."""
        logger.info(f"=== Epoch {epoch} Summary ===")
        
        # Training metrics
        train_metrics = {k: v for k, v in metrics.items() if not k.startswith('val_')}
        val_metrics = {k: v for k, v in metrics.items() if k.startswith('val_')}
        
        if train_metrics:
            logger.info("Training Metrics:")
            for key, value in train_metrics.items():
                logger.info(f"  {key}: {value:.6f}")
        
        if val_metrics:
            logger.info("Validation Metrics:")
            for key, value in val_metrics.items():
                logger.info(f"  {key}: {value:.6f}")
        
        # Loss function weights (if applicable)
        if self.loss_function and hasattr(self.loss_function, 'weights'):
            logger.info("Current Loss Weights:")
            for name, weight in self.loss_function.weights.items():
                logger.info(f"  {name}: {weight:.3f}")
        
        logger.info("=" * 30)


class EnhancedTrainingManager(TrainingManager):
    """
    Enhanced training manager with additional features:
    - Automatic hyperparameter tuning
    - Advanced monitoring and alerting
    - Integration with MLflow
    """
    
    def __init__(self, *args, mlflow_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlflow_enabled = mlflow_enabled
        
        if mlflow_enabled:
            try:
                import mlflow
                self.mlflow = mlflow
                logger.info("MLflow integration enabled")
            except ImportError:
                logger.warning("MLflow not available, disabling MLflow features")
                self.mlflow_enabled = False
    
    def log_to_mlflow(self, metrics: Dict[str, float], epoch: int):
        """Log metrics to MLflow if enabled."""
        if not self.mlflow_enabled:
            return
        
        try:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step=epoch)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def setup_hyperparameter_tuning(self, tuning_config: Dict[str, Any]):
        """Setup hyperparameter tuning (placeholder for future implementation)."""
        logger.info("Hyperparameter tuning setup (feature coming soon)")
        pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_training_manager(model_dir: str, model_name: str,
                          loss_preset: str = 'default_segmentation',
                          checkpoint_preset: str = 'standard',
                          enhanced: bool = False) -> TrainingManager:
    """
    Convenience function to create a fully configured training manager.
    
    Args:
        model_dir: Directory for model outputs
        model_name: Name of the model
        loss_preset: Loss function preset name
        checkpoint_preset: Checkpoint strategy preset
        enhanced: Whether to use enhanced manager
        
    Returns:
        Configured training manager
    """
    # Create manager
    if enhanced:
        manager = EnhancedTrainingManager(model_dir, model_name)
    else:
        manager = TrainingManager(model_dir, model_name)
    
    # Setup loss function
    loss_config = get_preset_loss_config(loss_preset)
    manager.setup_loss_function(loss_config)
    
    # Setup checkpointing
    checkpoint_configs = {
        'standard': {'save_strategy': 'best', 'max_checkpoints': 5},
        'frequent': {'save_strategy': 'epoch', 'max_checkpoints': 10},
        'adaptive': {'manager_type': 'auto', 'auto_strategy': 'adaptive'},
        'performance': {'manager_type': 'auto', 'auto_strategy': 'performance_based'}
    }
    
    checkpoint_config = checkpoint_configs.get(checkpoint_preset, checkpoint_configs['standard'])
    manager.setup_checkpointing(checkpoint_config)
    
    logger.info(f"Training manager created with {loss_preset} loss and {checkpoint_preset} checkpointing")
    return manager


def quick_training_setup(model_dir: str, model_name: str) -> TrainingManager:
    """Quick setup with sensible defaults."""
    return create_training_manager(
        model_dir=model_dir,
        model_name=model_name,
        loss_preset='default_segmentation',
        checkpoint_preset='standard'
    )
