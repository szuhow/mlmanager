"""
Training utilities for managing ML training processes.
"""
import os
import sys
import subprocess
import signal
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import json

logger = logging.getLogger(__name__)

class TrainingController:
    """
    Controller for managing ML training processes with proper stopping mechanisms.
    """
    
    def __init__(self, model, training_config: Dict[str, Any]):
        self.model = model
        self.training_config = training_config
        self.process = None
        self.should_stop = threading.Event()
        self.training_thread = None
        
    def train_with_monitoring(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Start training with monitoring and progress updates.
        """
        try:
            # Prepare training command
            training_command = self._prepare_training_command()
            
            logger.info(f"Starting training for model {self.model.id}")
            logger.info(f"Training command: {' '.join(training_command)}")
            
            # Start training process
            self.process = subprocess.Popen(
                training_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                preexec_fn=os.setsid  # Create new process group for proper cleanup
            )
            
            # Monitor training process
            result = self._monitor_training_process(progress_callback)
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed for model {self.model.id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_training_command(self) -> list:
        """Prepare the training command with all necessary arguments."""
        
        # Base training script path - use absolute path within container
        # In Docker container, we know the structure is /app/ml/training/train.py
        training_script = Path('/app/ml/training/train.py')
        
        # Base command
        cmd = [
            sys.executable,
            str(training_script),
            '--mode=train',  # Required mode argument
            '--model_id', str(self.model.id)
        ]
        
        # Add training configuration parameters
        config = self.training_config
        
        # Dataset parameters
        if 'dataset_name' in config:
            cmd.extend(['--dataset_name', config['dataset_name']])
        if 'data_path' in config:
            cmd.extend(['--data-path', config['data_path']])
        if 'dataset_type' in config:
            cmd.extend(['--dataset-type', config['dataset_type']])
            
        # Training parameters
        if 'epochs' in config:
            cmd.extend(['--epochs', str(config['epochs'])])
        if 'batch_size' in config:
            cmd.extend(['--batch_size', str(config['batch_size'])])
        if 'learning_rate' in config:
            cmd.extend(['--learning_rate', str(config['learning_rate'])])
        if 'validation_split' in config:
            cmd.extend(['--validation-split', str(config['validation_split'])])
        if 'optimizer' in config:
            cmd.extend(['--optimizer', config['optimizer']])
        if 'device' in config:
            cmd.extend(['--device', config['device']])
        if 'crop_size' in config:
            cmd.extend(['--crop-size', str(config['crop_size'])])
        if 'num_workers' in config:
            # CPU training fix: force num_workers=0 to prevent deadlocks
            num_workers = config['num_workers']
            if config.get('device') == 'cpu' or config.get('device') == 'auto':
                num_workers = 0
                logger.info(f"[CPU_OPTIMIZATION] Forcing num_workers=0 for CPU training (was {config['num_workers']})")
            cmd.extend(['--num-workers', str(num_workers)])
        if 'resolution' in config:
            cmd.extend(['--resolution', str(config['resolution'])])
            
        # Model parameters
        if 'model_type' in config:
            cmd.extend(['--model_type', config['model_type']])
        if 'architecture' in config:
            cmd.extend(['--architecture', config['architecture']])
            
        # Loss function parameters (enhanced)
        if 'loss_function' in config:
            cmd.extend(['--loss_function', config['loss_function']])
        if 'dice_weight' in config:
            cmd.extend(['--dice_weight', str(config['dice_weight'])])
        if 'focal_alpha' in config:
            cmd.extend(['--focal_alpha', str(config['focal_alpha'])])
        if 'focal_gamma' in config:
            cmd.extend(['--focal_gamma', str(config['focal_gamma'])])
            
        # Regularization parameters
        if 'weight_decay' in config:
            cmd.extend(['--weight_decay', str(config['weight_decay'])])
        if 'dropout_rate' in config:
            cmd.extend(['--dropout_rate', str(config['dropout_rate'])])
            
        # Early stopping parameters
        if 'early_stopping_patience' in config:
            cmd.extend(['--early_stopping_patience', str(config['early_stopping_patience'])])
        if 'early_stopping_metric' in config:
            cmd.extend(['--early_stopping_metric', config['early_stopping_metric']])
            
        # Output directories
        output_dir = Path('data/models') / str(self.model.id)
        cmd.extend(['--output_dir', str(output_dir)])
        
        # MLflow tracking
        if 'mlflow_run_id' in config and config['mlflow_run_id']:
            cmd.extend(['--mlflow-run-id', config['mlflow_run_id']])
        elif hasattr(self.model, 'mlflow_run_id') and self.model.mlflow_run_id:
            cmd.extend(['--mlflow-run-id', self.model.mlflow_run_id])
        
        # Add stop file path for graceful stopping
        stop_file = output_dir / 'stop_training.flag'
        cmd.extend(['--stop_file', str(stop_file)])
        
        return cmd
    
    def _monitor_training_process(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Monitor the training process and handle output."""
        
        metrics = {}
        last_epoch = 0
        total_epochs = self.training_config.get('epochs', 100)
        output_lines = []  # Capture all output for debugging
        
        try:
            # Read output line by line
            for line in iter(self.process.stdout.readline, ''):
                if self.should_stop.is_set():
                    logger.info("Stop signal received, terminating training")
                    self._stop_process()
                    return {'success': False, 'error': 'Training stopped by user'}
                
                line = line.strip()
                if not line:
                    continue
                
                # Store output for debugging
                output_lines.append(line)
                logger.info(f"Training output: {line}")
                
                # Parse training output for progress and metrics
                parsed_info = self._parse_training_output(line)
                
                if parsed_info:
                    if 'epoch' in parsed_info:
                        last_epoch = parsed_info['epoch']
                    if 'metrics' in parsed_info:
                        metrics.update(parsed_info['metrics'])
                    
                    # Call progress callback if provided
                    if progress_callback and 'epoch' in parsed_info:
                        progress_callback(
                            epoch=last_epoch,
                            total_epochs=total_epochs,
                            metrics=metrics
                        )
            
            # Wait for process to complete
            return_code = self.process.wait()
            
            if return_code == 0:
                logger.info(f"Training completed successfully for model {self.model.id}")
                return {
                    'success': True,
                    'final_epoch': last_epoch,
                    'best_metrics': metrics,
                    'model_path': self._find_best_model_path()
                }
            else:
                # Log all captured output for debugging
                logger.error(f"Training failed with return code {return_code}")
                logger.error(f"Training output captured: {output_lines}")
                return {
                    'success': False,
                    'error': f'Training process failed with return code {return_code}. Last output: {output_lines[-10:] if output_lines else "No output captured"}',
                    'last_epoch': last_epoch,
                    'metrics': metrics
                }
                
        except Exception as e:
            logger.error(f"Error monitoring training process: {e}")
            self._stop_process()
            return {'success': False, 'error': str(e)}
    
    def _parse_training_output(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse training output line for epoch, metrics, etc."""
        
        parsed = {}
        
        # Look for epoch information in multiple formats
        if 'Epoch' in line and '/' in line:
            try:
                # Format 1: "[EPOCH] 9/10 COMPLETED"
                if '[EPOCH]' in line:
                    epoch_part = line.split('[EPOCH]')[1].strip()
                    if '/' in epoch_part:
                        epoch_info = epoch_part.split()[0]  # Get "9/10"
                        current_epoch = int(epoch_info.split('/')[0])
                        total_epochs = int(epoch_info.split('/')[1])
                        parsed['epoch'] = current_epoch
                        parsed['total_epochs'] = total_epochs
                
                # Format 2: "Epoch 5/100" (legacy)
                else:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower().startswith('epoch'):
                            epoch_info = parts[i+1] if i+1 < len(parts) else part
                            if '/' in epoch_info:
                                current_epoch = int(epoch_info.split('/')[0])
                                parsed['epoch'] = current_epoch
                            break
            except (ValueError, IndexError):
                pass
        
        # Look for metrics (loss, accuracy, dice score, etc.)
        metrics = {}
        
        # Enhanced metric patterns for the current log format
        metric_patterns = [
            # Current format: "Train Loss: 0.8198, Train Dice: 0.0077"
            ('Train Loss:', 'train_loss'),
            ('Train Dice:', 'train_dice'),
            ('Val Loss:', 'val_loss'),
            ('Val Dice:', 'val_dice'),
            ('Best Val Dice:', 'best_val_dice'),
            ('Learning Rate:', 'learning_rate'),
            # Legacy patterns
            ('loss:', 'loss'),
            ('dice:', 'dice_score'),
            ('accuracy:', 'accuracy'),
            ('iou:', 'iou'),
            ('val_loss:', 'val_loss'),
            ('val_dice:', 'val_dice_score'),
            ('val_accuracy:', 'val_accuracy'),
            ('lr:', 'learning_rate')
        ]
        
        for pattern, metric_name in metric_patterns:
            if pattern in line:
                try:
                    # Find the value after the pattern
                    start_idx = line.find(pattern) + len(pattern)
                    remaining = line[start_idx:].strip()
                    
                    # Extract the numeric value (handle comma-separated values)
                    value_str = ''
                    for char in remaining:
                        if char.isdigit() or char in '.e-+':
                            value_str += char
                        elif char == ',' and value_str:  # Stop at comma
                            break
                        elif char == ' ' and value_str:  # Stop at space
                            break
                        elif value_str and not char.isspace():  # Stop at non-numeric, non-space
                            break
                    
                    if value_str:
                        metrics[metric_name] = float(value_str)
                except (ValueError, IndexError):
                    continue
        
        if metrics:
            parsed['metrics'] = metrics
        
        return parsed if parsed else None
    
    def _find_best_model_path(self) -> Optional[str]:
        """Find the path to the best saved model."""
        model_dir = Path('data/models') / str(self.model.id)
        
        # Look for best model files
        best_model_patterns = [
            'best_model*.pth',
            'model_best*.pth',
            'checkpoint_best*.pth'
        ]
        
        for pattern in best_model_patterns:
            model_files = list(model_dir.glob(pattern))
            if model_files:
                return str(model_files[0])  # Return the first match
        
        # Fallback: look for any .pth file
        model_files = list(model_dir.glob('*.pth'))
        if model_files:
            # Sort by modification time and return the newest
            newest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            return str(newest_file)
        
        return None
    
    def stop_training(self):
        """Request training to stop gracefully."""
        logger.info(f"Stopping training for model {self.model.id}")
        
        # Set stop event
        self.should_stop.set()
        
        # Create stop file for the training script to detect
        try:
            output_dir = Path('data/models') / str(self.model.id)
            output_dir.mkdir(parents=True, exist_ok=True)
            stop_file = output_dir / 'stop_training.flag'
            stop_file.touch()
            logger.info(f"Created stop file: {stop_file}")
        except Exception as e:
            logger.warning(f"Could not create stop file: {e}")
        
        # Give the process some time to stop gracefully
        if self.process:
            try:
                self.process.wait(timeout=30)  # Wait up to 30 seconds
            except subprocess.TimeoutExpired:
                logger.warning("Training process did not stop gracefully, forcing termination")
                self._stop_process()
    
    def _stop_process(self):
        """Force stop the training process."""
        if self.process:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                try:
                    # Force kill if SIGTERM didn't work
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            
            self.process = None


class EnhancedLossFunction:
    """
    Enhanced loss functions for better handling of class imbalance and noise reduction.
    """
    
    @staticmethod
    def combined_dice_focal_loss(
        outputs, 
        targets, 
        dice_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6
    ):
        """
        Combined Dice and Focal Loss for better segmentation performance.
        
        Args:
            outputs: Model predictions (logits)
            targets: Ground truth masks
            dice_weight: Weight for dice loss component (0.0-1.0)
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            smooth: Smoothing factor for dice loss
        """
        import torch
        import torch.nn.functional as F
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)
        
        # Dice Loss
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score.mean()
        
        # Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = focal_alpha * (1 - pt) ** focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # Combined loss
        total_loss = dice_weight * dice_loss + (1 - dice_weight) * focal_loss
        
        return total_loss, {
            'dice_loss': dice_loss.item(),
            'focal_loss': focal_loss.item(),
            'total_loss': total_loss.item()
        }
    
    @staticmethod
    def tversky_loss(
        outputs, 
        targets, 
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6
    ):
        """
        Tversky Loss - good for handling class imbalance.
        
        Args:
            outputs: Model predictions
            targets: Ground truth
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        import torch
        
        probs = torch.sigmoid(outputs)
        
        # True positives, false positives, false negatives
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        tversky_loss = 1 - tversky.mean()
        
        return tversky_loss


def create_enhanced_training_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create enhanced training configuration with better defaults for noise reduction.
    """
    enhanced_config = base_config.copy()
    
    # Enhanced loss function configuration
    if 'loss_function' not in enhanced_config:
        enhanced_config['loss_function'] = 'combined_dice_focal'
    
    # Dice loss weight (higher = more emphasis on segmentation quality)
    if 'dice_weight' not in enhanced_config:
        enhanced_config['dice_weight'] = 0.7
    
    # Focal loss parameters (helps with hard examples)
    if 'focal_alpha' not in enhanced_config:
        enhanced_config['focal_alpha'] = 0.25
    if 'focal_gamma' not in enhanced_config:
        enhanced_config['focal_gamma'] = 2.0
    
    # Enhanced regularization
    if 'weight_decay' not in enhanced_config:
        enhanced_config['weight_decay'] = 1e-4
    if 'dropout_rate' not in enhanced_config:
        enhanced_config['dropout_rate'] = 0.1
    
    # Early stopping configuration
    if 'early_stopping_patience' not in enhanced_config:
        enhanced_config['early_stopping_patience'] = 15
    if 'early_stopping_metric' not in enhanced_config:
        enhanced_config['early_stopping_metric'] = 'val_dice_score'
    
    # Learning rate scheduling
    if 'lr_scheduler' not in enhanced_config:
        enhanced_config['lr_scheduler'] = 'reduce_on_plateau'
    if 'lr_patience' not in enhanced_config:
        enhanced_config['lr_patience'] = 5
    if 'lr_factor' not in enhanced_config:
        enhanced_config['lr_factor'] = 0.5
    
    # Data augmentation for noise robustness
    if 'augmentation_probability' not in enhanced_config:
        enhanced_config['augmentation_probability'] = 0.5
    
    return enhanced_config
