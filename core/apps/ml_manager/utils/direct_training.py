"""
Direct training manager for simplified training management.
Allows only one active training session at a time.
"""
import os
import sys
import subprocess
import threading
import time
import logging
import signal
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from django.utils import timezone
from django.db import transaction

from ..models import MLModel

logger = logging.getLogger(__name__)

class DirectTrainingManager:
    """
    Manages direct training - only one training at a time.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.current_training = None
            self.training_thread = None
            self.stop_event = threading.Event()
            self.initialized = True
    
    def is_training_active(self) -> bool:
        """Check if any training is currently active."""
        return (self.current_training is not None and 
                self.training_thread is not None and 
                self.training_thread.is_alive())
    
    def get_active_training_model_id(self) -> Optional[int]:
        """Get ID of currently training model."""
        if self.is_training_active() and self.current_training:
            return self.current_training.get('model_id')
        return None
    
    def start_training(self, model_id: int, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start training for a model. Only one training allowed at a time.
        """
        with self._lock:
            # Check if training is already active
            if self.is_training_active():
                active_model_id = self.get_active_training_model_id()
                return {
                    'success': False,
                    'error': f'Training already active for model {active_model_id}. Only one training session allowed at a time.'
                }
            
            # Check model exists and is not already training
            try:
                model = MLModel.objects.get(id=model_id)
                if model.status == 'training':
                    return {
                        'success': False,
                        'error': f'Model {model_id} is already marked as training'
                    }
            except MLModel.DoesNotExist:
                return {
                    'success': False,
                    'error': f'Model {model_id} not found'
                }
            
            # Set up training
            self.current_training = {
                'model_id': model_id,
                'config': training_config,
                'start_time': time.time(),
                'process': None
            }
            
            # Clear stop event
            self.stop_event.clear()
            
            # Update model status to training immediately
            with transaction.atomic():
                model = MLModel.objects.select_for_update().get(id=model_id)
                old_status = model.status
                model.status = 'training'
                model.training_started_at = timezone.now()
                model.save()
                logger.info(f"Model {model_id} status changed from '{old_status}' to 'training'")
            
            # Start training thread
            self.training_thread = threading.Thread(
                target=self._run_training,
                args=(model_id, training_config),
                daemon=True
            )
            self.training_thread.start()
            
            logger.info(f"Started direct training for model {model_id}")
            return {'success': True, 'message': 'Training started successfully'}
    
    def stop_training(self, model_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Stop active training.
        """
        with self._lock:
            if not self.is_training_active():
                return {'success': False, 'error': 'No active training to stop'}
            
            active_model_id = self.get_active_training_model_id()
            if model_id is not None and active_model_id != model_id:
                return {'success': False, 'error': f'No training active for model {model_id}'}
            
            # Set stop event
            self.stop_event.set()
            
            # Try to terminate process gracefully
            if self.current_training and self.current_training.get('process'):
                process = self.current_training['process']
                try:
                    # Send SIGTERM to process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    
                    # Wait a bit for graceful shutdown
                    time.sleep(5)
                    
                    # If still running, force kill
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        
                except Exception as e:
                    logger.warning(f"Error stopping training process: {e}")
            
            # Update model status
            try:
                with transaction.atomic():
                    model = MLModel.objects.select_for_update().get(id=active_model_id)
                    model.status = 'stopped'
                    model.save()
            except Exception as e:
                logger.error(f"Error updating model status after stop: {e}")
            
            logger.info(f"Stopped training for model {active_model_id}")
            return {'success': True, 'message': 'Training stopped successfully'}
    
    def get_training_status(self, model_id: int) -> Dict[str, Any]:
        """
        Get current training status for a model.
        """
        if not self.is_training_active():
            return {'active': False, 'message': 'No active training'}
        
        active_model_id = self.get_active_training_model_id()
        if active_model_id != model_id:
            return {'active': False, 'message': f'No training active for model {model_id}'}
        
        # Get current training info
        training_info = self.current_training
        elapsed_time = time.time() - training_info['start_time']
        
        return {
            'active': True,
            'model_id': model_id,
            'elapsed_time': elapsed_time,
            'start_time': training_info['start_time']
        }
    
    def _run_training(self, model_id: int, training_config: Dict[str, Any]):
        """
        Internal method to run training in a separate thread.
        """
        try:
            # Prepare training command
            training_command = self._prepare_training_command(model_id, training_config)
            
            logger.info(f"Starting training process for model {model_id}")
            logger.info(f"Command: {' '.join(training_command)}")
            
            # Start training process
            process = subprocess.Popen(
                training_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                preexec_fn=os.setsid  # Create process group for proper cleanup
            )
            
            # Store process reference
            if self.current_training:
                self.current_training['process'] = process
            
            # Monitor training output
            result = self._monitor_training_process(model_id, process)
            
            # Update final status
            with transaction.atomic():
                model = MLModel.objects.select_for_update().get(id=model_id)
                if result.get('success', False):
                    model.status = 'completed'
                    model.training_completed_at = timezone.now()
                    if 'model_path' in result:
                        model.model_file_path = result['model_path']
                    if 'best_metrics' in result:
                        model.training_metrics = result['best_metrics']
                else:
                    model.status = 'failed'
                    model.error_message = result.get('error', 'Training failed')
                model.save()
            
        except Exception as e:
            logger.error(f"Training failed for model {model_id}: {e}")
            
            # Update model with error
            try:
                with transaction.atomic():
                    model = MLModel.objects.select_for_update().get(id=model_id)
                    model.status = 'failed'
                    model.error_message = str(e)
                    model.save()
            except Exception as update_error:
                logger.error(f"Failed to update model status: {update_error}")
        
        finally:
            # Clean up
            with self._lock:
                self.current_training = None
                if self.training_thread == threading.current_thread():
                    self.training_thread = None
            
            logger.info(f"Training cleanup completed for model {model_id}")
    
    def _prepare_training_command(self, model_id: int, training_config: Dict[str, Any]) -> list:
        """Prepare the training command."""
        
        # Base training script path
        training_script = Path('/app/ml/training/train.py')
        
        # Base command
        cmd = [
            sys.executable,
            str(training_script),
            '--mode=train',
            '--model_id', str(model_id)
        ]
        
        # Add configuration parameters
        config = training_config
        
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
            cmd.extend(['--num-workers', str(config['num_workers'])])
        
        # Model parameters
        if 'model_type' in config:
            cmd.extend(['--model-type', config['model_type']])
        if 'input_channels' in config:
            cmd.extend(['--input-channels', str(config['input_channels'])])
        if 'num_classes' in config:
            cmd.extend(['--num-classes', str(config['num_classes'])])
        
        # Advanced parameters
        if 'scheduler' in config:
            cmd.extend(['--scheduler', config['scheduler']])
        if 'patience' in config:
            cmd.extend(['--patience', str(config['patience'])])
        if 'min_lr' in config:
            cmd.extend(['--min-lr', str(config['min_lr'])])
        if 'data_augmentation' in config:
            cmd.extend(['--data-augmentation', str(config['data_augmentation']).lower()])
        
        return cmd
    
    def _monitor_training_process(self, model_id: int, process: subprocess.Popen) -> Dict[str, Any]:
        """Monitor training process and parse output."""
        
        output_lines = []
        current_epoch = 0
        total_epochs = 0
        
        try:
            while True:
                # Check if we should stop
                if self.stop_event.is_set():
                    logger.info(f"Stop requested for model {model_id}")
                    return {'success': False, 'error': 'Training stopped by user'}
                
                # Read output
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                
                if output:
                    line = output.strip()
                    output_lines.append(line)
                    
                    # Parse training progress
                    try:
                        # Look for epoch information
                        if 'Epoch' in line and '/' in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'Epoch' and i + 1 < len(parts):
                                    epoch_info = parts[i + 1]
                                    if '/' in epoch_info:
                                        current_epoch, total_epochs = map(int, epoch_info.split('/'))
                                        break
                        
                        # Parse metrics (simplified)
                        metrics = {}
                        if 'Loss:' in line:
                            # Extract loss value
                            loss_idx = line.find('Loss:')
                            if loss_idx != -1:
                                loss_part = line[loss_idx:].split(',')[0]
                                try:
                                    metrics['train_loss'] = float(loss_part.split(':')[1].strip())
                                except:
                                    pass
                        
                        # Update model progress periodically
                        if current_epoch > 0 and total_epochs > 0:
                            try:
                                with transaction.atomic():
                                    model = MLModel.objects.select_for_update().get(id=model_id)
                                    model.current_epoch = current_epoch
                                    model.total_epochs = total_epochs
                                    if metrics:
                                        model.training_metrics = metrics
                                    model.save()
                            except Exception as e:
                                logger.warning(f"Failed to update model progress: {e}")
                        
                    except Exception as e:
                        logger.debug(f"Error parsing training output: {e}")
            
            # Check final result
            return_code = process.poll()
            if return_code == 0:
                return {
                    'success': True,
                    'message': 'Training completed successfully',
                    'output': '\n'.join(output_lines[-100:])  # Last 100 lines
                }
            else:
                return {
                    'success': False,
                    'error': f'Training failed with return code {return_code}',
                    'output': '\n'.join(output_lines[-100:])
                }
        
        except Exception as e:
            logger.error(f"Error monitoring training process: {e}")
            return {
                'success': False,
                'error': f'Error monitoring training: {str(e)}'
            }


# Global instance
training_manager = DirectTrainingManager()
