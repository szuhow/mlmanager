# Celery tasks for ML training and inference
from celery import shared_task
from celery.utils.log import get_task_logger
import os
import sys
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import traceback
from django.conf import settings
from datetime import datetime
import threading
import time
import subprocess
import json
import re
import signal
import fcntl
import select
import psutil
import mlflow
from contextlib import contextmanager

logger = get_task_logger(__name__)

# Global flag for training interruption
training_stop_flags = {}

@contextmanager
def managed_process(cmd_args, model_id, timeout=3600):
    """Context manager for safe subprocess management"""
    process = None
    try:
        # Create process with proper settings for real-time output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # Force Python to be unbuffered
        
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for unified processing
            universal_newlines=True,
            bufsize=0,  # Unbuffered for real-time output
            preexec_fn=os.setsid,  # Create new process group
            env=env
        )
        
        # Set non-blocking I/O only for stdout (since stderr is merged)
        fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
        
        logger.info(f"Started training process PID: {process.pid} for model {model_id}")
        yield process
        
    except Exception as e:
        logger.error(f"Error in managed process: {e}")
        raise
    finally:
        if process:
            cleanup_process(process, model_id)

def cleanup_process(process, model_id):
    """Properly cleanup subprocess and its children"""
    if process.poll() is None:  # Process is still running
        try:
            # First, try graceful termination
            logger.info(f"Terminating process group for model {model_id}")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                logger.info(f"Process {process.pid} terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                logger.warning(f"Force killing process group for model {model_id}")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
                
        except (OSError, ProcessLookupError) as e:
            logger.warning(f"Error during process cleanup: {e}")

def monitor_process_resources(process, model_id):
    """Monitor process resources and return metrics"""
    try:
        psutil_process = psutil.Process(process.pid)
        cpu_percent = psutil_process.cpu_percent()
        memory_info = psutil_process.memory_info()
        
        # Log resource usage
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Model {model_id} - CPU: {cpu_percent}%, Memory: {memory_mb:.2f} MB")
        
        # Check memory limit (8GB)
        if memory_info.rss > 8 * 1024 * 1024 * 1024:
            logger.warning(f"Model {model_id} exceeding memory limit: {memory_mb:.2f} MB")
            return False
        
        return True
        
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def read_process_output(process, model_id, progress_callback=None, timeout=3600):
    """Safely read process output with proper error handling and real-time updates"""
    start_time = time.time()
    last_output_time = start_time
    output_buffer = []
    
    # Keep a complete log for the model
    training_log = []
    
    # Track current training state for consistent callback data
    current_epoch = 0
    total_epochs = 0
    
    while process.poll() is None:
        current_time = time.time()
        
        # Check timeout
        if current_time - start_time > timeout:
            logger.error(f"Training process timed out for model {model_id}")
            return False, "Training process timed out"
        
        # Check stop flag
        if training_stop_flags.get(model_id, threading.Event()).is_set():
            logger.info(f"Training stop requested for model {model_id}")
            return False, "Training stopped by user"
        
        # Check if process is responsive (no output for too long)
        if current_time - last_output_time > 600:  # 10 minutes without output
            logger.warning(f"No output from training process for model {model_id} for 10 minutes")
        
        # Monitor resources every 30 seconds
        if int(current_time) % 30 == 0:
            if not monitor_process_resources(process, model_id):
                logger.error(f"Resource monitoring failed for model {model_id}")
                return False, "Resource monitoring failed"
        
        # Use select for non-blocking read (only stdout since stderr is merged)
        try:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)  # Short timeout for responsiveness
            
            if ready:
                try:
                    line = process.stdout.readline()
                    if line:
                        last_output_time = current_time
                        line = line.strip()
                        
                        # Log immediately for real-time visibility
                        logger.info(f"[Model {model_id}] {line}")
                        output_buffer.append(line)
                        training_log.append(line)
                        
                        # Process metrics for callback
                        if progress_callback and line:
                            try:
                                metrics = parse_training_output_line(line, model_id)
                                if metrics:
                                    # Update tracked epoch info
                                    if 'epoch' in metrics:
                                        current_epoch = metrics['epoch']
                                    if 'total_epochs' in metrics:
                                        total_epochs = metrics['total_epochs']
                                    
                                    # Ensure epoch info is always available in callback
                                    if current_epoch >= 0 and total_epochs > 0:
                                        metrics['epoch'] = current_epoch
                                        metrics['total_epochs'] = total_epochs
                                    
                                    # If parsing returns any metrics, trigger the callback
                                    # to update the frontend state immediately.
                                    progress_callback(**metrics)
                                    
                            except Exception as e:
                                logger.error(f"Error in progress callback: {e}")
                                logger.error(f"Failed to parse line: {line}")
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                
                                # Try to send a basic progress update despite parsing error
                                try:
                                    basic_update = {
                                        'status': 'training_active', 
                                        'message': "Training in progress"
                                    }
                                    if current_epoch >= 0 and total_epochs > 0:
                                        basic_update['epoch'] = current_epoch
                                        basic_update['total_epochs'] = total_epochs
                                    progress_callback(**basic_update)
                                except:
                                    pass  # If callback completely fails, don't stop training
                        
                except IOError:
                    # Non-blocking read can raise IOError when no data available
                    pass
            else:
                # No data available, small sleep to prevent busy waiting
                time.sleep(0.1)
                    
        except (select.error, OSError) as e:
            logger.error(f"Select error for model {model_id}: {e}")
            break
    
    # Process finished, get return code
    return_code = process.wait()
    
    # Get the log file path instead of returning full log content
    try:
        from core.apps.ml_manager.models import MLModel
        model = MLModel.objects.get(id=model_id)
        
        # Try to find the log file path
        log_file_path = None
        if model.model_directory:
            potential_log_paths = [
                os.path.join(model.model_directory, 'logs', 'training.log'),
                os.path.join(model.model_directory, 'training.log'),
                os.path.join(model.model_directory, 'artifacts', 'training.log')
            ]
            for path in potential_log_paths:
                if os.path.exists(path):
                    log_file_path = path
                    break
        
        # If no log file found, create a summary
        if not log_file_path:
            log_file_path = f"Training logs for model {model_id} - check model directory"
            
    except Exception as e:
        logger.warning(f"Could not determine log file path for model {model_id}: {e}")
        log_file_path = f"Training logs for model {model_id} - path unavailable"
    
    if return_code == 0:
        return True, "Training completed successfully", log_file_path
    else:
        error_msg = f"Training failed with return code {return_code}"
        if output_buffer:  # Use output_buffer instead of error_buffer since stderr is merged
            error_msg += f". Last output: {'; '.join(output_buffer[-5:])}"
        return False, error_msg, log_file_path

def parse_training_output_line(line: str, model_id: int) -> Dict[str, Any]:
    """
    Parse training output line to extract metrics for callback with enhanced real-time updates
    """
    metrics = {}
    
    # Simple epoch pattern matching with multiple formats
    epoch_patterns = [
        r'[Ee]poch\s*(\d+)[/\s]+(\d+)',  # "Epoch 1/10" or "epoch 1 10"
        r'(\d+)/(\d+)\s*[Ee]poch',        # "1/10 epoch"
        r'\[EPOCH\]\s*(\d+)/(\d+)',       # "[EPOCH] 1/10"
        r'Starting epoch\s*(\d+)[/\s]+(\d+)',  # "Starting epoch 1/10"
    ]
    
    for pattern in epoch_patterns:
        match = re.search(pattern, line)
        if match:
            current_epoch = int(match.group(1))
            total_epochs = int(match.group(2))
            metrics['epoch'] = current_epoch - 1  # Make 0-indexed for internal use
            metrics['total_epochs'] = total_epochs
            metrics['progress_percent'] = min(100, int((current_epoch / total_epochs) * 100))
            
            if '[EPOCH]' in line and 'COMPLETED' in line:
                metrics['status'] = 'epoch_completed'
                metrics['message'] = f'Completed epoch {current_epoch}/{total_epochs}'
            else:
                metrics['status'] = 'epoch_active'
                metrics['message'] = f'Training epoch {current_epoch}/{total_epochs}'
            
            # Real-time database update
            try:
                from core.apps.ml_manager.models import MLModel
                model = MLModel.objects.get(id=model_id)
                model.current_epoch = current_epoch
                model.total_epochs = total_epochs
                model.save(update_fields=['current_epoch', 'total_epochs'])
                logger.debug(f"Parsed epoch info: Model {model_id} epoch {current_epoch}/{total_epochs}")
            except Exception as e:
                logger.warning(f"Failed to update model epoch: {e}")
            break
    
    # Extract batch information for real-time progress
    if '[TRAIN]' in line and ('Batch' in line or 'batch' in line):
        try:
            # Format: "[TRAIN] Epoch 1/1 - Batch 30/125 - Loss: 0.8942, Dice: 0.1648"
            if 'Epoch' in line and 'Batch' in line:
                # Extract epoch and batch info
                parts = line.split('[TRAIN]')[1].strip()
                
                # Extract epoch info
                if ' - ' in parts:
                    epoch_part = parts.split(' - ')[0]  # "Epoch 1/1"
                    batch_part = parts.split(' - ')[1]  # "Batch 30/125"
                    
                    if 'Epoch' in epoch_part and '/' in epoch_part:
                        epoch_numbers = epoch_part.replace('Epoch', '').strip()
                        current_epoch = int(epoch_numbers.split('/')[0])
                        total_epochs = int(epoch_numbers.split('/')[1])
                        
                    if 'Batch' in batch_part and '/' in batch_part:
                        batch_numbers = batch_part.replace('Batch', '').strip().split(' ')[0]  # Remove trailing text
                        current_batch = int(batch_numbers.split('/')[0])
                        total_batches = int(batch_numbers.split('/')[1])
                        
                        # Calculate progress
                        epoch_progress = (current_epoch - 1) / total_epochs  # Previous epochs
                        batch_progress = current_batch / total_batches / total_epochs  # Current batch progress
                        total_progress = (epoch_progress + batch_progress) * 100
                        
                        metrics['epoch'] = current_epoch - 1  # Make 0-indexed
                        metrics['total_epochs'] = total_epochs
                        metrics['current_batch'] = current_batch
                        metrics['total_batches'] = total_batches
                        metrics['progress_percent'] = min(100, int(total_progress))
                        metrics['status'] = 'batch_processing'
                        metrics['message'] = f'Epoch {current_epoch}/{total_epochs} - Batch {current_batch}/{total_batches}'
                        
                        # Real-time database update
                        try:
                            from core.apps.ml_manager.models import MLModel
                            model = MLModel.objects.get(id=model_id)
                            model.current_epoch = current_epoch
                            model.total_epochs = total_epochs
                            model.current_batch = current_batch
                            model.total_batches_per_epoch = total_batches
                            model.save(update_fields=['current_epoch', 'total_epochs', 'current_batch', 'total_batches_per_epoch'])
                            logger.debug(f"Real-time update: Model {model_id} - Epoch {current_epoch}/{total_epochs}, Batch {current_batch}/{total_batches}")
                        except Exception as e:
                            logger.warning(f"Failed to update model batch progress in real-time: {e}")
                    
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse batch info from line: {line[:100]}, error: {e}")
    
    # Extract epoch information from actual log format
    if '[EPOCH]' in line and ('COMPLETED' in line or 'Starting epoch' in line):
        try:
            # Format: "[EPOCH] 1/1 COMPLETED - Train Loss: 0.9094, Train Dice: 0.1497, Val Loss: 0.8883, Val Dice: 0.1803"
            # OR: "[EPOCH] Starting epoch 1/10"
            if 'COMPLETED' in line:
                # Extract from completion format
                epoch_part = line.split('[EPOCH]')[1].split('COMPLETED')[0].strip()
                if '/' in epoch_part:
                    current_epoch = int(epoch_part.split('/')[0])
                    total_epochs = int(epoch_part.split('/')[1])
                    metrics['epoch'] = current_epoch - 1  # Make 0-indexed for internal use
                    metrics['total_epochs'] = total_epochs
                    # Add progress percentage
                    metrics['progress_percent'] = min(100, int((current_epoch / total_epochs) * 100))
                    metrics['status'] = 'epoch_completed'
                    metrics['message'] = f'Completed epoch {current_epoch}/{total_epochs}'
                    
                    # Force database update for real-time refresh
                    try:
                        from core.apps.ml_manager.models import MLModel
                        model = MLModel.objects.get(id=model_id)
                        model.current_epoch = current_epoch
                        model.total_epochs = total_epochs
                        model.save(update_fields=['current_epoch', 'total_epochs'])
                        logger.info(f"Real-time update: Model {model_id} epoch {current_epoch}/{total_epochs} COMPLETED")
                    except Exception as e:
                        logger.warning(f"Failed to update model epoch in real-time: {e}")
            else:
                # Format: "[EPOCH] Starting epoch 1/10"
                epoch_part = line.split('Starting epoch')[1].strip()
                if '/' in epoch_part:
                    current_epoch = int(epoch_part.split('/')[0])
                    total_epochs = int(epoch_part.split('/')[1])
                    metrics['epoch'] = current_epoch - 1  # Make 0-indexed for internal use
                    metrics['total_epochs'] = total_epochs
                    # Add progress percentage
                    metrics['progress_percent'] = min(100, int((current_epoch / total_epochs) * 100))
                    metrics['status'] = 'epoch_started'
                    metrics['message'] = f'Starting epoch {current_epoch}/{total_epochs}'
                    
                    # Force database update for real-time refresh
                    try:
                        from core.apps.ml_manager.models import MLModel
                        model = MLModel.objects.get(id=model_id)
                        model.current_epoch = current_epoch
                        model.total_epochs = total_epochs
                        model.save(update_fields=['current_epoch', 'total_epochs'])
                        logger.info(f"Real-time update: Model {model_id} epoch {current_epoch}/{total_epochs}")
                    except Exception as e:
                        logger.warning(f"Failed to update model epoch in real-time: {e}")
                    
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse epoch info from line: {line[:100]}, error: {e}")
    
    # Extract training batch information with real-time batch updates
    elif '[TRAIN]' in line and ('Epoch' in line and 'Batch' in line):
        try:
            # Format: "[TRAIN] Epoch 1/1 - Batch 30/125 (24.0%) - Loss: 0.9258, Dice: 0.1259"
            parts = line.split(' - ')
            
            # Extract epoch info
            if 'Epoch' in parts[0]:
                epoch_part = parts[0].split('Epoch')[1].strip()
                if '/' in epoch_part:
                    current_epoch = int(epoch_part.split('/')[0])
                    total_epochs = int(epoch_part.split('/')[1])
                    metrics['epoch'] = current_epoch - 1  # Make 0-indexed
                    metrics['total_epochs'] = total_epochs
                    metrics['progress_percent'] = min(100, int((current_epoch / total_epochs) * 100))
                    
                    # Update epoch in database
                    try:
                        from core.apps.ml_manager.models import MLModel
                        model = MLModel.objects.get(id=model_id)
                        if model.current_epoch != current_epoch or model.total_epochs != total_epochs:
                            model.current_epoch = current_epoch
                            model.total_epochs = total_epochs
                            model.save(update_fields=['current_epoch', 'total_epochs'])
                            logger.info(f"Real-time epoch update: Model {model_id} epoch {current_epoch}/{total_epochs}")
                    except Exception as e:
                        logger.warning(f"Failed to update model epoch in real-time: {e}")
            
            # Extract batch info
            if len(parts) > 1 and 'Batch' in parts[1]:
                batch_part = parts[1].split('Batch')[1].strip()
                batch_info = batch_part.split('(')[0].strip()  # Get "30/125"
                if '/' in batch_info:
                    current_batch = int(batch_info.split('/')[0])
                    total_batches = int(batch_info.split('/')[1])
                    metrics['current_batch'] = current_batch
                    metrics['total_batches_per_epoch'] = total_batches
                    
                    # Calculate batch progress
                    batch_progress = min(100, int((current_batch / total_batches) * 100))
                    metrics['batch_progress'] = batch_progress
                    
                    # Real-time batch update to database
                    try:
                        from core.apps.ml_manager.models import MLModel
                        model = MLModel.objects.get(id=model_id)
                        model.current_batch = current_batch
                        model.total_batches_per_epoch = total_batches
                        # Update every 10th batch or at beginning/end to reduce database load
                        if current_batch % 10 == 0 or current_batch == 1 or current_batch == total_batches:
                            model.save(update_fields=['current_batch', 'total_batches_per_epoch'])
                            logger.info(f"Real-time batch update: Model {model_id} batch {current_batch}/{total_batches} ({batch_progress}%)")
                    except Exception as e:
                        logger.warning(f"Failed to update model batch in real-time: {e}")
            
            # Extract loss and metrics with real-time updates
            train_loss = None
            train_dice = None
            for part in parts:
                if 'Loss:' in part:
                    try:
                        loss_value = float(part.split('Loss:')[1].split(',')[0].strip())
                        metrics['train_loss'] = loss_value
                        train_loss = loss_value
                    except (ValueError, IndexError):
                        pass
                if 'Dice:' in part:
                    try:
                        dice_value = float(part.split('Dice:')[1].split(',')[0].strip())
                        metrics['train_dice'] = dice_value  
                        train_dice = dice_value
                    except (ValueError, IndexError):
                        pass
            
            # Update training metrics in database if we have them
            if train_loss is not None or train_dice is not None:
                try:
                    from core.apps.ml_manager.models import MLModel
                    model = MLModel.objects.get(id=model_id)
                    if train_loss is not None:
                        model.train_loss = train_loss
                    if train_dice is not None:
                        model.train_dice = train_dice
                    model.save(update_fields=['train_loss', 'train_dice'])
                    logger.debug(f"Real-time metrics update: Model {model_id} - Loss: {train_loss}, Dice: {train_dice}")
                except Exception as e:
                    logger.warning(f"Failed to update model metrics in real-time: {e}")
                    
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse batch info from line: {line[:100]}, error: {e}")
    
    # Extract epoch completion metrics
    elif '[EPOCH]' in line and 'COMPLETED' in line:
        try:
            # Format: "[EPOCH] 1/1 COMPLETED - Train Loss: 0.9094, Train Dice: 0.1497, Val Loss: 0.8883, Val Dice: 0.1803"
            if 'COMPLETED - ' in line:
                metrics_part = line.split('COMPLETED - ')[1]
                metric_pairs = metrics_part.split(', ')
                
                parsed_metrics = {}
                for pair in metric_pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        try:
                            parsed_metrics[key] = float(value.strip())
                            metrics[key] = float(value.strip())
                        except ValueError:
                            continue
                
                # Update metrics in database
                try:
                    from core.apps.ml_manager.models import MLModel
                    model = MLModel.objects.get(id=model_id)
                    
                    # Update all available metrics
                    update_fields = []
                    if 'train_loss' in parsed_metrics:
                        model.train_loss = parsed_metrics['train_loss']
                        update_fields.append('train_loss')
                    if 'train_dice' in parsed_metrics:
                        model.train_dice = parsed_metrics['train_dice']
                        update_fields.append('train_dice')
                    if 'val_loss' in parsed_metrics:
                        model.val_loss = parsed_metrics['val_loss']
                        update_fields.append('val_loss')
                    if 'val_dice' in parsed_metrics:
                        model.val_dice = parsed_metrics['val_dice']
                        update_fields.append('val_dice')
                    if 'train_iou' in parsed_metrics:
                        model.train_iou = parsed_metrics['train_iou']
                        update_fields.append('train_iou')
                    if 'val_iou' in parsed_metrics:
                        model.val_iou = parsed_metrics['val_iou']
                        update_fields.append('val_iou')
                    
                    if update_fields:
                        model.save(update_fields=update_fields)
                        logger.info(f"Real-time epoch completion metrics: Model {model_id} - {parsed_metrics}")
                        
                except Exception as e:
                    logger.warning(f"Failed to update epoch completion metrics: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to parse epoch completion from line: {line[:100]}, error: {e}")
    
    # Extract validation information with real-time updates
    elif '[VAL]' in line and ('Loss:' in line or 'Dice:' in line):
        try:
            # Extract validation metrics
            val_loss = None
            val_dice = None
            
            if 'Loss:' in line:
                loss_value = extract_numeric_value(line, 'Loss:')
                if loss_value is not None:
                    metrics['val_loss'] = loss_value
                    val_loss = loss_value
            
            if 'Dice:' in line:
                dice_value = extract_numeric_value(line, 'Dice:')
                if dice_value is not None:
                    metrics['val_dice'] = dice_value
                    val_dice = dice_value
            
            # Real-time validation metrics update to database
            if val_loss is not None or val_dice is not None:
                try:
                    from core.apps.ml_manager.models import MLModel
                    model = MLModel.objects.get(id=model_id)
                    update_fields = []
                    if val_loss is not None:
                        model.val_loss = val_loss
                        update_fields.append('val_loss')
                    if val_dice is not None:
                        model.val_dice = val_dice
                        update_fields.append('val_dice')
                        # Update best validation dice if current is better
                        if val_dice > model.best_val_dice:
                            model.best_val_dice = val_dice
                            update_fields.append('best_val_dice')
                    if update_fields:
                        model.save(update_fields=update_fields)
                        logger.info(f"Real-time validation update: Model {model_id} val_loss={val_loss}, val_dice={val_dice}")
                except Exception as e:
                    logger.warning(f"Failed to update model validation metrics in real-time: {e}")
            
            metrics['status'] = 'validation_active'
            metrics['message'] = 'Validation in progress'
            
        except (ValueError, IndexError):
            pass
    
    # Extract loss and metrics
    train_loss = None
    val_loss = None
    train_dice = None
    val_dice = None
    
    # Common patterns to extract metrics
    if 'Train Loss:' in line:
        train_loss = extract_numeric_value(line, 'Train Loss:')
    if 'Val Loss:' in line:
        val_loss = extract_numeric_value(line, 'Val Loss:')
    if 'Train Dice:' in line:
        train_dice = extract_numeric_value(line, 'Train Dice:')
    if 'Val Dice:' in line:
        val_dice = extract_numeric_value(line, 'Val Dice:')
    
    # Alternative patterns (legacy)
    if train_loss is None and 'train_loss:' in line.lower():
        train_loss = extract_numeric_value(line.lower(), 'train_loss:')
    if val_loss is None and 'val_loss:' in line.lower():
        val_loss = extract_numeric_value(line.lower(), 'val_loss:')
    if train_dice is None and 'train_dice:' in line.lower():
        train_dice = extract_numeric_value(line.lower(), 'train_dice:')
    if val_dice is None and 'val_dice:' in line.lower():
        val_dice = extract_numeric_value(line.lower(), 'val_dice:')
    
    # Additional patterns for training output based on actual log format
    if '[TRAINING] Starting training with parameters:' in line:
        metrics['status'] = 'training_started'
        metrics['message'] = 'Training process initiated'
    elif "Status updated to 'training'" in line or "starting model training" in line.lower():
        metrics['status'] = 'training_active'
        metrics['message'] = 'Model training started'
    elif '[EPOCH] Starting epoch' in line:
        metrics['status'] = 'epoch_started'
        # Already handled above in epoch parsing
    elif 'training started' in line.lower() or 'starting epoch' in line.lower():
        metrics['status'] = 'training_started'
        metrics['message'] = 'Training started'
    elif 'training completed' in line.lower() or 'finished training' in line.lower():
        metrics['status'] = 'training_completed'
        metrics['message'] = 'Training completed'
    elif 'model saved' in line.lower() or 'saving model' in line.lower() or 'checkpoint saved' in line.lower():
        metrics['status'] = 'model_saved'
        metrics['message'] = 'Model checkpoint saved'
    elif '[MODEL CONFIG]' in line or '[DATASET]' in line:
        # Configuration and setup phases
        metrics['status'] = 'initializing'
        metrics['message'] = 'Setting up training configuration'
    elif '[TRAIN BATCH]' in line:
        # Training batch processing
        metrics['status'] = 'training_active'
        metrics['message'] = 'Processing training batch'
    elif '[VAL]' in line:
        # Validation phase
        metrics['status'] = 'validation_active'
        metrics['message'] = 'Running validation'
    elif any(keyword in line for keyword in ['[TRAIN]', '[MODEL]', '[CONFIG]']):
        # Generic training activity detected
        metrics['status'] = 'training_active'
        metrics['message'] = 'Training in progress'
    
    # Add extracted metrics
    if train_loss is not None:
        metrics['train_loss'] = train_loss
    if val_loss is not None:
        metrics['val_loss'] = val_loss
    if train_dice is not None:
        metrics['train_dice'] = train_dice
    if val_dice is not None:
        metrics['val_dice'] = val_dice
    
    return metrics

def extract_numeric_value(line: str, pattern: str) -> Optional[float]:
    """Extract numeric value from line after a pattern"""
    try:
        start_idx = line.find(pattern) + len(pattern)
        remaining = line[start_idx:].strip()
        
        # Extract the numeric value using regex
        import re
        match = re.search(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', remaining)
        if match:
            value_str = match.group(1)
            return float(value_str)
    except (ValueError, IndexError, AttributeError):
        pass
    
    return None

def run_training_direct(model_id: int, training_params: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    """
    Run training directly using subprocess with proper process management.
    Implements best practices for subprocess handling in production.
    """
    logger.info(f"Starting secure training process for model {model_id}")
    
    mlflow_run_id = None
    result = {'success': False, 'error': 'Unknown error'}
    
    try:
        # Set up Django environment
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.container')

        # Find training script
        training_script_path = Path(__file__).parent.parent / 'training' / 'train.py'
        if not training_script_path.exists():
            logger.error(f"Training script not found at {training_script_path}")
            result = {'success': False, 'error': f"Training script not found at {training_script_path}"}
            return result
        
        # Validate required parameters
        if not training_params.get('data_path'):
            logger.error(f"Missing required parameter: data_path")
            result = {'success': False, 'error': 'Missing required parameter: data_path'}
            return result
        
        # Validate and clean training parameters
        try:
            cleaned_params = validate_training_params(training_params)
            logger.info(f"Validated training parameters for model {model_id}: {cleaned_params}")
        except ValueError as e:
            logger.error(f"Parameter validation failed for model {model_id}: {str(e)}")
            result = {'success': False, 'error': str(e)}
            return result
        
        # Build command arguments
        cmd_args = build_training_command(training_script_path, model_id, cleaned_params)
        
        # Debug: Log all command arguments
        logger.info(f"Training command arguments for model {model_id}:")
        for i, arg in enumerate(cmd_args):
            logger.info(f"  [{i}] {arg}")
        
        # Set up MLflow tracking and initialize run in Celery task
        mlflow_run_id = initialize_mlflow_tracking(model_id, cleaned_params)
        
        # Update model with MLflow run ID
        if mlflow_run_id:
            try:
                from core.apps.ml_manager.models import MLModel
                model = MLModel.objects.get(id=model_id)
                model.mlflow_run_id = mlflow_run_id
                model.save(update_fields=['mlflow_run_id'])
                logger.info(f"Updated model {model_id} with MLflow run ID: {mlflow_run_id}")
            except Exception as e:
                logger.warning(f"Failed to update model {model_id} with MLflow run ID: {e}")
        
        # Add MLflow run ID to command args for train.py to use
        cmd_args.append(f'--mlflow-run-id={mlflow_run_id}')
        
        # Log the command
        logger.info(f"Executing training command for model {model_id}: {' '.join(cmd_args)}")
        
        # Initial progress update to show command execution
        if progress_callback:
            progress_callback(
                status='training_initializing',
                message=f'Starting training process for model {model_id}',
                model_id=model_id
            )
        
        # Run training with proper process management
        start_time = datetime.now()
        
        with managed_process(cmd_args, model_id, timeout=7200) as process:  # 2 hours timeout
            success, message, training_logs = read_process_output(
                process, 
                model_id, 
                progress_callback=progress_callback,
                timeout=7200
            )
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Get final metrics from model
            try:
                from core.apps.ml_manager.models import MLModel
                model = MLModel.objects.get(id=model_id)
                final_metrics = {
                    'best_val_dice': model.val_dice or 0,
                    'final_epoch': model.current_epoch or 0,
                    'train_loss': model.train_loss,
                    'val_loss': model.val_loss,
                    'train_dice': model.train_dice,
                    'val_dice': model.val_dice,
                }
            except Exception as e:
                logger.warning(f"Could not fetch final metrics for model {model_id}: {e}")
                final_metrics = {
                    'best_val_dice': 0,
                    'final_epoch': 0,
                    'train_loss': None,
                    'val_loss': None,
                    'train_dice': None,
                    'val_dice': None,
                }
            
            if success:
                logger.info(f"Training completed successfully for model {model_id} in {training_time:.2f}s")
                result = {
                    'success': True,
                    'training_time': training_time,
                    'message': message,
                    'mlflow_run_id': mlflow_run_id,
                    'training_logs': training_logs,
                    **final_metrics  # Add all final metrics
                }
            else:
                logger.error(f"Training failed for model {model_id}: {message}")
                result = {
                    'success': False,
                    'error': message,
                    'training_time': training_time,
                    'training_logs': training_logs,
                    'exc_type': 'TrainingError',  # Add explicit exception type
                    **final_metrics  # Add final metrics even on failure
                }
        
    except Exception as e:
        logger.error(f"Training process failed for model {model_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        result = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        # Finalize MLflow run with proper artifacts logging
        finalize_mlflow_run(
            mlflow_run_id,
            success=result.get('success', False),
            final_metrics={'model_id': model_id, **result}
        )
    
    return result

def validate_training_params(training_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean training parameters
    """
    # Define valid choices for specific parameters
    valid_choices = {
        'early_stopping_metric': ['val_dice', 'val_loss', 'val_accuracy'],
        'optimizer': ['adam', 'sgd', 'rmsprop', 'adamw'],
        'model_type': [
            # Standard models
            'unet', 'resunet', 'deeplabv3', 'segnet', 
            # Advanced U-Net variants
            'attention_unet', 'unet_plus_plus', 'transunet',
            # MONAI models
            'monai_unet', 'monai_swin_unetr', 'monai_attentionunet',
            # Other architectures
            'fcn', 'pspnet', 'linknet', 'fpn', 'pan'
        ],
        'loss_function': ['dice', 'bce', 'combined', 'focal', 'tversky', 'focal_tversky'],
        'device': ['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
        'preprocessing_type': ['angiography', 'ct', 'mri', 'xray', 'ultrasound'],
        'model_family': [
            'UNet-Coronary', 'DeepLabV3-Coronary', 'ResUNet-Coronary', 
            'AttentionUNet-Coronary', 'MONAI-UNet-Coronary', 'TransUNet-Coronary'
        ],
    }
    
    cleaned_params = {}
    validation_errors = []
    
    for key, value in training_params.items():
        # Skip None and empty values
        if value is None or value == '':
            continue
            
        # Convert to string and strip whitespace
        if isinstance(value, str):
            value = value.strip()
            if not value:  # Skip empty strings after stripping
                continue
        
        # Validate specific parameters
        if key in valid_choices:
            if value not in valid_choices[key]:
                validation_errors.append(f"Invalid {key}: '{value}'. Valid choices: {valid_choices[key]}")
                continue
        
        # Add numeric validation
        if key in ['batch_size', 'epochs', 'crop_size', 'num_workers', 'early_stopping_patience', 'early_stopping_min_epochs']:
            try:
                value = int(value)
                if value <= 0:
                    validation_errors.append(f"{key} must be positive, got: {value}")
                    continue
            except (ValueError, TypeError):
                validation_errors.append(f"{key} must be a number, got: {value}")
                continue
        
        # Float validation with proper ranges
        if key in ['learning_rate', 'validation_split', 'threshold']:
            try:
                value = float(value)
                if key == 'learning_rate' and (value <= 0 or value > 1):
                    validation_errors.append(f"{key} must be between 0 and 1, got: {value}")
                    continue
                elif key == 'validation_split' and (value <= 0 or value >= 1):
                    validation_errors.append(f"{key} must be between 0 and 1, got: {value}")
                    continue
                elif key == 'threshold' and (value < 0 or value > 1):
                    validation_errors.append(f"{key} must be between 0 and 1, got: {value}")
                    continue
            except (ValueError, TypeError):
                validation_errors.append(f"{key} must be a number, got: {value}")
                continue
        
        # Float validation for preprocessing parameters
        if key in ['clahe_clip_limit', 'preprocessing_clahe_clip_limit']:
            try:
                value = float(value)
                if value <= 0:
                    validation_errors.append(f"{key} must be positive, got: {value}")
                    continue
            except (ValueError, TypeError):
                validation_errors.append(f"{key} must be a number, got: {value}")
                continue
        
        # Integer validation for tile size
        if key in ['clahe_tile_size', 'preprocessing_clahe_tile_size']:
            try:
                value = int(value)
                if value <= 0:
                    validation_errors.append(f"{key} must be positive, got: {value}")
                    continue
            except (ValueError, TypeError):
                validation_errors.append(f"{key} must be a number, got: {value}")
                continue
        
        cleaned_params[key] = value
    
    if validation_errors:
        raise ValueError(f"Training parameter validation failed: {'; '.join(validation_errors)}")
    
    return cleaned_params

def build_training_command(script_path: Path, model_id: int, training_params: Dict[str, Any]) -> list:
    """Build training command arguments"""
    cmd_args = [sys.executable, str(script_path), '--mode', 'train']
    
    # Add all training parameters
    param_mapping = {
        'model_id': model_id,
        'model_type': training_params.get('model_type', 'unet'),
        'data_path': training_params.get('data_path'),
        'batch_size': training_params.get('batch_size', 32),
        'epochs': training_params.get('epochs', 10),
        'learning_rate': training_params.get('learning_rate', 0.001),
        'validation_split': training_params.get('validation_split', 0.2),
        'crop_size': training_params.get('crop_size', 128),
        'optimizer': training_params.get('optimizer', 'adam'),
        'random_flip': training_params.get('use_random_flip', False),
        'random_rotate': training_params.get('use_random_rotate', False),
        'random_scale': training_params.get('use_random_scale', False),
        'random_intensity': training_params.get('use_random_intensity', False),
        'use_early_stopping': training_params.get('use_early_stopping', False),
        'early_stopping_patience': training_params.get('early_stopping_patience', 10),
        'early_stopping_min_epochs': training_params.get('early_stopping_min_epochs', 20),
        'early_stopping_metric': training_params.get('early_stopping_metric', 'val_dice'),
        'use_medical_preprocessing': training_params.get('use_medical_preprocessing', False),
        'medical_preprocessing_type': training_params.get('preprocessing_type', 'angiography'),
        'preprocessing_clahe_clip_limit': training_params.get('clahe_clip_limit', 3.0),
        'preprocessing_clahe_tile_size': training_params.get('clahe_tile_size', 8),
        'preprocessing_use_unsharp_masking': training_params.get('use_unsharp_masking', False),
        'preprocessing_use_frangi': training_params.get('use_frangi_filter', False),
        'preprocessing_use_denoising': training_params.get('use_denoising', False),
        'threshold': training_params.get('threshold', 0.5),
        'num_workers': min(training_params.get('num_workers', 1), 2),  # Limit workers to prevent shared memory issues
        'device': training_params.get('device', 'auto'),
        'model_family': training_params.get('model_family', 'UNet-Coronary'),
        'loss_function': training_params.get('loss_function', 'combined'),
    }
    
    # Convert parameters to command line arguments
    for key, value in param_mapping.items():
        if key == 'model_id':
            cmd_args.extend(['--model-id', str(value)])
        elif value is not None and value != '':  # Skip None and empty strings
            cmd_key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd_args.append(f'--{cmd_key}')
            else:
                # Convert value to string and check if it's not empty
                str_value = str(value).strip()
                if str_value:  # Only add non-empty values
                    cmd_args.append(f'--{cmd_key}={str_value}')
    
    return cmd_args

def initialize_mlflow_tracking(model_id: int, training_params: Dict[str, Any]) -> Optional[str]:
    """Initialize MLflow tracking in Celery task with comprehensive logging"""
    try:
        # Check if we already have an MLflow run ID from the view
        existing_run_id = training_params.get('mlflow_run_id')
        if existing_run_id and existing_run_id != 'None':
            logger.info(f"Using existing MLflow run ID: {existing_run_id}")
            
            # Set up MLflow tracking URI
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
            
            # Get the experiment name from training parameters
            experiment_name = training_params.get('mlflow_experiment', 'coronary-experiments')
            mlflow.set_experiment(experiment_name)
            
            try:
                # Try to resume the existing run
                mlflow.start_run(run_id=existing_run_id)
                logger.info(f"Resumed existing MLflow run: {existing_run_id}")
                return existing_run_id
            except Exception as e:
                logger.warning(f"Failed to resume MLflow run {existing_run_id}: {e}. Creating new run.")
                # Fall through to create a new run
        
        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        
        # Get experiment name from training parameters or use default
        experiment_name = training_params.get('mlflow_experiment', 'coronary-experiments')
        
        # Check if we need to create a new experiment
        create_new_experiment = training_params.get('create_new_experiment', False)
        if create_new_experiment:
            new_experiment_name = training_params.get('new_experiment_name', '').strip()
            new_experiment_description = training_params.get('new_experiment_description', '').strip()
            
            if new_experiment_name:
                try:
                    # Try to create the new experiment
                    experiment_id = mlflow.create_experiment(
                        name=new_experiment_name,
                        tags={'description': new_experiment_description} if new_experiment_description else None
                    )
                    experiment_name = new_experiment_name
                    logger.info(f"Created new MLflow experiment: {experiment_name} with ID: {experiment_id}")
                except Exception as e:
                    logger.warning(f"Error creating new experiment '{new_experiment_name}': {e}, using selected experiment")
                    # Keep the originally selected experiment_name
        
        # Use the selected experiment (don't create new one with model name)
        logger.info(f"Using MLflow experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)  # Use specified experiment
        
        # Start new MLflow run with custom name and artifact path
        run_name = training_params.get('name', f'Training-{model_id}')
        
        # Start MLflow run first to get run_id
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
        
        # Get experiment ID for tracking (but MLflow uses only run_id in path)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else "0"
        
        # MLflow faktycznie używa struktury: /mlflow/{run_id}/artifacts (bez experiment_id)
        mlflow_artifact_path = f"/mlflow/{run_id}/artifacts"
        
        # Set up model directory to use MLflow structure
        try:
            from core.apps.ml_manager.models import MLModel
            model = MLModel.objects.get(id=model_id)
            
            # Set tags for tracking
            mlflow.set_tag('mlflow_artifact_path', mlflow_artifact_path)
            mlflow.set_tag('experiment_id', experiment_id)
            mlflow.set_tag('run_id', run_id)
            
            # Set environment variable for the training script to use MLflow structure
            os.environ['MLFLOW_ARTIFACT_PATH'] = mlflow_artifact_path
            os.environ['MLFLOW_RUN_ID'] = run_id
            os.environ['MLFLOW_EXPERIMENT_ID'] = experiment_id
            
            logger.info(f"[MLFLOW] MLflow artifact path for run {run_id}: {mlflow_artifact_path}")
            
            # Update model directory if needed to reference MLflow structure
            if model.model_directory:
                mlflow.set_tag('original_model_directory', model.model_directory)
                os.environ['MODEL_DIRECTORY'] = model.model_directory
                
        except Exception as e:
            logger.error(f"Error setting up MLflow artifact path: {e}")
        
        # Also set the run name as a tag for better visibility
        mlflow.set_tag('mlflow.runName', run_name)
        mlflow.set_tag('training_name', run_name)
        mlflow.set_tag('model_id', str(model_id))
        mlflow.set_tag('experiment_name', experiment_name)
        mlflow.set_tag('created_by', 'celery-task')
        
        # Update model record with MLflow run ID and fix directory path
        try:
            from core.apps.ml_manager.models import MLModel
            model = MLModel.objects.get(id=model_id)
            model.mlflow_run_id = run_id
            
            # Update model directory path with actual MLflow run ID
            if model.model_directory and 'pending' in model.model_directory:
                # Replace 'pending' with actual MLflow run ID (first 8 chars)
                mlflow_short_id = str(run_id)[:8]
                updated_directory = model.model_directory.replace('pending', mlflow_short_id)
                model.model_directory = updated_directory
                logger.info(f"Updated model directory from {model.model_directory} to {updated_directory}")
            
            # Update unique identifier if it contains 'pending'
            if model.unique_identifier and 'pending' in model.unique_identifier:
                mlflow_short_id = str(run_id)[:8]
                updated_unique_id = model.unique_identifier.replace('pending', mlflow_short_id)
                model.unique_identifier = updated_unique_id
                logger.info(f"Updated unique identifier from {model.unique_identifier} to {updated_unique_id}")
            
            model.save(update_fields=['mlflow_run_id', 'model_directory', 'unique_identifier'])
            logger.info(f"✅ Updated model {model_id} with MLflow run ID: {run_id}")
        except Exception as e:
            logger.error(f"Failed to update model with MLflow run ID: {e}")

        logger.info(f"✅ Successfully created MLflow run {run_id} in experiment {experiment_name}")
        return run_id
        mlflow.set_tag('experiment_name', experiment_name)
        
        logger.info(f"Started MLflow run: {run_id} with name '{run_name}' for model {model_id} in '{experiment_name}' experiment")
        
        # Set comprehensive tags
        mlflow.set_tag('model_id', str(model_id))
        mlflow.set_tag('training_status', 'initializing')
        mlflow.set_tag('task', 'coronary_segmentation')
        mlflow.set_tag('execution_environment', 'celery')
        mlflow.set_tag('training_mode', 'direct_python_call')
        
        # Log all training parameters to MLflow
        for key, value in training_params.items():
            if value is not None and value != '':
                try:
                    mlflow.log_param(key, value)
                    logger.debug(f"Logged MLflow param {key}: {value}")
                except Exception as e:
                    logger.warning(f"Failed to log MLflow param {key}: {e}")
        
        # Initialize system monitoring for resource tracking
        try:
            from core.apps.ml_manager.utils.system_monitor import SystemMonitor
            system_monitor = SystemMonitor(log_interval=30, enable_gpu=True)
            system_monitor.start_monitoring()
            logger.info(f"[MONITORING] System monitoring started for model {model_id} - logging to MLflow every 30 seconds")
            mlflow.set_tag('system_monitoring', 'enabled')
        except Exception as e:
            logger.warning(f"[MONITORING] Failed to start system monitoring for model {model_id}: {e}")
            mlflow.set_tag('system_monitoring', 'disabled')
        
        # Log system information
        import platform
        import torch
        mlflow.log_param('python_version', platform.python_version())
        mlflow.log_param('platform', platform.platform())
        mlflow.log_param('torch_version', torch.__version__)
        if torch.cuda.is_available():
            mlflow.log_param('cuda_version', torch.version.cuda)
            mlflow.log_param('gpu_count', torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                mlflow.log_param(f'gpu_{i}_name', gpu_name)
        
        # Log initial artifacts if model directory already exists
        try:
            from core.apps.ml_manager.models import MLModel
            model = MLModel.objects.get(id=model_id)
            if model.model_directory and os.path.exists(model.model_directory):
                log_initial_training_artifacts(model.model_directory)
        except Exception as e:
            logger.debug(f"No initial artifacts to log: {e}")
        
        return run_id
        
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow tracking for model {model_id}: {e}")
        return None

def log_initial_training_artifacts(model_dir: str):
    """Log initial training artifacts to MLflow at the start of training"""
    try:
        model_path = Path(model_dir)
        
        # Get model directory from environment variable set by initialize_mlflow_tracking
        model_directory = os.environ.get('MODEL_DIRECTORY')
        
        # Define helper function for logging artifacts to MLflow structure
        def log_artifact_dual(artifact_path: str, artifact_type: str = None):
            """Log artifact to MLflow (/mlflow/{experiment_id}/{run_id}/artifacts) and optionally to model directory"""
            try:
                # Log to MLflow - artefakty będą w /mlflow/{experiment_id}/{run_id}/artifacts
                if artifact_type:
                    mlflow.log_artifact(str(artifact_path), artifact_path=artifact_type)
                    logger.debug(f"Logged artifact to MLflow: {artifact_path} -> {artifact_type}")
                else:
                    mlflow.log_artifact(str(artifact_path))
                    logger.debug(f"Logged artifact to MLflow: {artifact_path}")
                
                # Optionally also copy to model directory for local access (if needed)
                if model_directory:
                    model_artifacts_dir = os.path.join(model_directory, 'artifacts')
                    os.makedirs(model_artifacts_dir, exist_ok=True)
                    
                    if artifact_type:
                        dest_dir = os.path.join(model_artifacts_dir, artifact_type)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, os.path.basename(artifact_path))
                    else:
                        dest_path = os.path.join(model_artifacts_dir, os.path.basename(artifact_path))
                    
                    import shutil
                    shutil.copy2(artifact_path, dest_path)
                    logger.debug(f"Copied artifact to model directory: {dest_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to log artifact {artifact_path}: {e}")
        
        # Log training configuration if exists
        config_files = list(model_path.glob('**/training_config*.json'))
        for config_file in config_files:
            log_artifact_dual(str(config_file), "config")
            logger.info(f"Logged initial training config: {config_file.name}")
        
        # Log setup logs if exist
        setup_logs = list(model_path.glob('**/setup*.log'))
        for log_file in setup_logs:
            log_artifact_dual(str(log_file), "logs")
            logger.info(f"Logged setup log: {log_file.name}")
            
        # Create and log training metadata
        metadata = {
            'training_initialized_at': datetime.now().isoformat(),
            'model_directory': str(model_path),
            'artifacts_logged': True,
            'dual_storage_enabled': bool(model_directory)
        }
        
        metadata_file = model_path / 'training_metadata.json'
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
            
        log_artifact_dual(str(metadata_file), "metadata")
        logger.info("Logged initial training metadata with dual storage")
        
    except Exception as e:
        logger.warning(f"Failed to log initial training artifacts: {e}")

def finalize_mlflow_run(run_id: Optional[str], success: bool = True, final_metrics: Dict = None):
    """Finalize MLflow run with final metrics and artifacts in Celery task"""
    if not run_id:
        return
        
    try:
        # The run should still be active from initialize_mlflow_tracking
        if mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            logger.info(f"Finalizing active MLflow run: {run_id}")
            
            # Set final status tag
            mlflow.set_tag('training_status', 'completed' if success else 'failed')
            
            # Log final metrics if provided (only numerical values)
            if final_metrics:
                for key, value in final_metrics.items():
                    if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                        try:
                            mlflow.log_metric(key, value)
                            logger.debug(f"Logged final metric {key}: {value}")
                        except Exception as e:
                            logger.warning(f"Failed to log final metric {key}: {e}")
                    elif value is not None and isinstance(value, str):
                        # Log string values as tags instead of metrics
                        try:
                            mlflow.set_tag(f"final_{key}", str(value)[:250])  # Limit length
                            logger.debug(f"Logged final tag {key}: {str(value)[:50]}...")
                        except Exception as e:
                            logger.warning(f"Failed to log final tag {key}: {e}")
            
            # Log training artifacts from model directory
            try:
                from core.apps.ml_manager.models import MLModel
                model = MLModel.objects.get(id=final_metrics.get('model_id')) if final_metrics else None
                if model and model.model_directory:
                    model_dir = Path(model.model_directory)
                    if model_dir.exists():
                        # Log final model artifacts
                        for artifact_file in model_dir.rglob('*'):
                            if artifact_file.is_file() and artifact_file.suffix in ['.pth', '.json', '.txt', '.log']:
                                try:
                                    relative_path = artifact_file.relative_to(model_dir)
                                    mlflow.log_artifact(str(artifact_file), f"model_artifacts/{relative_path.parent}")
                                    logger.debug(f"Logged artifact: {relative_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to log artifact {artifact_file}: {e}")
            except Exception as e:
                logger.warning(f"Failed to log model artifacts: {e}")
            
            # End the run
            mlflow.end_run()
            logger.info(f"Successfully finalized MLflow run: {run_id}")
        else:
            logger.warning(f"No active MLflow run found to finalize: {run_id}")
            
    except Exception as e:
        logger.warning(f"Failed to finalize MLflow run {run_id}: {e}")
        # Try to end any active run
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except:
            pass

def run_inference_direct(model_id: int, image_path: str, inference_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference directly using functions from train.py
    """
    try:
        # Set up Django environment first
        import os
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.container')
        
        # Import inference functions from train.py
        training_script_path = Path(__file__).parent.parent / 'training'
        if str(training_script_path) not in sys.path:
            sys.path.insert(0, str(training_script_path))
        
        # Import functions directly from train.py
        from training.train import (
            run_inference,
            create_model_from_registry,
            get_default_model_config
        )
        
        # Get model path from inference_params or construct it
        model_path = inference_params.get('model_path', str(Path(settings.CORE_DATA_DIR) / 'models' / f'model_{model_id}'))
        
        # Run inference using the train.py function directly
        start_time = datetime.now()
        result = run_inference(
            model_path=model_path,
            input_path=image_path,
            output_dir=inference_params.get('output_dir', str(Path(settings.CORE_DATA_DIR) / 'inference_results')),
            device=inference_params.get('device', 'cuda'),
            weights_path=inference_params.get('weights_path'),
            model_type=inference_params.get('model_type', 'unet'),
            crop_size=inference_params.get('crop_size', 128),
            threshold=inference_params.get('threshold', 0.5)
        )
        end_time = datetime.now()
        
        inference_time = (end_time - start_time).total_seconds()
        
        return {
            'success': True,
            'inference_time': inference_time,
            'result': result,
            'message': 'Inference completed successfully'
        }
        
    except Exception as e:
        logger.error(f"Direct inference failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@shared_task(bind=True, name='ml_manager.train_model', queue='training')
def train_model_task(self, model_id: int, training_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task to train a ML model using direct Python calls
    """
    logger.info(f"Starting training task for model ID: {model_id}")
    
    try:
        # Import Django models
        from core.apps.ml_manager.models import MLModel
        
        # Get the model instance
        model = MLModel.objects.get(id=model_id)
        model.status = 'loading'
        # Store Celery task ID in training_logs for reference
        model.training_logs = f"Task ID: {self.request.id}\n"
        model.save()
        
        # Set up training stop flag
        training_stop_flags[model_id] = threading.Event()
        
        # Update status to 'training' once we're ready to start
        model.status = 'training'
        model.save()
        
        # Initial progress update to show training has started
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'training_started',
                'model_id': model_id,
                'current_epoch': 0,
                'total_epochs': training_params.get('epochs', 10),
                'progress_percent': 0,  # Start at 0%
                'message': 'Preparing training environment'
            }
        )
        
        # Create training callback for progress updates
        def training_callback(epoch=None, total_epochs=None, train_loss=None, val_loss=None, train_dice=None, val_dice=None, **kwargs):
            """Callback to update training progress"""
            try:
                # Check if training should be stopped
                if training_stop_flags.get(model_id, threading.Event()).is_set():
                    logger.info(f"Training stop requested for model {model_id}")
                    return True  # Signal to stop training
                
                # Update model in database
                update_fields = []
                
                if epoch is not None:
                    model.current_epoch = epoch + 1  # +1 because we're 0-indexed in the code but 1-indexed in UI
                    update_fields.append('current_epoch')
                    
                if total_epochs is not None:
                    model.total_epochs = total_epochs
                    update_fields.append('total_epochs')
                    
                if train_loss is not None:
                    model.train_loss = train_loss
                    update_fields.append('train_loss')
                    
                if val_loss is not None:
                    model.val_loss = val_loss
                    update_fields.append('val_loss')
                    
                if train_dice is not None:
                    model.train_dice = train_dice
                    update_fields.append('train_dice')
                    
                if val_dice is not None:
                    model.val_dice = val_dice
                    update_fields.append('val_dice')
                
                if update_fields:  # Only save if we have fields to update
                    model.save(update_fields=update_fields)
                
                # Update Celery task state with all available metrics from the callback
                progress_data = {
                    'model_id': model_id,
                    'status': kwargs.get('status', 'training_active'),
                    'message': kwargs.get('message', 'Training in progress...'),
                }

                # Add all metrics from the callback arguments and kwargs
                # to the progress data dictionary for Celery state update.
                # This ensures that any metric parsed from the logs is passed to the frontend.
                for key, value in list(locals().items()) + list(kwargs.items()):
                    if key not in ['self', 'kwargs', 'progress_data', 'model', 'update_fields'] and value is not None:
                        progress_data[key] = value

                # Recalculate progress percentage if epoch info is available
                if epoch is not None and total_epochs is not None and total_epochs > 0:
                    progress_data['progress_percent'] = min(100, int(((epoch + 1) / total_epochs) * 100))
                
                self.update_state(state='PROGRESS', meta=progress_data)
                
                # Log progress for debugging with more details
                progress_info = f"Training progress for model {model_id}: epoch {epoch+1 if epoch is not None else 'N/A'}/{total_epochs if total_epochs else 'N/A'}"
                if train_loss is not None:
                    progress_info += f", train_loss={train_loss:.4f}"
                if val_loss is not None:
                    progress_info += f", val_loss={val_loss:.4f}"
                if train_dice is not None:
                    progress_info += f", train_dice={train_dice:.4f}"
                if val_dice is not None:
                    progress_info += f", val_dice={val_dice:.4f}"
                logger.info(progress_info)
                
                return False  # Continue training
                
            except Exception as e:
                logger.error(f"Error in training callback: {e}")
                logger.error(f"Callback traceback: {traceback.format_exc()}")
                # Continue training despite error in callback
                return False
        
        # Direct training function call
        result = run_training_direct(
            model_id=model_id,
            training_params=training_params,
            progress_callback=training_callback
        )
        
        # Clean up stop flag
        if model_id in training_stop_flags:
            del training_stop_flags[model_id]
        
        # Update model status based on result
        if result['success']:
            model.status = 'completed'
            model.best_val_dice = result.get('best_val_dice', model.val_dice)
            # Don't save training logs to database - they're saved in files
            # Save only specific fields to preserve mlflow_run_id and training_data_info
            model.save(update_fields=['status', 'best_val_dice'])
            
            logger.info(f"Training completed successfully for model ID: {model_id}")
            logger.info(f"Final metrics - Best Val Dice: {result.get('best_val_dice', 0):.4f}, Final Epoch: {result.get('final_epoch', 0)}, Training Time: {result.get('training_time', 0):.2f}s")
            return {
                'success': True,
                'model_id': model_id,
                'status': 'completed',
                'best_val_dice': result.get('best_val_dice', 0),
                'final_epoch': result.get('final_epoch', 0),
                'model_path': result.get('model_path', ''),
                'training_time': result.get('training_time', 0),
                'train_loss': result.get('train_loss'),
                'val_loss': result.get('val_loss'),
                'train_dice': result.get('train_dice'),
                'val_dice': result.get('val_dice'),
            }
        else:
            # Check if training was stopped by user
            if result.get('error') == 'Training stopped by user' or 'stopped by user' in result.get('error', '').lower():
                model.status = 'stopped'
                logger.info(f"Training stopped by user for model ID: {model_id}")
            else:
                model.status = 'failed'
                error_message = result.get('error', 'Unknown error')
                logger.error(f"Training failed for model ID: {model_id}: {error_message}")
            
            # Don't save training logs to database - they're saved in files
            # Save only specific fields to preserve mlflow_run_id and training_data_info
            model.save(update_fields=['status'])
            
            # Update state and fail the task properly
            self.update_state(
                state='FAILURE',
                meta={
                    'model_id': model_id,
                    'status': 'failed',
                    'error': error_message
                }
            )
            
            # Raise exception to mark task as FAILURE in Celery
            from celery.exceptions import Ignore
            raise Exception(f"Training failed for model {model_id}: {error_message}")
            
    except Exception as e:
        logger.error(f"Training task failed with exception: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Clean up stop flag
        if model_id in training_stop_flags:
            del training_stop_flags[model_id]
        
        # Update model status
        try:
            model.status = 'failed'
            # Don't save training logs to database - they're saved in files
            model.save(update_fields=['status'])
        except Exception as save_error:
            logger.error(f"Failed to save model status: {save_error}")
            pass
        
        # Update state and re-raise exception to mark task as FAILURE
        self.update_state(
            state='FAILURE',
            meta={
                'model_id': model_id,
                'status': 'failed',
                'error': str(e),
                'exc_type': type(e).__name__
            }
        )
        
        # Re-raise the exception to mark task as FAILURE in Celery
        raise

@shared_task(bind=True, name='ml_manager.run_inference', queue='inference')
def run_inference_task(self, model_id: int, image_path: str, inference_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task to run inference on a model using direct Python calls
    """
    logger.info(f"Starting inference task for model ID: {model_id}")
    
    try:
        # Import Django models
        from core.apps.ml_manager.models import MLModel
        
        # Get the model instance
        model = MLModel.objects.get(id=model_id)
        
        # Direct inference function call
        result = run_inference_direct(
            model_id=model_id,
            image_path=image_path,
            inference_params=inference_params
        )
        
        if result['success']:
            logger.info(f"Inference completed successfully for model ID: {model_id}")
            return {
                'success': True,
                'model_id': model_id,
                'status': 'completed',
                'prediction_path': result.get('prediction_path', ''),
                'confidence_score': result.get('confidence_score', 0.0)
            }
        else:
            error_message = result.get('error', 'Unknown error')
            logger.error(f"Inference failed for model ID: {model_id}: {error_message}")
            
            # Update state and fail the task properly
            self.update_state(
                state='FAILURE',
                meta={
                    'model_id': model_id,
                    'status': 'failed',
                    'error': error_message
                }
            )
            
            # Raise exception to mark task as FAILURE in Celery
            raise Exception(f"Inference failed for model {model_id}: {error_message}")
            
    except Exception as e:
        logger.error(f"Inference task failed with exception: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update state and re-raise exception to mark task as FAILURE
        self.update_state(
            state='FAILURE',
            meta={
                'model_id': model_id,
                'status': 'failed',
                'error': str(e)
            }
        )
        
        # Re-raise the exception to mark task as FAILURE in Celery
        raise

@shared_task(bind=True, name='ml_manager.stop_training', queue='training')
def stop_training_task(self, model_id: int) -> Dict[str, Any]:
    """
    Celery task to stop training process with proper cleanup
    """
    logger.info(f"Stopping training for model ID: {model_id}")
    
    try:
        # Import Django models
        from core.apps.ml_manager.models import MLModel
        
        # Get the model instance
        model = MLModel.objects.get(id=model_id)
        
        # Set the stop flag for this model
        if model_id in training_stop_flags:
            training_stop_flags[model_id].set()
            logger.info(f"Stop flag set for model {model_id}")
        
        # Try to find and kill the training process
        killed_processes = kill_training_processes(model_id)
        
        # Update model status
        model.status = 'stopped'
        model.stop_requested = True
        model.save()
        
        logger.info(f"Training stopped successfully for model ID: {model_id}")
        logger.info(f"Killed {killed_processes} processes")
        
        return {
            'success': True,
            'model_id': model_id,
            'status': 'stopped',
            'killed_processes': killed_processes
        }
            
    except Exception as e:
        logger.error(f"Stop training task failed with exception: {str(e)}")
        return {
            'success': False,
            'model_id': model_id,
            'error': str(e)
        }

def kill_training_processes(model_id: int) -> int:
    """Find and kill training processes for a specific model"""
    killed_count = 0
    
    try:
        # Look for processes that match our training pattern
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and isinstance(cmdline, list):
                    # Check if this is our training process
                    if any('train.py' in arg for arg in cmdline) and \
                       any(f'--model-id={model_id}' in arg or f'--model-id {model_id}' in ' '.join(cmdline) for arg in cmdline):
                        
                        logger.info(f"Found training process for model {model_id}: PID {proc.info['pid']}")
                        
                        # Kill the process group
                        try:
                            os.killpg(os.getpgid(proc.info['pid']), signal.SIGTERM)
                            time.sleep(2)  # Wait for graceful shutdown
                            
                            # Check if still running, force kill if needed
                            if psutil.pid_exists(proc.info['pid']):
                                os.killpg(os.getpgid(proc.info['pid']), signal.SIGKILL)
                                
                            killed_count += 1
                            logger.info(f"Killed training process PID {proc.info['pid']} for model {model_id}")
                            
                        except (OSError, ProcessLookupError):
                            # Process already dead
                            pass
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process disappeared or access denied
                continue
                
    except Exception as e:
        logger.error(f"Error while killing training processes for model {model_id}: {e}")
    
    return killed_count

@shared_task(name='ml_manager.cleanup_failed_trainings', queue='default')
def cleanup_failed_trainings():
    """
    Periodic task to clean up failed or stuck training processes with proper resource cleanup
    """
    try:
        from core.apps.ml_manager.models import MLModel
        from django.utils import timezone
        from datetime import timedelta
        
        # Find models that have been in 'training' status for too long
        cutoff_time = timezone.now() - timedelta(hours=24)  # 24 hours
        
        stuck_models = MLModel.objects.filter(
            status='training',
            updated_at__lt=cutoff_time
        )
        
        cleaned_count = 0
        killed_processes = 0
        
        for model in stuck_models:
            logger.info(f"Cleaning up stuck training for model {model.id}")
            
            # Kill any remaining processes
            killed = kill_training_processes(model.id)
            killed_processes += killed
            
            # Update model status
            model.status = 'failed'
            # Don't save training logs to database - they're saved in files
            model.save(update_fields=['status'])
            
            # Clean up stop flag if exists
            if model.id in training_stop_flags:
                del training_stop_flags[model.id]
            
            cleaned_count += 1
            logger.info(f"Cleaned up stuck training for model {model.id}, killed {killed} processes")
        
        # Also clean up orphaned processes (processes without corresponding model)
        orphaned_killed = cleanup_orphaned_training_processes()
        
        logger.info(f"Cleanup completed: {cleaned_count} stuck trainings cleaned, {killed_processes + orphaned_killed} total processes killed")
        
        return {
            'cleaned_count': cleaned_count,
            'killed_processes': killed_processes + orphaned_killed
        }
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {str(e)}")
        return {'error': str(e)}

def cleanup_orphaned_training_processes() -> int:
    """Clean up training processes that don't have corresponding active models"""
    killed_count = 0
    
    try:
        from core.apps.ml_manager.models import MLModel
        
        # Get all active training model IDs
        active_training_ids = set(
            MLModel.objects.filter(status='training').values_list('id', flat=True)
        )
        
        # Find training processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and isinstance(cmdline, list):
                    # Check if this is a training process
                    if any('train.py' in arg for arg in cmdline):
                        # Extract model ID from command line
                        model_id = None
                        for arg in cmdline:
                            if '--model-id=' in arg:
                                model_id = int(arg.split('=')[1])
                                break
                        
                        if model_id and model_id not in active_training_ids:
                            # This is an orphaned process
                            process_age = time.time() - proc.info['create_time']
                            
                            # Only kill processes older than 1 hour
                            if process_age > 3600:
                                logger.info(f"Killing orphaned training process PID {proc.info['pid']} for model {model_id}")
                                
                                try:
                                    os.killpg(os.getpgid(proc.info['pid']), signal.SIGTERM)
                                    time.sleep(2)
                                    
                                    if psutil.pid_exists(proc.info['pid']):
                                        os.killpg(os.getpgid(proc.info['pid']), signal.SIGKILL)
                                    
                                    killed_count += 1
                                    
                                except (OSError, ProcessLookupError):
                                    pass
                                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                continue
                
    except Exception as e:
        logger.error(f"Error during orphaned process cleanup: {e}")
    
    return killed_count

@shared_task(name='ml_manager.system_health_check', queue='default')
def system_health_check():
    """
    Periodic task to check system health and training process status
    """
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check training processes
        training_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and isinstance(cmdline, list) and any('train.py' in arg for arg in cmdline):
                    # Extract model ID
                    model_id = None
                    for arg in cmdline:
                        if '--model-id=' in arg:
                            model_id = int(arg.split('=')[1])
                            break
                    
                    training_processes.append({
                        'pid': proc.info['pid'],
                        'model_id': model_id,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                continue
        
        # Log system status
        logger.info(f"System Health - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
        logger.info(f"Active training processes: {len(training_processes)}")
        
        # Check for resource issues
        alerts = []
        if cpu_percent > 90:
            alerts.append(f"High CPU usage: {cpu_percent}%")
        if memory.percent > 85:
            alerts.append(f"High memory usage: {memory.percent}%")
        if disk.percent > 90:
            alerts.append(f"High disk usage: {disk.percent}%")
        
        # Check for problematic training processes
        for proc in training_processes:
            if proc['memory_mb'] > 7000:  # 7GB
                alerts.append(f"Model {proc['model_id']} using {proc['memory_mb']:.1f}MB memory")
            if proc['cpu_percent'] > 95:
                alerts.append(f"Model {proc['model_id']} using {proc['cpu_percent']}% CPU")
        
        if alerts:
            logger.warning(f"System alerts: {'; '.join(alerts)}")
        
        return {
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            },
            'training_processes': training_processes,
            'alerts': alerts
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {'error': str(e)}


@shared_task(name='ml_manager.sync_mlflow_data', queue='default')
def sync_mlflow_data_task():
    """Periodic task to synchronize MLflow data with database"""
    try:
        from django.core.management import call_command
        from io import StringIO
        
        logger.info("Starting periodic MLflow data sync")
        
        # Capture command output
        output = StringIO()
        call_command('sync_mlflow_data', stdout=output, stderr=output)
        
        output_text = output.getvalue()
        logger.info(f"MLflow sync completed: {output_text}")
        
        return {
            'success': True,
            'message': 'MLflow data sync completed successfully',
            'output': output_text
        }
        
    except Exception as e:
        logger.error(f"Failed to sync MLflow data: {e}")
        return {
            'success': False,
            'error': str(e)
        }
