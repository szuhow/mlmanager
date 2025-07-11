import mlflow
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import django
import logging

# Add the project root to Python path
base_dir = Path(__file__).resolve().parent.parent.parent.parent
core_path = str(base_dir / 'core')
ml_path = str(base_dir / 'ml')

if core_path not in sys.path:
    sys.path.append(core_path)
if ml_path not in sys.path:
    sys.path.append(ml_path)

# Set up Django (if not already done)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')

# Initialize Django to get settings
django.setup()

# Import Django settings after setup
from django.conf import settings

# Use Django settings for core data directory
core_data_dir = settings.CORE_DATA_DIR

logging.info(f"Base directory: {base_dir}")
logging.info(f"Core data directory: {core_data_dir}")
# Ensure log directory exists
log_dir = core_data_dir / 'artifacts'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'training.log'

# --- Import MLModel with better error handling ---
try:
    # Ensure Django is properly setup before importing models
    try:
        django.setup()
    except RuntimeError as e:
        # Django may already be configured
        if "populated" not in str(e):
            raise
    
    # Import model after ensuring Django is ready
    from core.apps.ml_manager.models import MLModel
    
except Exception as e:
    # Setup basic logging to file for debugging
    logging.basicConfig(
        filename=str(log_file), 
        level=logging.DEBUG, 
        format='%(asctime)s %(levelname)s %(message)s', 
        force=True
    )
    logging.error(f"Failed to import MLModel in training_callback.py: {e}")
    logging.error(f"Django setup failed. DJANGO_SETTINGS_MODULE: {os.environ.get('DJANGO_SETTINGS_MODULE')}")
    logging.error(f"Python path: {sys.path}")
    logging.error(f"Base dir: {base_dir}")
    logging.error(f"Current working directory: {os.getcwd()}")
    raise

class TrainingCallback:
    def __init__(self, model_id, run_id):
        self.model_id = model_id
        self.run_id = run_id
        self.best_val_dice = float('-inf')
        self.best_val_iou = float('-inf')
        
        # Add debug logging for callback initialization
        logging.info(f"[CALLBACK INIT] Initializing callback for model_id: {model_id}, run_id: {run_id}")
        
        try:
            self.model = MLModel.objects.get(id=model_id)
            logging.info(f"[CALLBACK INIT] Successfully loaded model: {self.model.name}")
        except Exception as e:
            logging.error(f"[CALLBACK INIT] Failed to load model {model_id}: {e}")
            raise
        
        # Initialize best values in model if they are None
        if self.model.best_val_dice is None:
            self.model.best_val_dice = 0.0
        if self.model.best_val_iou is None:
            self.model.best_val_iou = 0.0
            
        logging.info(f"[CALLBACK] Initialized for model {model_id}, run {run_id}")
        logging.info(f"[CALLBACK] Initial best_val_dice: {self.model.best_val_dice}, best_val_iou: {self.model.best_val_iou}")
        self.model.save()
    
    def on_training_start(self):
        """Called when training starts - updates status to loading"""
        logging.info(f"[CALLBACK] Training start: Setting model {self.model_id} status to 'loading'")
        self.model.status = 'loading'
        self.model.save()
        return True
    
    def on_dataset_loaded(self):
        """Called when dataset loading is complete - updates status to training"""
        logging.info(f"[CALLBACK] Dataset loaded: Setting model {self.model_id} status to 'training'")
        self.model.status = 'training'
        self.model.save()
        return True
    
    def set_model_directory(self, model_directory):
        """Store the model directory path in the Django model"""
        logging.info(f"[CALLBACK] Setting model directory: {model_directory}")
        logging.info(f"[CALLBACK] Before update - model_directory: {self.model.model_directory}")
        logging.info(f"[CALLBACK] Before update - unique_identifier: {self.model.unique_identifier}")
        
        self.model.model_directory = model_directory
        # Extract the actual folder name from the path to use as unique_identifier
        if model_directory:
            actual_folder_name = os.path.basename(model_directory.rstrip('/'))
            self.model.unique_identifier = actual_folder_name
            logging.info(f"[CALLBACK] Set model directory: {model_directory}")
            logging.info(f"[CALLBACK] Updated unique_identifier to match folder name: {actual_folder_name}")
        
        self.model.save(update_fields=['model_directory', 'unique_identifier'])
        
        logging.info(f"[CALLBACK] After save - model_directory: {self.model.model_directory}")
        logging.info(f"[CALLBACK] After save - unique_identifier: {self.model.unique_identifier}")
        return True
    
    def on_epoch_start(self, epoch, total_epochs):
        """Called at the start of each epoch"""
        logging.info(f"[CALLBACK] on_epoch_start called: epoch={epoch}, total_epochs={total_epochs}, model_id={self.model_id}")
        
        try:
            self.model.refresh_from_db()  # Refresh to get latest stop_requested value
            self.model.current_epoch = epoch + 1  # Convert 0-based to 1-based for UI
            self.model.total_epochs = total_epochs
            logging.info(f"[CALLBACK] Epoch {epoch + 1}/{total_epochs} started - updating model {self.model_id}")
            self.model.save()
            
            # Check if training should be stopped
            if self.model.stop_requested:
                logging.info(f"[CALLBACK] Stop requested for model {self.model_id}")
                return False
                
            logging.info(f"[CALLBACK] on_epoch_start completed successfully for model {self.model_id}")
            return True
        except Exception as e:
            logging.error(f"[CALLBACK] Error in on_epoch_start for model {self.model_id}: {e}")
            return True  # Continue training even if callback fails
    
    def on_epoch_end(self, epoch, logs):
        """Called at the end of each epoch with the metrics"""
        logging.info(f"[CALLBACK] on_epoch_end called: epoch={epoch}, model_id={self.model_id}, logs={logs}")
        
        try:
            self.model.train_loss = logs.get('train_loss', 0.0)
            self.model.val_loss = logs.get('val_loss', 0.0)
            self.model.train_dice = logs.get('train_dice', 0.0)
            self.model.val_dice = logs.get('val_dice', 0.0)
            self.model.train_iou = logs.get('train_iou', 0.0)
            self.model.val_iou = logs.get('val_iou', 0.0)
            
            # Update best validation dice and iou if current is better
            current_val_dice = logs.get('val_dice', 0.0)
            current_val_iou = logs.get('val_iou', 0.0)
            
            # Initialize best values if this is the first epoch or if current is better
            if self.model.best_val_dice is None or current_val_dice > self.model.best_val_dice:
                self.model.best_val_dice = current_val_dice
                logging.info(f"[CALLBACK] New best val_dice: {current_val_dice}")
            
            if self.model.best_val_iou is None or current_val_iou > self.model.best_val_iou:
                self.model.best_val_iou = current_val_iou
                logging.info(f"[CALLBACK] New best val_iou: {current_val_iou}")
            
            # Log detailed callback info
            logging.info(f"[CALLBACK] Epoch {epoch} metrics - Val Dice: {current_val_dice:.4f} (best: {self.model.best_val_dice:.4f}), Val IoU: {current_val_iou:.4f} (best: {self.model.best_val_iou:.4f})")
            
            self.model.save()
            logging.info(f"[CALLBACK] Model saved successfully for epoch {epoch}")
            
            # Log metrics to MLflow with enhanced synchronization
            try:
                import mlflow
                
                # Basic metrics
                mlflow_metrics = {
                    'train_loss': logs.get('train_loss', 0),
                    'val_loss': logs.get('val_loss', 0),
                    'train_dice': logs.get('train_dice', 0),
                    'val_dice': logs.get('val_dice', 0),
                    'train_iou': logs.get('train_iou', 0),
                    'val_iou': logs.get('val_iou', 0),
                    'best_val_dice': self.model.best_val_dice,
                    'best_val_iou': self.model.best_val_iou
                }
                
                # Enhanced metrics
                mlflow_metrics.update({
                    'current_epoch': epoch + 1,
                    'epoch_progress': epoch + 1,
                })
                
                # Log all metrics
                mlflow.log_metrics(mlflow_metrics, step=epoch + 1)
                
                # Update MLflow tags with current training state
                mlflow.set_tag('training_status', 'in_progress')
                mlflow.set_tag('last_epoch_completed', epoch + 1)
                mlflow.set_tag('current_best_val_dice', f"{self.model.best_val_dice:.4f}")
                mlflow.set_tag('current_val_dice', f"{current_val_dice:.4f}")
                
                logging.info(f"[CALLBACK] MLflow metrics logged for epoch {epoch+1}")
                
            except Exception as mlflow_error:
                logging.warning(f"[CALLBACK] Failed to log metrics to MLflow: {mlflow_error}")
            
            # Trigger sample generation for this epoch if model directory is available
            if hasattr(self.model, 'model_directory') and self.model.model_directory:
                try:
                    # Signal that sample generation should happen for this epoch
                    # This will be picked up by the training loop
                    logging.info(f"[CALLBACK] Triggering sample generation for epoch {epoch+1}")
                    
                    # Store flag in training_data_info to trigger sample generation
                    if not self.model.training_data_info:
                        self.model.training_data_info = {}
                    self.model.training_data_info['generate_samples_epoch'] = epoch + 1
                    self.model.training_data_info['last_sample_trigger'] = f"epoch_{epoch+1}"
                    
                    # Save model to trigger sample generation flag
                    self.model.save(update_fields=['training_data_info'])
                    logging.info(f"[CALLBACK] Sample generation flag set for epoch {epoch+1}")
                    
                except Exception as e:
                    logging.warning(f"[CALLBACK] Failed to trigger sample generation: {e}")
            else:
                logging.warning(f"[CALLBACK] No model directory available for sample generation")
            
            logging.info(f"[CALLBACK] on_epoch_end completed successfully for model {self.model_id}")
            
        except Exception as e:
            logging.error(f"[CALLBACK] Error in on_epoch_end for model {self.model_id}: {e}")
            logging.error(f"[CALLBACK] Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logging.error(f"[CALLBACK] Traceback: {traceback.format_exc()}")
    
    def sync_metrics_from_mlflow(self):
        """Synchronize metrics from MLflow to Django model"""
        try:
            import mlflow
            logging.info(f"[CALLBACK] Syncing metrics from MLflow for model {self.model_id}")
            
            # Get MLflow run data
            run = mlflow.get_run(self.run_id)
            metrics = run.data.metrics
            
            logging.info(f"[CALLBACK] MLflow metrics: {metrics}")
            
            # Update Django model with MLflow metrics
            if 'best_val_dice' in metrics:
                self.model.best_val_dice = metrics['best_val_dice']
                logging.info(f"[CALLBACK] Updated best_val_dice from MLflow: {metrics['best_val_dice']}")
            
            if 'best_val_iou' in metrics:
                self.model.best_val_iou = metrics['best_val_iou']
                logging.info(f"[CALLBACK] Updated best_val_iou from MLflow: {metrics['best_val_iou']}")
            
            # Update current epoch metrics
            if 'val_dice' in metrics:
                self.model.val_dice = metrics['val_dice']
            if 'val_iou' in metrics:
                self.model.val_iou = metrics['val_iou']
            if 'train_dice' in metrics:
                self.model.train_dice = metrics['train_dice']
            if 'train_iou' in metrics:
                self.model.train_iou = metrics['train_iou']
            if 'train_loss' in metrics:
                self.model.train_loss = metrics['train_loss']
            if 'val_loss' in metrics:
                self.model.val_loss = metrics['val_loss']
            
            self.model.save()
            logging.info(f"[CALLBACK] Successfully synced metrics from MLflow to Django model")
            
        except Exception as e:
            logging.error(f"[CALLBACK] Failed to sync metrics from MLflow: {e}")
            import traceback
            logging.error(f"[CALLBACK] Traceback: {traceback.format_exc()}")

    def on_train_end(self, logs=None):
        """Called at the end of training - Enhanced MLflow synchronization."""
        logs = logs or {}
        
        try:
            import mlflow
            from django.utils import timezone
            
            # Update training completion status
            mlflow.set_tag('training_status', 'completed')
            mlflow.set_tag('training_completed_at', timezone.now().isoformat())
            
            # Log final metrics summary
            final_metrics = {
                'final_best_val_dice': self.model.best_val_dice or 0.0,
                'final_best_val_iou': self.model.best_val_iou or 0.0,
                'total_epochs_completed': self.model.current_epoch or 0,
                'training_duration_seconds': (timezone.now() - self.model.created_at).total_seconds()
            }
            
            # Add current metrics if available
            if self.model.val_dice is not None:
                final_metrics['final_val_dice'] = self.model.val_dice
            if self.model.val_loss is not None:
                final_metrics['final_val_loss'] = self.model.val_loss
            if self.model.train_dice is not None:
                final_metrics['final_train_dice'] = self.model.train_dice
            if self.model.train_loss is not None:
                final_metrics['final_train_loss'] = self.model.train_loss
            
            mlflow.log_metrics(final_metrics)
            
            # Set final status and completion tags
            mlflow.set_tag('status', 'FINISHED')
            mlflow.set_tag('final_status', 'completed')
            mlflow.set_tag('completion_timestamp', timezone.now().isoformat())
            
            # Log training summary as params
            mlflow.log_params({
                'completed_epochs': self.model.current_epoch or 0,
                'total_planned_epochs': self.model.total_epochs or 0,
                'final_best_dice': f"{self.model.best_val_dice:.4f}" if self.model.best_val_dice else "0.0000"
            })
            
            logging.info(f"[CALLBACK] MLflow training completed - Final best val_dice: {self.model.best_val_dice:.4f}")
            
        except Exception as mlflow_error:
            logging.warning(f"[CALLBACK] Failed to finalize MLflow logging: {mlflow_error}")
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training - Initialize MLflow tracking."""
        try:
            import mlflow
            from django.utils import timezone
            
            # Set initial training status
            mlflow.set_tag('training_status', 'started')
            mlflow.set_tag('training_started_at', self.model.created_at.isoformat())
            mlflow.set_tag('model_id', str(self.model.id))
            mlflow.set_tag('model_name', self.model.name)
            mlflow.set_tag('status', 'RUNNING')
            
            # Log initial training parameters
            mlflow.log_params({
                'model_type': self.model.model_type or 'unknown',
                'model_family': self.model.model_family or 'unknown',
                'planned_epochs': self.model.total_epochs or 0
            })
            
            logging.info(f"[CALLBACK] MLflow training initialized for model {self.model.name}")
            
        except Exception as mlflow_error:
            logging.warning(f"[CALLBACK] Failed to initialize MLflow logging: {mlflow_error}")

    def on_training_end(self, logs=None):
        """Called when training is complete"""
        logging.info(f"[CALLBACK] Training ended for model {self.model_id}. Current epoch: {self.model.current_epoch}, Stop requested: {self.model.stop_requested}")
        
        # First call the enhanced MLflow synchronization
        self.on_train_end(logs)
        
        # Check if training was stopped by user request
        if self.model.stop_requested:
            self.model.status = 'stopped'
            logging.info(f"[CALLBACK] Model {self.model_id} marked as stopped")
            
            # Update MLflow for stopped training
            try:
                import mlflow
                mlflow.set_tag('training_status', 'stopped')
                mlflow.set_tag('final_status', 'stopped')
                mlflow.set_tag('status', 'KILLED')
            except Exception as e:
                logging.warning(f"[CALLBACK] Failed to update MLflow for stopped training: {e}")
                
        else:
            # Set to completed if training progressed through at least one epoch or if total_epochs was 1
            if self.model.current_epoch > 0 or self.model.total_epochs == 1:
                self.model.status = 'completed'
                # Ensure current_epoch equals total_epochs for completed training
                if self.model.current_epoch < self.model.total_epochs:
                    self.model.current_epoch = self.model.total_epochs
                logging.info(f"[CALLBACK] Model {self.model_id} marked as completed")
                
                # Update MLflow for completed training
                try:
                    import mlflow
                    mlflow.set_tag('training_status', 'completed')
                    mlflow.set_tag('final_status', 'completed')
                    mlflow.set_tag('status', 'FINISHED')
                except Exception as e:
                    logging.warning(f"[CALLBACK] Failed to update MLflow for completed training: {e}")
                    
            else:
                # If no epochs were completed, mark as failed
                self.model.status = 'failed'
                if not self.model.performance_metrics:
                    self.model.performance_metrics = {}
                self.model.performance_metrics['error'] = 'Training terminated before completing any epochs'
                logging.info(f"[CALLBACK] Model {self.model_id} marked as failed - no epochs completed")
                
                # Update MLflow for failed training
                try:
                    import mlflow
                    mlflow.set_tag('training_status', 'failed')
                    mlflow.set_tag('final_status', 'failed')
                    mlflow.set_tag('status', 'FAILED')
                    mlflow.set_tag('failure_reason', 'Training terminated before completing any epochs')
                except Exception as e:
                    logging.warning(f"[CALLBACK] Failed to update MLflow for failed training: {e}")
        
        if logs:
            if not self.model.performance_metrics:
                self.model.performance_metrics = {}
            self.model.performance_metrics.update(logs)
            logging.info(f"[CALLBACK] Updated performance metrics: {logs}")
        
        # Save final metrics summary
        final_summary = {
            'final_train_loss': self.model.train_loss,
            'final_val_loss': self.model.val_loss,
            'final_train_dice': self.model.train_dice,
            'final_val_dice': self.model.val_dice,
            'final_best_val_dice': self.model.best_val_dice,
            'final_train_iou': getattr(self.model, 'train_iou', 0.0),
            'final_val_iou': getattr(self.model, 'val_iou', 0.0),
            'final_best_val_iou': getattr(self.model, 'best_val_iou', 0.0),
        }
        logging.info(f"[CALLBACK] Final metrics summary for model {self.model_id}: {final_summary}")
        
        # Sync final metrics from MLflow to ensure they are saved in Django
        self.sync_metrics_from_mlflow()
        
        # Save training logs to MLflow as artifacts
        self._save_logs_to_mlflow()
        
        self.model.save()
        logging.info(f"[CALLBACK] Model {self.model_id} saved with status: {self.model.status}")
    
    def on_training_stopped(self, logs=None):
        """Called when training is stopped by user request"""
        self.model.status = 'stopped'
        if logs:
            self.model.performance_metrics.update(logs)
        
        # Save training logs to MLflow as artifacts
        self._save_logs_to_mlflow()
        
        self.model.save()
    
    def on_training_failed(self, exception):
        """Called if training fails"""
        self.model.status = 'failed'
        if not self.model.performance_metrics:
            self.model.performance_metrics = {}
        self.model.performance_metrics['error'] = str(exception)
        
        # Save training logs to MLflow as artifacts even on failure
        self._save_logs_to_mlflow()
        
        self.model.save()
    
    def _save_logs_to_mlflow(self):
        """Save training logs and other artifacts to MLflow"""
        try:
            # Save training log file if it exists
            log_paths = [
                log_file,  # Main training log from callback setup
                core_data_dir / 'logs' / 'training.log',  # Global training log
                core_data_dir / 'logs' / f'model_{self.model_id}' / f'training_{self.model_id}*.log'  # Model specific logs
            ]
            
            for log_path in log_paths:
                if '*' in str(log_path):
                    # Handle glob patterns
                    import glob
                    matching_files = glob.glob(str(log_path))
                    for file_path in matching_files:
                        if Path(file_path).exists():
                            try:
                                mlflow.log_artifact(file_path, "logs")
                                logging.info(f"[CALLBACK] Saved log file to MLflow: {file_path}")
                            except Exception as e:
                                logging.warning(f"[CALLBACK] Failed to save log file {file_path} to MLflow: {e}")
                else:
                    if Path(log_path).exists():
                        try:
                            mlflow.log_artifact(str(log_path), "logs")
                            logging.info(f"[CALLBACK] Saved log file to MLflow: {log_path}")
                        except Exception as e:
                            logging.warning(f"[CALLBACK] Failed to save log file {log_path} to MLflow: {e}")
            
            # Save model checkpoints directory if it exists
            checkpoint_dir = core_data_dir / 'models' / f'model_{self.model_id}'
            if checkpoint_dir.exists():
                try:
                    mlflow.log_artifacts(str(checkpoint_dir), "checkpoints")
                    logging.info(f"[CALLBACK] Saved model checkpoints to MLflow: {checkpoint_dir}")
                except Exception as e:
                    logging.warning(f"[CALLBACK] Failed to save checkpoints to MLflow: {e}")
            
            # Save performance metrics as a JSON file
            if hasattr(self.model, 'performance_metrics') and self.model.performance_metrics:
                metrics_file = core_data_dir / 'temp' / f'metrics_model_{self.model_id}.json'
                metrics_file.parent.mkdir(parents=True, exist_ok=True)
                
                import json
                with open(metrics_file, 'w') as f:
                    json.dump(self.model.performance_metrics, f, indent=2)
                
                try:
                    mlflow.log_artifact(str(metrics_file), "metrics")
                    logging.info(f"[CALLBACK] Saved performance metrics to MLflow: {metrics_file}")
                    # Clean up temporary file
                    metrics_file.unlink()
                except Exception as e:
                    logging.warning(f"[CALLBACK] Failed to save metrics file to MLflow: {e}")
                    
        except Exception as e:
            logging.error(f"[CALLBACK] Error saving artifacts to MLflow: {e}")
    
    def update_registry_info(self, registry_model_name, registry_model_version, is_registered=True):
        """Update the model with MLflow Registry information"""
        self.model.registry_model_name = registry_model_name
        self.model.registry_model_version = registry_model_version
        self.model.is_registered = is_registered
        self.model.save()
    
    def update_registry_stage(self, stage):
        """Update the model's registry stage"""
        self.model.registry_stage = stage
        self.model.save()
    
    def update_model_metadata(self, model_family=None, model_type=None, architecture_info=None, training_data_info=None):
        """Update model metadata including family, type, architecture, and training data info"""
        if model_family:
            self.model.model_family = model_family
        if model_type:
            self.model.model_type = model_type
        if architecture_info:
            # Ensure model_architecture_info is initialized as a dict
            if not self.model.model_architecture_info:
                self.model.model_architecture_info = {}
            
            # Convert SimpleNamespace or other objects to dict if needed
            if hasattr(architecture_info, '__dict__'):
                arch_dict = architecture_info.__dict__.copy()
            elif isinstance(architecture_info, dict):
                arch_dict = architecture_info.copy()
            else:
                # Try to convert to dict using vars()
                try:
                    arch_dict = vars(architecture_info).copy()
                except TypeError:
                    # If all else fails, create a basic representation
                    arch_dict = {
                        'display_name': getattr(architecture_info, 'display_name', 'Unknown'),
                        'framework': getattr(architecture_info, 'framework', 'Unknown'),
                        'description': getattr(architecture_info, 'description', 'No description'),
                        'category': getattr(architecture_info, 'category', 'general'),
                        'author': getattr(architecture_info, 'author', 'Unknown'),
                        'version': getattr(architecture_info, 'version', '1.0.0')
                    }
            
            # Filter out non-JSON serializable objects (like model_class)
            json_serializable_dict = {}
            for key, value in arch_dict.items():
                try:
                    # Test if the value is JSON serializable
                    import json
                    json.dumps(value)
                    json_serializable_dict[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values (like model_class)
                    print(f"Skipping non-serializable field: {key} = {type(value)}")
                    # For model_class, save just the class name as string
                    if key == 'model_class' and hasattr(value, '__name__'):
                        json_serializable_dict['model_class_name'] = value.__name__
                    elif key == 'model_class' and hasattr(value, '__class__'):
                        json_serializable_dict['model_class_name'] = value.__class__.__name__
            
            self.model.model_architecture_info.update(json_serializable_dict)
            
        if training_data_info:
            # Ensure training_data_info is initialized as a dict
            if not self.model.training_data_info:
                self.model.training_data_info = {}
            self.model.training_data_info.update(training_data_info)
        self.model.save()
    
    def update_training_config(self, config):
        """Update model with training configuration parameters"""
        if not self.model.training_data_info:
            self.model.training_data_info = {}
        
        # Store training configuration
        self.model.training_data_info.update({
            'batch_size': config.get('batch_size'),
            'epochs': config.get('epochs'),
            'learning_rate': config.get('learning_rate'),
            'crop_size': config.get('crop_size'),
            'validation_split': config.get('validation_split'),
            'num_workers': config.get('num_workers'),
            'augmentation': {
                'random_flip': config.get('random_flip', False),
                'random_rotate': config.get('random_rotate', False),
                'random_scale': config.get('random_scale', False),
                'random_intensity': config.get('random_intensity', False)
            }
        })
        self.model.save()
    
    def update_architecture_info(self, model, model_config):
        """Update model architecture information based on the trained model"""
        if not self.model.model_architecture_info:
            self.model.model_architecture_info = {}
        
        # Get model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.model.model_architecture_info.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_config': model_config,
            'architecture_type': self.model.model_type,
            'input_channels': getattr(model, 'in_channels', 1),
            'output_channels': getattr(model, 'out_channels', 1),
            'spatial_dims': getattr(model, 'spatial_dims', 2)
        })
        self.model.save()
    
    def on_batch_start(self, batch_idx, total_batches):
        """Called at the start of each batch"""
        try:
            # Always refresh from database to get latest stop_requested state
            self.model.refresh_from_db()
            self.model.current_batch = batch_idx + 1  # Convert 0-based to 1-based for UI
            self.model.total_batches_per_epoch = total_batches
            self.model.save(update_fields=['current_batch', 'total_batches_per_epoch'])
            
            # Check if training should be stopped
            if self.model.stop_requested:
                logging.info(f"[CALLBACK] Stop requested for model {self.model.id}, terminating training")
                return False
            return True
        except Exception as e:
            # Model might have been deleted
            logging.warning(f"[CALLBACK] Error accessing model {self.model_id}, might be deleted: {e}")
            logging.info(f"[CALLBACK] Terminating training due to model access error")
            return False
    
    def on_batch_end(self, batch_idx, batch_logs=None):
        """Called at the end of each batch with optional metrics"""
        try:
            self.model.current_batch = batch_idx + 1  # Already 1-based since batch completed
            if batch_logs:
                # Update running training metrics if provided
                self.model.train_loss = batch_logs.get('train_loss', self.model.train_loss)
                self.model.train_dice = batch_logs.get('train_dice', self.model.train_dice)
                self.model.train_iou = batch_logs.get('train_iou', self.model.train_iou)
            # Always save to ensure progress bar updates in real-time
            self.model.save(update_fields=['current_batch', 'train_loss', 'train_dice', 'train_iou'])
        except Exception as e:
            logging.warning(f"[CALLBACK] Error updating batch progress: {e}")
    
    def set_epoch_batches(self, total_batches):
        """Set total number of batches per epoch"""
        self.model.total_batches_per_epoch = total_batches
        self.model.save(update_fields=['total_batches_per_epoch'])
        self.model.save()
