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

# Ensure log directory exists
log_dir = base_dir / 'data' / 'artifacts'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'training.log'

# Set up Django (if not already done)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')

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
        self.model = MLModel.objects.get(id=model_id)
    
    def on_training_start(self):
        """Called when training starts - updates status to loading"""
        self.model.status = 'loading'
        self.model.save()
        return True
    
    def on_dataset_loaded(self):
        """Called when dataset loading is complete - updates status to training"""
        self.model.status = 'training'
        self.model.save()
        return True
    
    def set_model_directory(self, model_directory):
        """Store the model directory path in the Django model"""
        self.model.model_directory = model_directory
        self.model.save()
        return True
    
    def on_epoch_start(self, epoch, total_epochs):
        """Called at the start of each epoch"""
        self.model.refresh_from_db()  # Refresh to get latest stop_requested value
        self.model.current_epoch = epoch + 1  # Convert 0-based to 1-based for UI
        self.model.total_epochs = total_epochs
        self.model.save()
        
        # Check if training should be stopped
        if self.model.stop_requested:
            return False
        return True
    
    def on_epoch_end(self, epoch, logs):
        """Called at the end of each epoch with the metrics"""
        self.model.train_loss = logs.get('train_loss', 0.0)
        self.model.val_loss = logs.get('val_loss', 0.0)
        self.model.train_dice = logs.get('train_dice', 0.0)
        self.model.val_dice = logs.get('val_dice', 0.0)
        
        # Update best validation dice if current is better
        if logs.get('val_dice', 0.0) > self.model.best_val_dice:
            self.model.best_val_dice = logs.get('val_dice', 0.0)
        
        self.model.save()
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'train_loss': logs.get('train_loss', 0),
            'val_loss': logs.get('val_loss', 0),
            'train_dice': logs.get('train_dice', 0),
            'val_dice': logs.get('val_dice', 0),
            'best_val_dice': self.model.best_val_dice
        }, step=epoch)
    
    def on_training_end(self, logs=None):
        """Called when training is complete"""
        self.model.status = 'completed'
        if logs:
            self.model.performance_metrics.update(logs)
            self.model.save()
    
    def on_training_failed(self, exception):
        """Called if training fails"""
        self.model.status = 'failed'
        if not self.model.performance_metrics:
            self.model.performance_metrics = {}
        self.model.performance_metrics['error'] = str(exception)
        self.model.save()
    
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
        self.model.refresh_from_db()
        self.model.current_batch = batch_idx + 1  # Convert 0-based to 1-based for UI
        self.model.total_batches_per_epoch = total_batches
        self.model.save()
        
        # Check if training should be stopped
        if self.model.stop_requested:
            return False
        return True
    
    def on_batch_end(self, batch_idx, batch_logs=None):
        """Called at the end of each batch with optional metrics"""
        self.model.current_batch = batch_idx + 1  # Already 1-based since batch completed
        if batch_logs:
            # Update running training metrics if provided
            self.model.train_loss = batch_logs.get('train_loss', self.model.train_loss)
            self.model.train_dice = batch_logs.get('train_dice', self.model.train_dice)
        self.model.save()
    
    def set_epoch_batches(self, total_batches):
        """Set total number of batches per epoch"""
        self.model.total_batches_per_epoch = total_batches
        self.model.save()
