import mlflow
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import django

# Add the project root to Python path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

# --- Robust import error logging for Django/MLModel ---
try:
    from ml_manager.models import MLModel
except Exception as e:
    import logging
    logging.basicConfig(filename='models/artifacts/training.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', force=True)
    logging.error(f"Failed to import MLModel in training_callback.py: {e}")
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
        self.model.current_epoch = epoch
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
