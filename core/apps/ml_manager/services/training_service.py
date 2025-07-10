"""
ML training services with Celery integration.
"""

import subprocess
import sys
from pathlib import Path
from django.conf import settings
from ..models import MLModel
from ..utils.mlflow_utils import setup_mlflow_experiment


class MLTrainingService:
    """Service for handling ML model training with Celery."""
    
    def __init__(self, model_id):
        self.model = MLModel.objects.get(id=model_id)
        
    def start_training(self, training_params):
        """Start training process for the model using Celery."""
        try:
            # Import Celery task
            from ..tasks.tasks import train_model_task
            
            # DON'T setup MLflow experiment here - it will be handled in Celery task
            # based on user's experiment selection from the form
            
            # Update model status
            self.model.status = 'training'  # Changed from 'loading' to 'training' for consistency
            self.model.save()
            
            # Start training task asynchronously
            task = train_model_task.delay(self.model.id, training_params)
            
            # Store only task ID, not full logs
            self.model.training_logs = f"Training started - Task ID: {task.id}. Logs are saved to files in model directory."
            self.model.celery_task_id = task.id  # Add task_id to model if field exists
            self.model.save()
            
            return {
                'success': True, 
                'task_id': task.id,
                'model_id': self.model.id,
                'status': 'queued'
            }
            
        except Exception as e:
            self.model.status = 'failed'
            self.model.training_logs = f"Training failed to start: {str(e)}. Check logs in model directory for details."
            self.model.save()
            return {'success': False, 'error': str(e)}
    
    def stop_training(self):
        """Stop training process using Celery."""
        try:
            # Import Celery task
            from ..tasks.tasks import stop_training_task
            
            # Start stop task asynchronously
            task = stop_training_task.delay(self.model.id)
            
            # Update model status
            self.model.status = 'stopping'
            self.model.save()
            
            return {
                'success': True,
                'task_id': task.id,
                'model_id': self.model.id,
                'status': 'stopping'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_training_status(self):
        """Get current training status."""
        return {
            'status': self.model.status,
            'progress': getattr(self.model, 'training_progress', 0),
            'current_epoch': getattr(self.model, 'current_epoch', 0),
            'total_epochs': getattr(self.model, 'total_epochs', 0),
            'train_loss': getattr(self.model, 'train_loss', 0.0),
            'val_loss': getattr(self.model, 'val_loss', 0.0),
            'train_dice': getattr(self.model, 'train_dice', 0.0),
            'val_dice': getattr(self.model, 'val_dice', 0.0),
        }
