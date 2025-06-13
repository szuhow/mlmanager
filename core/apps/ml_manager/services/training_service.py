"""
ML training services.
"""

import subprocess
import sys
from pathlib import Path
from django.conf import settings
from ..models import MLModel
from ..utils.mlflow_utils import setup_mlflow_experiment


class MLTrainingService:
    """Service for handling ML model training."""
    
    def __init__(self, model_id):
        self.model = MLModel.objects.get(id=model_id)
        
    def start_training(self, training_params):
        """Start training process for the model."""
        # Setup MLflow experiment
        experiment_id = setup_mlflow_experiment(self.model.name)
        
        # Update model status
        self.model.status = 'loading'
        self.model.save()
        
        # Prepare training command
        training_script = Path(settings.BASE_DIR).parent / 'ml' / 'training' / 'train.py'
        
        command = [
            sys.executable,
            str(training_script),
            '--mode=train',
            f'--model-id={self.model.id}',
            f'--experiment-id={experiment_id}',
            f'--model-type={training_params.get("model_type", "unet")}',
            f'--data-path={training_params.get("data_path")}',
            f'--batch-size={training_params.get("batch_size", 32)}',
            f'--epochs={training_params.get("epochs", 10)}',
            f'--learning-rate={training_params.get("learning_rate", 0.001)}',
        ]
        
        # Start training process
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process ID for monitoring
            self.model.training_process_id = process.pid
            self.model.status = 'training'
            self.model.save()
            
            return {'success': True, 'process_id': process.pid}
            
        except Exception as e:
            self.model.status = 'failed'
            self.model.save()
            return {'success': False, 'error': str(e)}
    
    def get_training_status(self):
        """Get current training status."""
        return {
            'status': self.model.status,
            'progress': getattr(self.model, 'training_progress', 0),
            'current_epoch': getattr(self.model, 'current_epoch', 0),
            'total_epochs': getattr(self.model, 'total_epochs', 0),
        }
