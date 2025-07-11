"""
Django-based Celery configuration for worker containers.
This configuration requires Django and all ML dependencies.
"""
import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

app = Celery('coronary_experiments')

# Configure Celery using settings from Django settings.py.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Configure task routes for different queues
app.conf.task_routes = {
    # Training tasks go to training queue
    'core.apps.ml_manager.tasks.train_model': {'queue': 'training'},
    'core.apps.ml_manager.tasks.retrain_model': {'queue': 'training'},
    'core.apps.ml_manager.tasks.preprocess_dataset': {'queue': 'training'},
    'core.apps.ml_manager.tasks.validate_model': {'queue': 'training'},
    
    # Inference tasks go to inference queue
    'core.apps.ml_manager.tasks.run_inference': {'queue': 'inference'},
    'core.apps.ml_manager.tasks.batch_inference': {'queue': 'inference'},
    'core.apps.ml_manager.tasks.predict_image': {'queue': 'inference'},
    
    # Default tasks remain in default queue
    'core.apps.ml_manager.tasks.cleanup_temp_files': {'queue': 'default'},
    'core.apps.ml_manager.tasks.update_model_status': {'queue': 'default'},
}

# Configure queue priorities
app.conf.task_default_queue = 'default'
app.conf.task_default_exchange = 'default'
app.conf.task_default_routing_key = 'default'

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')

# Health check tasks for each worker type
@app.task(bind=True, queue='training')
def training_worker_health_check(self):
    """Health check for training workers"""
    import torch
    import monai
    return {
        'worker_type': 'training',
        'torch_version': torch.__version__,
        'monai_version': monai.__version__,
        'cuda_available': torch.cuda.is_available(),
        'status': 'healthy'
    }

@app.task(bind=True, queue='inference')
def inference_worker_health_check(self):
    """Health check for inference workers"""
    import torch
    import numpy
    return {
        'worker_type': 'inference',
        'torch_version': torch.__version__,
        'numpy_version': numpy.__version__,
        'cuda_available': torch.cuda.is_available(),
        'status': 'healthy'
    }
