"""
Worker-specific settings for Celery workers with GPU support.
Used in docker-compose.worker.yml
"""

from .base import *
import sys

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-worker-key-change-in-production')

# Add ML modules to Python path for worker
sys.path.insert(0, str(BASE_DIR / 'ml'))
sys.path.insert(0, str(BASE_DIR))

# Worker settings dictionary
WORKER_SETTINGS = {
    'worker_type': 'worker',
    'gpu_enabled': True,
    'max_gpu_memory_gb': int(os.environ.get('MAX_GPU_MEMORY_GB', '8')),
    'training_timeout': int(os.environ.get('TRAINING_TIMEOUT', '86400')),  # 24 hours
    'inference_timeout': int(os.environ.get('INFERENCE_TIMEOUT', '300')),   # 5 minutes
    'max_concurrent_gpu_tasks': int(os.environ.get('MAX_CONCURRENT_GPU_TASKS', '1')),
}

# Client settings dictionary
CLIENT_SETTINGS = {
    'enable_training_ui': False,
    'enable_inference_ui': False,
    'enable_preset_management': False,
    'enable_model_management': False,
    'enable_dataset_management': False,
    'enable_experiment_tracking': False,
}

# Health check settings dictionary
HEALTH_CHECK_SETTINGS = {
    'gpu_health_check_interval': 60,
    'model_loading_timeout': 300,
    'training_heartbeat_interval': 30,
}

# Feature flags dictionary
FEATURE_FLAGS = {
    'use_modern_training_system': True,
    'enable_gpu_training': True,
    'enable_distributed_training': True,
    'enable_model_checkpointing': True,
    'enable_automatic_cleanup': True,
}

# Worker-specific Celery configuration
CELERY_WORKER_CONCURRENCY = int(os.environ.get('CELERY_WORKER_CONCURRENCY', '1'))
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # Critical for GPU memory management
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1  # Prevent GPU memory leaks

# Worker queues configuration
CELERY_WORKER_QUEUES = os.environ.get('CELERY_WORKER_QUEUES', 'training,default').split(',')

# Database configuration for worker (read-only mostly)
import dj_database_url

# Use DATABASE_URL if provided, otherwise fall back to individual variables
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    DATABASES = {
        'default': dj_database_url.parse(DATABASE_URL)
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.environ.get('POSTGRES_DB', 'mlmanager'),
            'USER': os.environ.get('POSTGRES_USER', 'mlmanager'),
            'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'mlmanager_pass'),
            'HOST': os.environ.get('POSTGRES_HOST', 'postgres'),
            'PORT': os.environ.get('POSTGRES_PORT', '5432'),
            'OPTIONS': {
                'connect_timeout': 60,
            }
        }
    }

# Minimal static files for worker
STATIC_URL = '/static/'
STATIC_ROOT = '/app/core/data/staticfiles'

# Media files configuration for worker
MEDIA_URL = '/media/'
MEDIA_ROOT = '/app/core/data/media'

# Worker-specific paths
MODELS_DIR = Path('/app/core/data/models')
CHECKPOINTS_DIR = Path('/app/core/data/checkpoints')
TRAINING_RESULTS_DIR = Path('/app/core/data/training_results')
INFERENCE_RESULTS_DIR = Path('/app/core/data/inference_results')

# Ensure worker directories exist
for directory in [MODELS_DIR, CHECKPOINTS_DIR, TRAINING_RESULTS_DIR, INFERENCE_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# MLflow configuration for worker - artefakty w strukturze /app/core/data/mlflow
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_BACKEND_STORE_URI = os.environ.get('MLFLOW_BACKEND_STORE_URI', 'sqlite:////app/core/data/mlflow/data/mlflow.db')

# MLflow artifact root wskazuje na /app/core/data/mlflow - eksperyment i run ID będą automatycznie dodane
# Struktura: /app/core/data/mlflow/{experiment_id}/{run_id}/artifacts/
BASE_ORGANIZED_MODELS_DIR = '/app/core/data/models/organized'  # zachowane dla kompatybilności
MLFLOW_ARTIFACT_ROOT = os.environ.get('MLFLOW_ARTIFACT_ROOT', '/app/core/data/mlflow')
MLFLOW_ARTIFACTS_DESTINATION = '/app/core/data/mlflow'

# MLflow system paths (for database and logs)
MLFLOW_DATA_PATH = os.environ.get('MLFLOW_DATA_PATH', '/app/core/data/mlflow/data') 
MLFLOW_LOGS_PATH = os.environ.get('MLFLOW_LOGS_PATH', '/app/core/data/mlflow/logs')

# MLflow artifact settings for training workers
MLFLOW_LOG_ARTIFACTS = True
MLFLOW_LOG_MODELS = True
MLFLOW_LOG_PARAMS = True
MLFLOW_LOG_METRICS = True
MLFLOW_SERVE_ARTIFACTS = True

# Training-specific artifact structure:
# /app/core/data/models/organized/{year}/{month}/{model_family}/{model_name}/
#   ├── artifacts/          <- MLflow artifacts
#   ├── checkpoints/        <- Model checkpoints  
#   ├── logs/              <- Training logs
#   ├── predictions/       <- Sample predictions
#   └── config/            <- Training configuration

# GPU monitoring settings
GPU_MONITORING = {
    'enable_gpu_monitoring': True,
    'gpu_memory_threshold': 0.9,  # 90% memory usage warning
    'gpu_utilization_threshold': 0.95,  # 95% utilization warning
    'monitoring_interval': 30,  # seconds
}

# Training-specific settings
TRAINING_SETTINGS = {
    'max_epochs_per_training': int(os.environ.get('MAX_EPOCHS_PER_TRAINING', '1000')),
    'checkpoint_save_interval': int(os.environ.get('CHECKPOINT_SAVE_INTERVAL', '10')),
    'enable_mixed_precision': os.environ.get('ENABLE_MIXED_PRECISION', 'true').lower() == 'true',
    'gradient_clipping_threshold': float(os.environ.get('GRADIENT_CLIPPING_THRESHOLD', '1.0')),
}

# Worker health check settings
HEALTH_CHECK_SETTINGS.update({
    'gpu_health_check_interval': 60,
    'model_loading_timeout': 300,
    'training_heartbeat_interval': 30,
})

# Worker-specific logging - use console logging for Docker
LOGGING['handlers']['worker_console'] = {
    'class': 'logging.StreamHandler',
    'formatter': 'verbose',
    'stream': 'ext://sys.stdout',
}

# Create log directory and try to use file logging, but fall back to console if permissions fail
log_dir = '/app/core/data/logs'
try:
    os.makedirs(log_dir, exist_ok=True)
    LOGGING['handlers']['worker_file'] = {
        'class': 'logging.FileHandler',
        'filename': '/app/core/data/logs/worker.log',
        'formatter': 'verbose',
    }
    use_file_logging = True
except Exception as e:
    use_file_logging = False
    print(f"Warning: Could not setup file logging: {e}")

# Override base.py file handler to avoid django.log issues
try:
    os.makedirs('/app/core/data/logs', exist_ok=True)
    LOGGING['handlers']['file'] = {
        'class': 'logging.StreamHandler',  # Use console instead of file for worker
        'formatter': 'verbose',
        'stream': 'ext://sys.stdout',
    }
except Exception as e:
    print(f"Warning: Could not setup logs directory: {e}")

LOGGING['loggers'].update({
    'ml.training.core': {
        'handlers': ['console', 'worker_console'] + (['worker_file'] if use_file_logging else []),
        'level': 'DEBUG',
        'propagate': False,
    },
    'ml.training.celery_integration': {
        'handlers': ['console', 'worker_console'] + (['worker_file'] if use_file_logging else []),
        'level': 'DEBUG',
        'propagate': False,
    },
    'celery.worker': {
        'handlers': ['console', 'worker_console'] + (['worker_file'] if use_file_logging else []),
        'level': 'INFO',
        'propagate': False,
    },
    'torch': {
        'handlers': ['console', 'worker_console'] + (['worker_file'] if use_file_logging else []),
        'level': 'WARNING',
        'propagate': False,
    },
})

# Fix admin requirements for worker
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',  # Required for admin
    'django.contrib.messages',  # Required for admin
    'django.contrib.staticfiles',
    'rest_framework',
    'core.apps.ml_manager',
    'core.apps.dataset_manager',
]

# Complete middleware stack for worker
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',  # Required for admin
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',  # Required for admin
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Disable web-specific features
DEBUG = False
ALLOWED_HOSTS = ['*']  # Worker doesn't serve HTTP

# Feature flags for worker
FEATURE_FLAGS.update({
    'use_modern_training_system': True,
    'enable_gpu_training': True,
    'enable_distributed_training': True,
    'enable_model_checkpointing': True,
    'enable_automatic_cleanup': True,
})

# Worker-specific environment variables
WORKER_ID = os.environ.get('WORKER_ID', 'worker-1')
WORKER_GPU_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

# Task result expiration
CELERY_RESULT_EXPIRES = 3600  # 1 hour
