"""
Enhanced Django settings for Docker container deployment.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-dev-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1,0.0.0.0').split(',')

# Application definition
DJANGO_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

LOCAL_APPS = [
    'apps.ml_manager',
    'apps.dataset_manager',
]

THIRD_PARTY_APPS = [
    'rest_framework',
]

INSTALLED_APPS = DJANGO_APPS + LOCAL_APPS + THIRD_PARTY_APPS

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'data' / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'data' / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ==============================================
# Enhanced ML Manager Settings
# ==============================================
ML_MANAGER_SETTINGS = {
    'MAX_INFERENCE_IMAGES': int(os.environ.get('MAX_INFERENCE_IMAGES', 50)),
    'TEMP_FILE_CLEANUP_HOURS': int(os.environ.get('TEMP_FILE_CLEANUP_HOURS', 24)),
    'TRAINING_LOG_MAX_LINES': 1000,
    
    # Post-processing defaults
    'DEFAULT_POST_PROCESSING': {
        'threshold': float(os.environ.get('DEFAULT_THRESHOLD', 0.6)),
        'min_component_size': int(os.environ.get('DEFAULT_MIN_COMPONENT_SIZE', 50)),
        'morphology_kernel_size': int(os.environ.get('DEFAULT_MORPHOLOGY_KERNEL_SIZE', 3)),
        'apply_opening': os.environ.get('ENABLE_MORPHOLOGICAL_OPENING', 'true').lower() == 'true',
        'apply_closing': os.environ.get('ENABLE_MORPHOLOGICAL_CLOSING', 'true').lower() == 'true',
        'confidence_threshold': float(os.environ.get('DEFAULT_CONFIDENCE_THRESHOLD', 0.6)),
        'use_adaptive_threshold': os.environ.get('ENABLE_ADAPTIVE_THRESHOLD', 'false').lower() == 'true',
    },
    
    # Enhanced loss function defaults
    'DEFAULT_LOSS_CONFIG': {
        'loss_function': os.environ.get('DEFAULT_LOSS_FUNCTION', 'combined_dice_focal'),
        'dice_weight': float(os.environ.get('DEFAULT_DICE_WEIGHT', 0.7)),
        'focal_alpha': float(os.environ.get('DEFAULT_FOCAL_ALPHA', 0.25)),
        'focal_gamma': float(os.environ.get('DEFAULT_FOCAL_GAMMA', 2.0)),
        'weight_decay': float(os.environ.get('DEFAULT_WEIGHT_DECAY', 1e-4)),
        'dropout_rate': float(os.environ.get('DEFAULT_DROPOUT_RATE', 0.1)),
    },
    
    # Training monitoring
    'TRAINING_MONITORING': {
        'log_interval': 10,
        'save_checkpoint_interval': 5,
        'max_checkpoints_keep': 5,
        'early_stopping_patience': int(os.environ.get('DEFAULT_EARLY_STOPPING_PATIENCE', 15)),
        'early_stopping_metric': os.environ.get('DEFAULT_EARLY_STOPPING_METRIC', 'val_dice_score'),
    }
}

# MLflow configuration
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_ARTIFACT_ROOT = os.environ.get('MLFLOW_ARTIFACT_ROOT', '/app/data/mlflow')

# ==============================================
# Logging Configuration
# ==============================================
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': os.environ.get('ML_MANAGER_LOG_LEVEL', 'INFO'),
            'class': 'logging.FileHandler',
            'filename': '/app/logs/ml_manager.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': os.environ.get('LOG_LEVEL', 'INFO'),
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'apps.ml_manager': {
            'handlers': ['file', 'console'],
            'level': os.environ.get('ML_MANAGER_LOG_LEVEL', 'INFO'),
            'propagate': True,
        },
        'django': {
            'handlers': ['console'],
            'level': os.environ.get('DJANGO_LOG_LEVEL', 'INFO'),
            'propagate': True,
        },
    },
}

# ==============================================
# Security Settings
# ==============================================
# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get('FILE_UPLOAD_MAX_MEMORY_SIZE', 10 * 1024 * 1024))  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get('DATA_UPLOAD_MAX_MEMORY_SIZE', 50 * 1024 * 1024))  # 50MB
FILE_UPLOAD_PERMISSIONS = 0o644

# Model storage settings
MODEL_STORAGE_SETTINGS = {
    'ROOT_DIR': MEDIA_ROOT / 'models',
    'CHECKPOINT_DIR': MEDIA_ROOT / 'checkpoints',
    'INFERENCE_TEMP_DIR': MEDIA_ROOT / 'temp' / 'inference',
    'LOG_DIR': BASE_DIR / 'logs',
    'MAX_MODEL_SIZE': int(os.environ.get('MAX_MODEL_SIZE', 500 * 1024 * 1024)),  # 500MB
}

# Create necessary directories
for directory in MODEL_STORAGE_SETTINGS.values():
    if isinstance(directory, Path):
        directory.mkdir(parents=True, exist_ok=True)

# ==============================================
# REST Framework Configuration
# ==============================================
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

# ==============================================
# Cache Configuration (simplified - no Redis)
# ==============================================
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'enhanced_ml_manager_cache',
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
            'CULL_FREQUENCY': 3,
        },
        'TIMEOUT': 300,  # 5 minutes default timeout
    }
}

# Use Redis for sessions
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
