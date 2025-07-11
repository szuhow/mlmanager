"""
Development settings for MLManager.
These settings are used during development and include debugging features.
"""

from .base import *
import os

# Override LOGGING before any other settings are processed
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'core.apps.ml_manager': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'django.db.backends': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.utils.autoreload': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
}

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get(
    'SECRET_KEY', 
    'django-insecure-dev-key-change-in-production'
)

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
    'mlmanager',
    'testserver',  # For Django test client
]

# Database for development - SQLite
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': CORE_DATA_DIR / 'db.sqlite3',
    }
}

# Development-specific installed apps
# INSTALLED_APPS += [
#     'django_extensions',  # Useful development tools - commented out for Docker compatibility
# ]

# Development middleware - disable CSRF for testing
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',  # Disabled for testing
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Email backend for development (console output)
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# MLflow settings for development - temporarily disabled for debugging
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'coronary-experiments-dev')
MLFLOW_UI_URL = os.environ.get('MLFLOW_UI_URL', 'http://localhost:5000')

# Enable MLflow connection for artifact recording
MLFLOW_ENABLED = True

# Artifact storage for development
BASE_MLRUNS_DIR = CORE_DATA_DIR / 'mlflow'
MLFLOW_ARTIFACT_ROOT = os.environ.get('MLFLOW_ARTIFACT_ROOT', str(BASE_MLRUNS_DIR))

# Organized models storage for development
BASE_ORGANIZED_MODELS_DIR = CORE_DATA_DIR / 'models' / 'organized'

# Development-specific caching (use in-memory cache for better performance)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
        'TIMEOUT': 300,  # 5 minutes default
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        }
    }
}

# Development logging configuration - handled at the top of the file

# Development-specific settings for Django REST Framework
REST_FRAMEWORK.update({
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',  # Keep browsable API in dev
    ],
})

# Development security settings (more relaxed)
SECURE_SSL_REDIRECT = False
SECURE_HSTS_SECONDS = 0
SECURE_HSTS_INCLUDE_SUBDOMAINS = False
SECURE_HSTS_PRELOAD = False
SECURE_CONTENT_TYPE_NOSNIFF = False
SECURE_BROWSER_XSS_FILTER = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# CSRF settings for Docker development
CSRF_TRUSTED_ORIGINS = [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://0.0.0.0:8000',
]

CSRF_COOKIE_HTTPONLY = False
CSRF_USE_SESSIONS = False

# Internal IPs for Django Debug Toolbar (if added)
INTERNAL_IPS = [
    '127.0.0.1',
    'localhost',
]

# Celery Beat configuration for periodic tasks
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    'sync-mlflow-data': {
        'task': 'ml_manager.sync_mlflow_data',
        'schedule': crontab(minute='*/10'),  # Every 10 minutes
        'options': {'expires': 60 * 5},  # Expire task if not run within 5 minutes
    },
}

CELERY_TIMEZONE = 'UTC'
