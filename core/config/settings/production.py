"""
Production settings for MLManager.
These settings are optimized for production deployment with security and performance.
"""

from .base import *
from . import get_env_variable
import os
import logging

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = get_env_variable('SECRET_KEY', 'change-me-in-production-please')

# Production hosts
ALLOWED_HOSTS = [
    get_env_variable('DOMAIN_NAME', 'localhost'),
    get_env_variable('SERVER_IP', '127.0.0.1'),
    'mlmanager',
    'mlmanager.local',
    '0.0.0.0',  # Dodajemy dla Docker
]

# Database for production - PostgreSQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': get_env_variable('DB_NAME'),
        'USER': get_env_variable('DB_USER'),
        'PASSWORD': get_env_variable('DB_PASSWORD'),
        'HOST': get_env_variable('DB_HOST', 'localhost'),
        'PORT': get_env_variable('DB_PORT', '5432'),
        'OPTIONS': {
            'connect_timeout': 60,
        },
    }
}

# Production caching (simplified without Redis)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'enhanced_ml_manager_prod_cache',
        'OPTIONS': {
            'MAX_ENTRIES': 2000,
            'CULL_FREQUENCY': 3,
        }
    }
}

# Session configuration for production
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
SESSION_COOKIE_AGE = 3600  # 1 hour
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_SAMESITE = 'Lax'

# CSRF configuration for production
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = 'Lax'
CSRF_TRUSTED_ORIGINS = [
    f"https://{get_env_variable('DOMAIN_NAME', 'localhost')}",
]

# Security settings for production
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

# Email configuration for production
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = get_env_variable('EMAIL_HOST')
EMAIL_PORT = get_env_variable('EMAIL_PORT', '587')
EMAIL_USE_TLS = True
EMAIL_HOST_USER = get_env_variable('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = get_env_variable('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = get_env_variable('DEFAULT_FROM_EMAIL')

# MLflow settings for production
MLFLOW_TRACKING_URI = get_env_variable('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_EXPERIMENT_NAME = get_env_variable('MLFLOW_EXPERIMENT_NAME', 'coronary-experiments-prod')
MLFLOW_UI_URL = get_env_variable('MLFLOW_UI_URL', 'http://mlflow:5000')

# Artifact storage for production
BASE_MLRUNS_DIR = BASE_DIR / 'data' / 'artifacts'
MLFLOW_ARTIFACT_ROOT = get_env_variable('MLFLOW_ARTIFACT_ROOT', str(BASE_MLRUNS_DIR))

# Organized models storage for production
BASE_ORGANIZED_MODELS_DIR = BASE_DIR / 'data' / 'models' / 'organized'

# Production static files configuration
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'

# Production media files configuration
DEFAULT_FILE_STORAGE = 'django.core.files.storage.FileSystemStorage'

# Production logging configuration
LOGGING['handlers']['file']['filename'] = BASE_DIR / 'data' / 'logs' / 'production.log'
LOGGING['loggers']['django']['level'] = 'WARNING'
LOGGING['loggers']['core.apps.ml_manager']['level'] = 'INFO'

# Add error logging handler for production
LOGGING['handlers']['error_file'] = {
    'class': 'logging.FileHandler',
    'filename': BASE_DIR / 'data' / 'logs' / 'errors.log',
    'level': 'ERROR',
    'formatter': 'verbose',
}

LOGGING['loggers']['django']['handlers'].append('error_file')

# Production-specific Django REST Framework settings
REST_FRAMEWORK.update({
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        # Remove BrowsableAPIRenderer in production for security
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour'
    }
})

# Production performance settings
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50 MB

# Sentry configuration for error tracking (if using Sentry)
SENTRY_DSN = os.environ.get('SENTRY_DSN')
if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.django import DjangoIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    
    sentry_logging = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR
    )
    
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[DjangoIntegration(), sentry_logging],
        traces_sample_rate=0.1,
        send_default_pii=False,
        environment='production'
    )
