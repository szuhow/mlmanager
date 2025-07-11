"""
Minimal settings for Celery Beat scheduler.
Contains only core Django functionality without ML applications.
"""

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Environment detection
def get_env_variable(var_name, default=None):
    """Get environment variable or return default."""
    return os.environ.get(var_name, default)

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = get_env_variable('SECRET_KEY', 'django-insecure-beat-scheduler-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = get_env_variable('DEBUG', 'False').lower() in ('true', '1', 'yes')

ALLOWED_HOSTS = ['*']  # Beat runs internally

# Minimal application definition - only what's needed for Celery Beat
DJANGO_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
]

THIRD_PARTY_APPS = [
    'django_celery_beat',  # Required for beat scheduler
]

# Only include minimal apps without ML dependencies
LOCAL_APPS = []

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'core.config.urls'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': get_env_variable('POSTGRES_DB', 'mlmanager'),
        'USER': get_env_variable('POSTGRES_USER', 'postgres'),
        'PASSWORD': get_env_variable('POSTGRES_PASSWORD', 'postgres'),
        'HOST': get_env_variable('POSTGRES_HOST', 'postgres'),
        'PORT': get_env_variable('POSTGRES_PORT', '5432'),
        'OPTIONS': {
            'connect_timeout': 60,
            'options': '-c default_transaction_isolation=read_committed'
        }
    }
}

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Celery Configuration
CELERY_BROKER_URL = get_env_variable('CELERY_BROKER_URL', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = get_env_variable('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

# Celery Beat specific settings
CELERY_BEAT_SCHEDULER = 'django_celery_beat.schedulers:DatabaseScheduler'
CELERY_TIMEZONE = TIME_ZONE

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'celery': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

print("🔧 Celery Beat Settings: Minimal configuration without ML dependencies")
