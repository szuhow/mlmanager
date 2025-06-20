"""
Testing settings for MLManager.
These settings are optimized for running tests with faster execution and isolation.
"""

from .base import *
import tempfile
import os

# Testing should never be in debug mode for performance
DEBUG = False

# Simple secret key for testing
SECRET_KEY = 'test-secret-key-not-for-production'

ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'testserver']

# In-memory SQLite database for fast testing
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
        'OPTIONS': {
            'timeout': 20,
        }
    }
}

# Disable migrations for faster test execution
class DisableMigrations:
    def __contains__(self, item):
        return True
    
    def __getitem__(self, item):
        return None

MIGRATION_MODULES = DisableMigrations()

# Fast password hasher for testing
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.MD5PasswordHasher',
]

# Dummy cache for testing
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
}

# Console email backend for testing
EMAIL_BACKEND = 'django.core.mail.backends.locmem.EmailBackend'

# Temporary directories for testing
TEMP_DIR = tempfile.mkdtemp()
MEDIA_ROOT = os.path.join(TEMP_DIR, 'media')
STATIC_ROOT = os.path.join(TEMP_DIR, 'static')

# MLflow settings for testing
MLFLOW_TRACKING_URI = 'sqlite:///test_mlflow.db'
MLFLOW_EXPERIMENT_NAME = 'test-experiments'
MLFLOW_UI_URL = 'http://localhost:5000'

# Test artifact storage
BASE_MLRUNS_DIR = os.path.join(TEMP_DIR, 'artifacts')
MLFLOW_ARTIFACT_ROOT = BASE_MLRUNS_DIR

# Simplified logging for testing
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'core.apps.ml_manager': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# Testing-specific Django REST Framework settings
REST_FRAMEWORK.update({
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',  # More permissive for testing
    ],
    'TEST_REQUEST_DEFAULT_FORMAT': 'json',
})

# Disable security features for testing
SECURE_SSL_REDIRECT = False
SECURE_HSTS_SECONDS = 0
SECURE_HSTS_INCLUDE_SUBDOMAINS = False
SECURE_HSTS_PRELOAD = False
SECURE_CONTENT_TYPE_NOSNIFF = False
SECURE_BROWSER_XSS_FILTER = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# Testing performance settings
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB

# Disable collectstatic during testing
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

# Test-specific installed apps
INSTALLED_APPS += [
    'django_nose',  # If using nose for testing
]

# Use nose as test runner (optional)
TEST_RUNNER = 'django_nose.NoseTestSuiteRunner'

# Nose configuration
NOSE_ARGS = [
    '--with-coverage',
    '--cover-package=core.apps.ml_manager,ml',
    '--cover-html',
    '--cover-html-dir=data/coverage',
    '--cover-branches',
    '--nocapture',
    '--nologcapture',
]
