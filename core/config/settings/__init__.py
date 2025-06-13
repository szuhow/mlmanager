"""
Django settings package for MLManager.

This package contains modular Django settings that can be used in different environments:
- base.py: Common settings shared across all environments
- development.py: Development-specific settings
- production.py: Production-specific settings
- testing.py: Testing-specific settings

Usage:
    Set DJANGO_SETTINGS_MODULE environment variable:
    - For development: 'core.config.settings.development'
    - For production: 'core.config.settings.production'
    - For testing: 'core.config.settings.testing'
"""

import os
from django.core.exceptions import ImproperlyConfigured

def get_env_variable(var_name, default=None):
    """Get environment variable or raise exception."""
    try:
        return os.environ[var_name]
    except KeyError:
        if default is not None:
            return default
        error_msg = f'Set the {var_name} environment variable'
        raise ImproperlyConfigured(error_msg)
