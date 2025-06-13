"""
Django settings router for MLManager.

This module automatically loads the appropriate settings module based on the
DJANGO_SETTINGS_MODULE environment variable or ENVIRONMENT variable.

Available environments:
- development (default)
- production
- testing

Usage:
    Set environment variable:
    export ENVIRONMENT=production
    
    Or use specific settings module:
    export DJANGO_SETTINGS_MODULE=core.config.settings.production
"""

import os
import sys

# Determine which settings to use
environment = os.environ.get('ENVIRONMENT', 'development').lower()

# Mapping of environment names to settings modules
SETTINGS_MODULES = {
    'development': 'core.config.settings.development',
    'dev': 'core.config.settings.development',
    'production': 'core.config.settings.production',
    'prod': 'core.config.settings.production',
    'testing': 'core.config.settings.testing',
    'test': 'core.config.settings.testing',
}

# Get the appropriate settings module
settings_module = SETTINGS_MODULES.get(environment)

if not settings_module:
    available_envs = ', '.join(SETTINGS_MODULES.keys())
    raise Exception(
        f"Unknown environment: {environment}. "
        f"Available environments: {available_envs}"
    )

# Import all settings from the chosen module
try:
    module = __import__(settings_module, fromlist=['*'])
    for setting in dir(module):
        if setting.isupper():
            globals()[setting] = getattr(module, setting)
except ImportError as e:
    raise Exception(
        f"Could not import settings module '{settings_module}': {e}"
    )

# Print which settings are being used
print(f"🔧 Django Settings Router Debug:")
print(f"🌍 ENVIRONMENT env var: {os.environ.get('ENVIRONMENT', 'NOT_SET')}")
print(f"🎯 Resolved environment: {environment}")
print(f"📦 Settings module: {settings_module}")

if environment in ['development', 'dev'] and 'runserver' in sys.argv:
    print(f"🔧 Using Django settings: {settings_module}")
    print(f"🌍 Environment: {environment}")
    print(f"🐛 Debug mode will be determined by chosen module")
