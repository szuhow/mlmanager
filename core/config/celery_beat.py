"""
Celery configuration specifically for beat scheduler.
This configuration includes periodic tasks but avoids heavy ML dependencies.
"""
from celery import Celery
from celery.schedules import crontab
import os

# Create the Celery app for beat
app = Celery('coronary_experiments_beat')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Define periodic tasks directly here to avoid importing ml_manager apps
app.conf.beat_schedule = {
    'cleanup-failed-trainings': {
        'task': 'core.apps.ml_manager.tasks.cleanup_failed_trainings',
        'schedule': crontab(minute=0, hour='*/4'),  # Every 4 hours
        'options': {
            'expires': 3600,  # Task expires after 1 hour
            'queue': 'default',  # Route to default worker which has ML dependencies
        }
    },
    'system-health-check': {
        'task': 'core.apps.ml_manager.tasks.system_health_check',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
        'options': {
            'expires': 900,  # Task expires after 15 minutes
            'queue': 'default',  # Route to default worker which has ML dependencies
        }
    },
}

# Configure timezone
app.conf.timezone = 'UTC'

# Configure queue routing
app.conf.task_routes = {
    'core.apps.ml_manager.tasks.*': {'queue': 'default'},
}

# Configure basic queue settings
app.conf.task_default_queue = 'default'
app.conf.task_default_exchange = 'default'
app.conf.task_default_routing_key = 'default'

# DO NOT autodiscover tasks to avoid loading Django apps with cv2 dependencies
# The tasks will be executed by workers that have full ML environment

@app.task(bind=True)
def debug_task(self):
    print(f'Beat scheduler debug task: {self.request!r}')

print("🔧 Beat Celery Configuration: Periodic tasks defined, routing to ML workers")
