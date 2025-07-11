"""
Standalone Celery configuration for beat and flower containers.
This configuration doesn't use Django at all to avoid any app loading issues.
"""
import os

# CRITICAL: Disable Django BEFORE importing Celery
os.environ.setdefault('CELERY_ALWAYS_EAGER', 'False')
# Prevent Django from being loaded by unsetting Django settings
if 'DJANGO_SETTINGS_MODULE' in os.environ:
    del os.environ['DJANGO_SETTINGS_MODULE']

# Force celery to NOT use Django fixup by pre-importing and disabling
os.environ['CELERY_LOADER'] = 'celery.loaders.app.AppLoader'

from celery import Celery
from celery.schedules import crontab

# Create the Celery app without Django
app = Celery('coronary_experiments_standalone')

# CRITICAL: Disable ALL fixups to prevent Django from loading
app.fixups = []

# Force non-Django loader BEFORE any configuration
app.loader_cls = 'celery.loaders.app:AppLoader'

# Configure Celery directly without Django settings
app.conf.update(
    broker_url=os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    result_backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
    timezone='UTC',
    beat_scheduler='celery.beat:PersistentScheduler',  # File-based scheduler instead of Django
    beat_schedule_filename='/app/core/data/celerybeat-schedule',
)

# Get container type
CONTAINER_TYPE = os.environ.get('CONTAINER_TYPE', 'beat')

if CONTAINER_TYPE == 'beat':
    # Configure periodic tasks for beat scheduler 
    # TODO: Re-enable once worker containers are running
    app.conf.beat_schedule = {
        # Temporarily disable all tasks to prevent Django module loading errors
        # 'cleanup-failed-trainings': {
        #     'task': 'core.apps.ml_manager.tasks.cleanup_failed_trainings',
        #     'schedule': crontab(minute=0, hour='*/4'),  # Every 4 hours
        #     'options': {
        #         'expires': 3600,  # Task expires after 1 hour
        #         'queue': 'default',
        #     }
        # },
        # 'system-health-check': {
        #     'task': 'core.apps.ml_manager.tasks.system_health_check', 
        #     'schedule': crontab(minute='*/15'),  # Every 15 minutes
        #     'options': {
        #         'expires': 900,  # Task expires after 15 minutes
        #         'queue': 'default',
        #     }
        # },
    }

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')

print(f"🔧 Standalone Celery ({CONTAINER_TYPE}): No Django dependencies, file-based beat scheduler")
