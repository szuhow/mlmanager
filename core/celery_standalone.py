"""
Standalone Celery configuration for beat scheduler.
This runs without Django to avoid module import issues.
"""
from celery import Celery
from celery.schedules import crontab
import os

# Get environment variables
REDIS_URL = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

# Create the Celery app without Django - important: don't include Django fixups
app = Celery('coronary_beat_standalone', 
             broker=REDIS_URL,
             backend=RESULT_BACKEND,
             fixups=[])  # Empty fixups list to disable Django fixup

# Configure basic settings
app.conf.update(
    # Timezone
    timezone='UTC',
    
    # Queue settings
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
    
    # Serialization
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    
    # Routing - all tasks go to default queue for ML workers
    task_routes={
        'core.apps.ml_manager.tasks.*': {'queue': 'default'},
    },
    
    # Beat schedule - periodic tasks defined here
    beat_schedule={
        'cleanup-failed-trainings': {
            'task': 'core.apps.ml_manager.tasks.cleanup_failed_trainings',
            'schedule': crontab(minute=0, hour='*/4'),  # Every 4 hours
            'options': {
                'expires': 3600,
                'queue': 'default',
            }
        },
        'system-health-check': {
            'task': 'core.apps.ml_manager.tasks.system_health_check',
            'schedule': crontab(minute='*/15'),  # Every 15 minutes
            'options': {
                'expires': 900,
                'queue': 'default',
            }
        },
    }
)

print("🔧 Standalone Beat Celery: No Django, direct Redis connection")
print(f"📡 Broker: {REDIS_URL}")
print(f"📊 Backend: {RESULT_BACKEND}")
print("⏰ Periodic tasks configured, routing to default queue")

if __name__ == '__main__':
    app.start()
