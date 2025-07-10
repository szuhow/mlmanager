# Periodic tasks configuration for ML Manager
from celery.schedules import crontab
from django.conf import settings

# Periodic task schedule
CELERY_BEAT_SCHEDULE = {
    'cleanup-failed-trainings': {
        'task': 'ml_manager.cleanup_failed_trainings',
        'schedule': crontab(minute=0, hour='*/4'),  # Every 4 hours
        'options': {
            'expires': 3600,  # Task expires after 1 hour
        }
    },
    'system-health-check': {
        'task': 'ml_manager.system_health_check',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
        'options': {
            'expires': 900,  # Task expires after 15 minutes
        }
    },
}

# Configure timezone for beat schedule
CELERY_TIMEZONE = getattr(settings, 'TIME_ZONE', 'UTC')
