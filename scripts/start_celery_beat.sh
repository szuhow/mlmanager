#!/bin/bash
# Start Celery Beat scheduler for periodic tasks

# Exit on any error
set -e

# Set Django settings
export DJANGO_SETTINGS_MODULE=core.config.settings.container

# Navigate to the core directory
cd /app/core

# Start Celery Beat
echo "Starting Celery Beat scheduler..."
celery -A core.config.celery beat \
    --loglevel=INFO \
    --scheduler=django_celery_beat.schedulers:DatabaseScheduler \
    --pidfile=/tmp/celerybeat.pid \
    --schedule=/tmp/celerybeat-schedule
