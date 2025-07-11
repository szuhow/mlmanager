#!/bin/bash

# Enhanced ML Manager - Celery startup script
# This script starts Celery workers and beat scheduler for ML training management

set -e

echo "🚀 Starting Enhanced ML Manager Celery Services"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Redis is running
echo "🔍 Checking Redis connection..."
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is running${NC}"
else
    echo -e "${RED}❌ Redis is not running. Please start Redis first:${NC}"
    echo "   sudo service redis-server start"
    echo "   or: redis-server"
    exit 1
fi

# Check if Django settings are configured
echo "🔍 Checking Django configuration..."
python -c "
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.dev')
import django
django.setup()
from django.conf import settings
assert hasattr(settings, 'CELERY_BROKER_URL'), 'CELERY_BROKER_URL not configured'
print('✅ Django Celery configuration OK')
" || {
    echo -e "${RED}❌ Django Celery configuration missing${NC}"
    echo "Please add Celery configuration to your Django settings"
    exit 1
}

# Create log directory
mkdir -p logs

# Function to start Celery worker
start_worker() {
    local queue_name=$1
    local worker_name=$2
    local concurrency=${3:-1}
    
    echo -e "${YELLOW}🏃 Starting Celery worker: ${worker_name}${NC}"
    
    celery -A core worker \
        --loglevel=info \
        --queues="${queue_name}" \
        --hostname="${worker_name}@%h" \
        --concurrency="${concurrency}" \
        --logfile="logs/celery_${worker_name}.log" \
        --pidfile="logs/celery_${worker_name}.pid" \
        --detach
    
    echo -e "${GREEN}✅ ${worker_name} worker started${NC}"
}

# Function to start Celery beat
start_beat() {
    echo -e "${YELLOW}🥁 Starting Celery beat scheduler${NC}"
    
    celery -A core beat \
        --loglevel=info \
        --logfile="logs/celery_beat.log" \
        --pidfile="logs/celery_beat.pid" \
        --detach
    
    echo -e "${GREEN}✅ Celery beat started${NC}"
}

# Start different workers for different types of tasks
echo "🔧 Starting specialized Celery workers..."

# Training worker (single concurrency for GPU tasks)
start_worker "training" "training_worker" 1

# Inference worker (can handle multiple concurrent inference tasks)
start_worker "inference" "inference_worker" 2

# Control worker (for stopping tasks, etc.)
start_worker "control" "control_worker" 1

# Maintenance worker (for cleanup tasks)
start_worker "maintenance" "maintenance_worker" 1

# Start beat scheduler for periodic tasks
start_beat

echo ""
echo -e "${GREEN}🎉 All Celery services started successfully!${NC}"
echo ""
echo "📊 To monitor workers:"
echo "   celery -A core status"
echo "   celery -A core inspect active"
echo ""
echo "📝 Log files:"
echo "   Training:    logs/celery_training_worker.log"
echo "   Inference:   logs/celery_inference_worker.log"
echo "   Control:     logs/celery_control_worker.log"
echo "   Maintenance: logs/celery_maintenance_worker.log"
echo "   Beat:        logs/celery_beat.log"
echo ""
echo "🛑 To stop all workers:"
echo "   ./scripts/stop_celery.sh"
echo ""

# Optional: Start Flower for monitoring (if installed)
if command -v flower &> /dev/null; then
    echo -e "${YELLOW}🌸 Starting Flower monitoring (optional)${NC}"
    flower -A core --port=5555 --broker=redis://localhost:6379/0 &
    echo -e "${GREEN}✅ Flower started at http://localhost:5555${NC}"
else
    echo -e "${YELLOW}💡 Install flower for web monitoring: pip install flower${NC}"
fi

echo ""
echo -e "${GREEN}🚀 Enhanced ML Manager is ready for training and inference!${NC}"
