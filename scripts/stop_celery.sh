#!/bin/bash

# Enhanced ML Manager - Celery stop script
# This script stops all Celery workers and beat scheduler

set -e

echo "🛑 Stopping Enhanced ML Manager Celery Services"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to stop worker by PID file
stop_worker() {
    local worker_name=$1
    local pid_file="logs/celery_${worker_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo -e "${YELLOW}🛑 Stopping ${worker_name} (PID: ${pid})${NC}"
        
        if kill -TERM "$pid" 2>/dev/null; then
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${RED}⚠️  Force killing ${worker_name}${NC}"
                kill -KILL "$pid" 2>/dev/null || true
            fi
            
            rm -f "$pid_file"
            echo -e "${GREEN}✅ ${worker_name} stopped${NC}"
        else
            echo -e "${RED}❌ Failed to stop ${worker_name} (PID: ${pid})${NC}"
            rm -f "$pid_file"  # Clean up stale PID file
        fi
    else
        echo -e "${YELLOW}ℹ️  ${worker_name} PID file not found${NC}"
    fi
}

# Stop all workers
echo "🔧 Stopping Celery workers..."

stop_worker "training_worker"
stop_worker "inference_worker" 
stop_worker "control_worker"
stop_worker "maintenance_worker"

# Stop beat scheduler
echo -e "${YELLOW}🥁 Stopping Celery beat scheduler${NC}"
stop_worker "beat"

# Also try to stop any remaining Celery processes
echo "🧹 Cleaning up any remaining Celery processes..."

# Find and stop any remaining celery processes
if pgrep -f "celery.*worker" > /dev/null; then
    echo -e "${YELLOW}⚠️  Found remaining Celery worker processes, stopping them...${NC}"
    pkill -TERM -f "celery.*worker" || true
    sleep 2
    pkill -KILL -f "celery.*worker" 2>/dev/null || true
fi

if pgrep -f "celery.*beat" > /dev/null; then
    echo -e "${YELLOW}⚠️  Found remaining Celery beat processes, stopping them...${NC}"
    pkill -TERM -f "celery.*beat" || true
    sleep 2
    pkill -KILL -f "celery.*beat" 2>/dev/null || true
fi

# Stop Flower if running
if pgrep -f "flower" > /dev/null; then
    echo -e "${YELLOW}🌸 Stopping Flower monitoring${NC}"
    pkill -TERM -f "flower" || true
fi

# Clean up any stale PID files
echo "🧹 Cleaning up PID files..."
rm -f logs/celery_*.pid

echo ""
echo -e "${GREEN}✅ All Celery services stopped successfully!${NC}"
echo ""

# Check if any Celery processes are still running
if pgrep -f "celery" > /dev/null; then
    echo -e "${RED}⚠️  Warning: Some Celery processes may still be running:${NC}"
    pgrep -f "celery" -l
    echo ""
    echo "You may need to manually kill them with:"
    echo "   pkill -9 -f celery"
else
    echo -e "${GREEN}🎉 No Celery processes are running${NC}"
fi

echo ""
echo -e "${GREEN}🚀 To restart Celery services, run:${NC}"
echo "   ./scripts/start_celery.sh"
