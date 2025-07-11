#!/bin/bash
# MLManager Backup Script
# Automated backup for data, database, and configurations

set -e

# Configuration
BACKUP_DIR="/data/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "Starting MLManager backup - $TIMESTAMP"

# Backup database
echo "Backing up database..."
if [ -f "/data/mlflow/mlflow.db" ]; then
    cp "/data/mlflow/mlflow.db" "$BACKUP_DIR/mlflow_db_$TIMESTAMP.db"
fi

# Backup trained models
echo "Backing up trained models..."
if [ -d "/data/models" ]; then
    tar -czf "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" -C /data models/
fi

# Backup MLflow artifacts
echo "Backing up MLflow artifacts..."
if [ -d "/data/artifacts" ]; then
    tar -czf "$BACKUP_DIR/artifacts_$TIMESTAMP.tar.gz" -C /data artifacts/
fi

# Backup datasets (if small enough)
echo "Backing up datasets..."
if [ -d "/data/datasets" ]; then
    DATASET_SIZE=$(du -sb /data/datasets | cut -f1)
    MAX_SIZE=1073741824  # 1GB in bytes
    
    if [ "$DATASET_SIZE" -lt "$MAX_SIZE" ]; then
        tar -czf "$BACKUP_DIR/datasets_$TIMESTAMP.tar.gz" -C /data datasets/
    else
        echo "Datasets too large for backup (>1GB), skipping..."
        echo "Dataset paths:" > "$BACKUP_DIR/dataset_paths_$TIMESTAMP.txt"
        find /data/datasets -type f >> "$BACKUP_DIR/dataset_paths_$TIMESTAMP.txt"
    fi
fi

# Backup configuration files
echo "Backing up configurations..."
tar -czf "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" -C / \
    app/config \
    app/core/config \
    app/infrastructure 2>/dev/null || true

# Clean old backups
echo "Cleaning old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "*.tar.gz" -o -name "*.db" -o -name "*.txt" | \
    while read -r file; do
        if [ "$(find "$file" -mtime +$RETENTION_DAYS 2>/dev/null)" ]; then
            rm -f "$file"
            echo "Removed old backup: $(basename "$file")"
        fi
    done

# Create backup manifest
echo "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest_$TIMESTAMP.txt" << EOF
MLManager Backup Manifest
Timestamp: $TIMESTAMP
Date: $(date)

Files in this backup:
$(ls -la "$BACKUP_DIR"/*_$TIMESTAMP.*)

System Info:
Disk Usage: $(df -h /data)
Memory Usage: $(free -h)
Container Status: $(docker ps --format "table {{.Names}}\t{{.Status}}")
EOF

echo "Backup completed successfully - $TIMESTAMP"
echo "Backup location: $BACKUP_DIR"
echo "Files created:"
ls -la "$BACKUP_DIR"/*_$TIMESTAMP.*
