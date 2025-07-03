#!/usr/bin/env python3
"""
Test script to verify fixes for:
1. Batch deletion of models
2. Model directory creation and logging
3. Metrics display issues
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
sys.path.insert(0, 'core')
django.setup()

from apps.ml_manager.models import MLModel
from django.test import Client
from django.contrib.auth.models import User
from django.urls import reverse
import json

def test_model_creation():
    """Test that model directory paths are properly set"""
    print("Testing model creation and directory setup...")
    
    # Count existing models
    initial_count = MLModel.objects.count()
    print(f"Initial model count: {initial_count}")
    
    # Find a model with completed status but zero metrics
    problematic_models = MLModel.objects.filter(
        status='completed',
        train_dice=0.0,
        val_dice=0.0
    )
    
    print(f"Found {problematic_models.count()} completed models with zero metrics:")
    for model in problematic_models[:5]:  # Show first 5
        print(f"  Model {model.id}: {model.name}")
        print(f"    Status: {model.status}")
        print(f"    Epochs: {model.current_epoch}/{model.total_epochs}")
        print(f"    Train Dice: {model.train_dice}, Val Dice: {model.val_dice}")
        print(f"    Model Directory: {model.model_directory}")
        
        # Check if log directory exists
        if model.model_directory:
            log_path = os.path.join(model.model_directory, 'logs', 'training.log')
            print(f"    Log path exists: {os.path.exists(log_path)}")
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        print(f"    Log lines: {len(lines)}")
                except Exception as e:
                    print(f"    Error reading log: {e}")
        print()

def test_batch_deletion():
    """Test batch deletion functionality"""
    print("Testing batch deletion...")
    
    # Create test user and client
    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        user = User.objects.create_user('admin', 'admin@test.com', 'admin')
    
    client = Client()
    client.force_login(user)
    
    # Get some model IDs to delete (but don't actually delete important ones)
    test_models = MLModel.objects.filter(status__in=['failed', 'stopped'])[:2]
    
    if test_models.exists():
        model_ids = [str(model.id) for model in test_models]
        print(f"Testing deletion of models: {model_ids}")
        
        # Test the batch delete endpoint
        response = client.post(
            reverse('ml_manager:batch-delete-models'),
            {'model_ids': model_ids}
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            data = json.loads(response.content)
            print(f"Response data: {data}")
        else:
            print(f"Response content: {response.content}")
    else:
        print("No test models available for deletion test")

def check_file_system():
    """Check the file system for model directories and logs"""
    print("Checking file system...")
    
    # Check data/models structure
    models_dir = 'data/models'
    if os.path.exists(models_dir):
        print(f"Models directory exists: {models_dir}")
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        print(f"Subdirectories: {subdirs}")
        
        # Check organized directory structure
        organized_dir = os.path.join(models_dir, 'organized')
        if os.path.exists(organized_dir):
            print("Organized models directory exists")
            # List some recent directories
            for root, dirs, files in os.walk(organized_dir):
                if 'logs' in dirs:
                    logs_dir = os.path.join(root, 'logs')
                    if os.path.exists(os.path.join(logs_dir, 'training.log')):
                        print(f"Found training log: {logs_dir}/training.log")
                        
    # Check global logs
    global_log = 'data/logs/training.log'
    if os.path.exists(global_log):
        print(f"Global training log exists: {global_log}")
        try:
            with open(global_log, 'r') as f:
                lines = f.readlines()
                print(f"Global log has {len(lines)} lines")
                # Show last few lines
                print("Last 3 lines:")
                for line in lines[-3:]:
                    print(f"  {line.strip()}")
        except Exception as e:
            print(f"Error reading global log: {e}")

if __name__ == '__main__':
    print("=== ML Manager Fix Verification ===\n")
    
    test_model_creation()
    print("\n" + "="*50 + "\n")
    
    check_file_system() 
    print("\n" + "="*50 + "\n")
    
    test_batch_deletion()
    
    print("\n=== Test completed ===")
