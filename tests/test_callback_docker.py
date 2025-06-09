#!/usr/bin/env python3
"""
Direct test of the callback system to see if it's working in Docker
"""

import os
import sys
import django
import time

# Setup Django for Docker environment
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
from shared.utils.training_callback import TrainingCallback

def test_callback_direct():
    """Test the callback system directly"""
    print("Testing callback system directly in Docker...")
    
    # Create a test model
    test_model = MLModel.objects.create(
        name="Callback Test Model Docker",
        description="Test callback functionality in Docker",
        status="pending"
    )
    
    print(f"Created test model with ID: {test_model.id}")
    print(f"Initial status: {test_model.status}")
    
    # Create callback
    callback = TrainingCallback(test_model.id, "test-run-callback-docker")
    
    # Test on_training_start
    print("Calling on_training_start()...")
    callback.on_training_start()
    
    # Refresh and check status
    test_model.refresh_from_db()
    print(f"Status after on_training_start(): {test_model.status}")
    
    # Test on_dataset_loaded
    print("Calling on_dataset_loaded()...")
    callback.on_dataset_loaded()
    
    # Refresh and check status
    test_model.refresh_from_db()
    print(f"Status after on_dataset_loaded(): {test_model.status}")
    
    # Test some epoch progress
    print("Testing epoch progress...")
    callback.on_epoch_end(0, {'train_loss': 0.8, 'val_loss': 0.75, 'train_dice': 0.85, 'val_dice': 0.80})
    
    # Refresh and check
    test_model.refresh_from_db()
    print(f"Status after epoch update: {test_model.status}")
    # Note: training_progress field may not exist in this model
    
    # Cleanup
    test_model.delete()
    print("Test completed and cleaned up")

if __name__ == "__main__":
    test_callback_direct()
