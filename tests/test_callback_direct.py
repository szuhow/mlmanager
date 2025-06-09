#!/usr/bin/env python3
"""
Direct test of the callback system to see if it's working
"""

import os
import sys
import django
import time

# Setup Django
sys.path.append('/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel
from shared.utils.training_callback import TrainingCallback

def test_callback_direct():
    """Test the callback system directly"""
    print("Testing callback system directly...")
    
    # Create a test model
    test_model = MLModel.objects.create(
        name="Callback Test Model",
        description="Test callback functionality",
        status="pending"
    )
    
    print(f"Created test model with ID: {test_model.id}")
    print(f"Initial status: {test_model.status}")
    
    # Create callback
    callback = TrainingCallback(test_model.id, "test-run-callback")
    
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
    
    # Cleanup
    test_model.delete()
    print("Test completed and cleaned up")

if __name__ == "__main__":
    test_callback_direct()
