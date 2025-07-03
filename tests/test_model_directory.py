#!/usr/bin/env python3
"""
Test script to verify model_directory is properly set during training.
"""
import os
import sys
import tempfile

# Add project paths to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'core'))
sys.path.append(os.path.join(project_root, 'ml'))

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Initialize Django
import django
django.setup()

from core.apps.ml_manager.models import MLModel
from ml.utils.utils.training_callback import TrainingCallback

def test_model_directory_logic():
    """Test that model_directory gets set correctly."""
    print("Testing model_directory setting logic...")
    
    # Create a test model
    test_model = MLModel.objects.create(
        name="Test Model Directory",
        description="Test model for directory setting",
        status="pending",
        model_type="unet"
    )
    
    print(f"Created test model {test_model.id}")
    print(f"Initial model_directory: {test_model.model_directory}")
    
    # Create a temporary directory to simulate model output
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Test model directory: {temp_dir}")
        
        # Create a TrainingCallback and set model directory
        callback = TrainingCallback(test_model.id, "test_run_id")
        
        # Test setting model directory
        callback.set_model_directory(temp_dir)
        
        # Refresh model from database
        test_model.refresh_from_db()
        
        print(f"Updated model_directory: {test_model.model_directory}")
        
        # Verify the directory was set correctly
        if test_model.model_directory == temp_dir:
            print("✅ model_directory was set correctly!")
        else:
            print("❌ model_directory was NOT set correctly!")
            
        # Test creating log directory structure
        log_dir = os.path.join(temp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training.log')
        
        # Write some test log content
        with open(log_file, 'w') as f:
            f.write("Test training log content\n")
            f.write("Epoch 1/10 - Loss: 0.5\n")
            f.write("Training completed successfully\n")
        
        print(f"Created test log file: {log_file}")
        
        # Verify Django can find the log file
        expected_log_path = os.path.join(test_model.model_directory, 'logs', 'training.log')
        if os.path.exists(expected_log_path):
            print("✅ Django can access the log file!")
            print("Log content preview:")
            with open(expected_log_path, 'r') as f:
                print(f.read())
        else:
            print("❌ Django cannot access the log file!")
    
    # Clean up test model
    test_model.delete()
    print("Test model cleaned up")

if __name__ == "__main__":
    test_model_directory_logic()
