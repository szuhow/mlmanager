#!/usr/bin/env python3
"""
Debug device detection in ModelDetailView
"""

import os
import sys
import django

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

from core.apps.ml_manager.models import MLModel
from ml_manager.views import ModelDetailView

def test_device_detection():
    """Test device detection on actual models"""
    
    print("=== DEBUGGING DEVICE DETECTION ===")
    
    # Get the latest model
    models = MLModel.objects.all().order_by('-id')
    if not models:
        print("No models found in database")
        return
    
    model = models.first()
    print(f"Testing with model ID: {model.id}")
    print(f"Model status: {model.status}")
    print(f"Model directory: {model.model_directory}")
    print(f"MLflow run ID: {model.mlflow_run_id}")
    
    # Create a ModelDetailView instance
    view = ModelDetailView()
    view.object = model
    
    # Test _get_training_logs method
    print("\n--- Testing _get_training_logs() ---")
    logs = view._get_training_logs()
    print(f"Number of log lines: {len(logs)}")
    
    if logs:
        print("First 5 log lines:")
        for i, line in enumerate(logs[:5]):
            print(f"  {i+1}: {line}")
        
        print("\nLast 5 log lines:")
        for i, line in enumerate(logs[-5:]):
            print(f"  {len(logs)-4+i}: {line}")
            
        # Look for device-related lines
        print("\n--- Device-related log lines ---")
        device_lines = []
        for i, line in enumerate(logs):
            if 'device' in line.lower() or 'cuda' in line.lower() or 'cpu' in line.lower():
                device_lines.append((i+1, line))
        
        if device_lines:
            for line_num, line in device_lines:
                print(f"  Line {line_num}: {line}")
        else:
            print("  No device-related lines found")
    else:
        print("No logs found!")
    
    # Test _extract_runtime_device_from_logs method
    print("\n--- Testing _extract_runtime_device_from_logs() ---")
    runtime_device = view._extract_runtime_device_from_logs()
    print(f"Extracted runtime device: {runtime_device}")
    
    # Test _get_training_details method
    print("\n--- Testing _get_training_details() ---")
    try:
        training_details = view._get_training_details()
        if training_details and 'hardware' in training_details:
            hardware = training_details['hardware']
            print(f"Hardware device: {hardware.get('device', 'N/A')}")
            print(f"Config device: {hardware.get('config_device', 'N/A')}")
        else:
            print("No training details found")
    except Exception as e:
        print(f"Error getting training details: {e}")
    
    # Check actual file paths
    print("\n--- Checking file paths ---")
    
    # Check model-specific log path
    if model.model_directory:
        model_log_path = os.path.join(model.model_directory, 'logs', 'training.log')
        print(f"Model log path: {model_log_path}")
        print(f"  Exists: {os.path.exists(model_log_path)}")
        if os.path.exists(model_log_path):
            with open(model_log_path, 'r') as f:
                content = f.read()
                print(f"  File size: {len(content)} characters")
                if 'Device:' in content:
                    print("  ✓ Contains 'Device:' pattern")
                else:
                    print("  ✗ Does not contain 'Device:' pattern")
    
    # Check global log path
    global_log_path = os.path.join('models', 'artifacts', 'training.log')
    print(f"Global log path: {global_log_path}")
    print(f"  Exists: {os.path.exists(global_log_path)}")
    if os.path.exists(global_log_path):
        with open(global_log_path, 'r') as f:
            content = f.read()
            print(f"  File size: {len(content)} characters")
            if 'Device:' in content:
                print("  ✓ Contains 'Device:' pattern")
            else:
                print("  ✗ Does not contain 'Device:' pattern")

if __name__ == "__main__":
    test_device_detection()
