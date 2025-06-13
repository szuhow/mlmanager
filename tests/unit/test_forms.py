#!/usr/bin/env python
"""
Test script to debug form choices issues
"""

import os
import sys
import django
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'ml_manager'))
sys.path.insert(0, str(project_root / 'shared'))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def test_device_choices():
    """Test device choices"""
    print("=== Testing Device Choices ===")
    try:
        from ml_manager.device_utils import get_device_choices, get_default_device, detect_cuda_availability
        
        cuda_available, device_info = detect_cuda_availability()
        print(f"CUDA Available: {cuda_available}")
        print(f"Device Info: {device_info}")
        
        device_choices = get_device_choices()
        print(f"Device Choices: {device_choices}")
        
        default_device = get_default_device()
        print(f"Default Device: {default_device}")
        
    except Exception as e:
        print(f"Device choices test failed: {e}")
        import traceback
        traceback.print_exc()

def test_model_choices():
    """Test model architecture choices"""
    print("\n=== Testing Model Choices ===")
    try:
        # Try the primary method
        try:
            from shared.architecture_registry import get_available_models
            models = get_available_models()
            print(f"Available models from registry: {models}")
        except Exception as e:
            print(f"Registry method failed: {e}")
            
            # Try fallback method
            from ml_manager.forms import get_available_models as fallback_models
            models = fallback_models()
            print(f"Available models from fallback: {models}")
            
    except Exception as e:
        print(f"Model choices test failed: {e}")
        import traceback
        traceback.print_exc()

def test_form_instantiation():
    """Test form instantiation"""
    print("\n=== Testing Form Instantiation ===")
    try:
        from ml_manager.forms import TrainingForm, TrainingTemplateForm
        
        # Test TrainingForm
        print("Creating TrainingForm...")
        training_form = TrainingForm()
        print(f"TrainingForm model_type choices: {training_form.fields['model_type'].choices}")
        print(f"TrainingForm device choices: {training_form.fields['device'].choices}")
        
        # Test TrainingTemplateForm  
        print("Creating TrainingTemplateForm...")
        template_form = TrainingTemplateForm()
        print(f"TrainingTemplateForm model_type choices: {template_form.fields['model_type'].choices}")
        print(f"TrainingTemplateForm device choices: {template_form.fields['device'].choices}")
        
    except Exception as e:
        print(f"Form instantiation test failed: {e}")
        import traceback
        traceback.print_exc()

def test_form_validation():
    """Test form validation with common values"""
    print("\n=== Testing Form Validation ===")
    try:
        from ml_manager.forms import TrainingForm
        
        # Test with unet and auto device
        form_data = {
            'name': 'Test Training',
            'model_type': 'unet',
            'data_path': '/test/path',
            'device': 'auto',
            'resolution': '256',
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'crop_size': 128,
            'num_workers': 4,
        }
        
        form = TrainingForm(data=form_data)
        print(f"Form is valid: {form.is_valid()}")
        if not form.is_valid():
            print(f"Form errors: {form.errors}")
        
        # Test with alternative models
        print("\nTesting with monai_unet:")
        form_data['model_type'] = 'monai_unet'
        form2 = TrainingForm(data=form_data)
        print(f"Form is valid: {form2.is_valid()}")
        if not form2.is_valid():
            print(f"Form errors: {form2.errors}")
            
    except Exception as e:
        print(f"Form validation test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_device_choices()
    test_model_choices()
    test_form_instantiation()
    test_form_validation()
