#!/usr/bin/env python3
"""
Test script to verify architecture details display correctly.
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
from core.apps.ml_manager.views import ModelDetailView

def test_architecture_details():
    """Test that architecture details are correctly determined."""
    print("Testing architecture details for different model types...")
    
    # Test model types
    test_cases = [
        'deep_resunet_attention',
        'resunet_attention',
        'resunet',
        'deep_resunet',
        'unet',
        'monai_unet',
        'attention_unet'
    ]
    
    for model_type in test_cases:
        print(f"\n=== Testing {model_type} ===")
        
        # Create a test model
        test_model = MLModel.objects.create(
            name=f"Test {model_type}",
            description=f"Test model for {model_type}",
            status="completed",
            model_type=model_type
        )
        
        # Create a mock view instance
        view = ModelDetailView()
        view.object = test_model
        
        # Get architecture details
        arch_details = view._get_architecture_details()
        
        print(f"  Name: {arch_details.get('name', 'Unknown')}")
        print(f"  Type: {arch_details.get('type', 'Unknown')}")
        print(f"  Framework: {arch_details.get('framework', 'Unknown')}")
        
        if arch_details.get('details'):
            details = arch_details['details']
            print(f"  Architecture Family: {details.get('architecture_family', 'Unknown')}")
            if 'residual_connections' in details:
                print(f"  Residual Connections: {details['residual_connections']}")
            if 'attention_gates' in details:
                print(f"  Attention Gates: {details['attention_gates']}")
        
        # Check if model summary was generated
        if arch_details.get('model_summary'):
            print(f"  ✅ Model summary generated successfully")
            summary = arch_details['model_summary']
            print(f"  Architecture Type: {summary.get('architecture_type', 'Unknown')}")
            print(f"  Model Class: {summary.get('model_class_name', 'Unknown')}")
        else:
            print(f"  ❌ Model summary failed: {arch_details.get('model_summary_error', 'Unknown error')}")
        
        # Verify that architecture info is not "Unknown"
        if (arch_details.get('type') != 'Unknown' and 
            arch_details.get('framework') != 'Unknown'):
            print(f"  ✅ Architecture details populated correctly")
        else:
            print(f"  ❌ Architecture details still showing as Unknown")
        
        # Clean up test model
        test_model.delete()

if __name__ == "__main__":
    test_architecture_details()
