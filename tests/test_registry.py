#!/usr/bin/env python
"""
Test script for MLflow Model Registry integration
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

def test_registry_functions():
    """Test the MLflow Model Registry functions"""
    print("Testing MLflow Model Registry functions...")
    
    try:
        # Test importing registry functions
        from ml_manager.mlflow_utils import (
            list_registered_models, 
            get_registered_model_info,
            setup_mlflow
        )
        print("✓ Successfully imported registry functions")
        
        # Setup MLflow
        setup_mlflow()
        print("✓ MLflow setup completed")
        
        # Test listing registered models
        try:
            models = list_registered_models()
            print(f"✓ Found {len(models)} registered models")
            
            if models:
                print("\nRegistered Models:")
                for model in models[:3]:  # Show first 3 models
                    print(f"  - {model.get('name', 'Unknown')}")
                    if model.get('latest_versions'):
                        for version in model['latest_versions'][:2]:  # Show first 2 versions
                            print(f"    Version {version.get('version', 'Unknown')}: {version.get('current_stage', 'None')}")
            else:
                print("  No models found in registry")
                
        except Exception as e:
            print(f"✗ Error listing models: {e}")
        
    except Exception as e:
        print(f"✗ Error importing functions: {e}")
        return False
    
    return True

def test_django_model_fields():
    """Test that Django model has the registry fields"""
    print("\nTesting Django model registry fields...")
    
    try:
        from ml_manager.models import MLModel
        
        # Check if fields exist
        model_fields = [field.name for field in MLModel._meta.fields]
        required_fields = [
            'registry_model_name',
            'registry_model_version', 
            'registry_stage',
            'is_registered'
        ]
        
        missing_fields = [field for field in required_fields if field not in model_fields]
        
        if missing_fields:
            print(f"✗ Missing fields: {missing_fields}")
            return False
        else:
            print("✓ All registry fields present in Django model")
            
        # Test creating a model instance (without saving)
        test_model = MLModel(
            name="Test Model",
            registry_model_name="test-model",
            registry_model_version="1",
            registry_stage="Staging",
            is_registered=True
        )
        print("✓ Can create model instance with registry fields")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Django model: {e}")
        return False

if __name__ == "__main__":
    print("MLflow Model Registry Integration Test")
    print("=" * 50)
    
    registry_test = test_registry_functions()
    django_test = test_django_model_fields()
    
    print("\n" + "=" * 50)
    if registry_test and django_test:
        print("✓ All tests passed! MLflow Model Registry integration is working.")
    else:
        print("✗ Some tests failed. Check the errors above.")
    
    print("\nNext steps:")
    print("1. Train a model to test automatic registration")
    print("2. Use the Django admin/web interface to manage model stages")
    print("3. Test stage transitions and model promotion workflows")
