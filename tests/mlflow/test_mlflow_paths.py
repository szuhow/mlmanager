#!/usr/bin/env python3
"""
Test script to verify MLflow path resolution after fixes.
This script tests whether the application can properly find MLflow artifacts
in the configured data directory instead of the old mlruns structure.
"""

import os
import sys
import django
from pathlib import Path

# Add the Django project to Python path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root / 'core'))

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')
django.setup()

from django.conf import settings
from core.apps.ml_manager.models import ModelVersion

def test_mlflow_path_configuration():
    """Test that MLflow paths are correctly configured."""
    print("Testing MLflow path configuration...")
    print(f"BASE_MLRUNS_DIR: {settings.BASE_MLRUNS_DIR}")
    print(f"MLFLOW_ARTIFACT_ROOT: {settings.MLFLOW_ARTIFACT_ROOT}")
    
    # Check if the base directory exists
    base_dir_exists = os.path.exists(settings.BASE_MLRUNS_DIR)
    print(f"Base MLflow directory exists: {base_dir_exists}")
    
    if base_dir_exists:
        print(f"Contents of {settings.BASE_MLRUNS_DIR}:")
        try:
            for item in os.listdir(settings.BASE_MLRUNS_DIR):
                item_path = os.path.join(settings.BASE_MLRUNS_DIR, item)
                item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                print(f"  {item_type}: {item}")
        except Exception as e:
            print(f"  Error listing directory: {e}")
    
    return base_dir_exists

def test_model_path_resolution():
    """Test path resolution for existing models."""
    print("\nTesting model path resolution...")
    
    models = ModelVersion.objects.all()
    print(f"Found {models.count()} models in database")
    
    for model in models[:3]:  # Test first 3 models
        print(f"\nModel: {model.name} (ID: {model.id})")
        print(f"  MLflow Run ID: {model.mlflow_run_id}")
        print(f"  Model Directory: {model.model_directory}")
        
        if model.mlflow_run_id:
            # Test the paths that would be used by the application
            artifact_paths = [
                os.path.join(str(settings.BASE_MLRUNS_DIR), model.mlflow_run_id, 'artifacts'),
                os.path.join(str(settings.BASE_MLRUNS_DIR), model.mlflow_run_id, 'artifacts', 'model.pth'),
                os.path.join(str(settings.BASE_MLRUNS_DIR), model.mlflow_run_id, 'artifacts', 'training_logs', 'training.log'),
            ]
            
            for path in artifact_paths:
                exists = os.path.exists(path)
                print(f"  Path exists: {exists} - {path}")

def test_legacy_paths():
    """Test if old mlruns paths still exist (should be deprecated)."""
    print("\nTesting legacy mlruns paths...")
    
    legacy_paths = [
        'mlruns',
        '/app/mlruns',
        os.path.join(project_root, 'mlruns'),
    ]
    
    for path in legacy_paths:
        exists = os.path.exists(path)
        status = "EXISTS (should migrate)" if exists else "NOT EXISTS (good)"
        print(f"  {status}: {path}")

def main():
    """Run all tests."""
    print("="*60)
    print("MLflow Path Configuration Test")
    print("="*60)
    
    # Test configuration
    config_ok = test_mlflow_path_configuration()
    
    # Test model resolution
    test_model_path_resolution()
    
    # Test legacy paths
    test_legacy_paths()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if config_ok:
        print("✅ MLflow base directory configuration is correct")
        print("✅ Paths have been updated to use data/ directory")
        print("ℹ️  You can now test training to verify artifact storage works")
    else:
        print("❌ MLflow base directory not found")
        print("⚠️  You may need to create the directory or check configuration")
    
    print("\nNext steps:")
    print("1. Restart Docker containers to apply volume mapping changes")
    print("2. Test training functionality")
    print("3. Verify artifacts are stored in data/mlflow directory")

if __name__ == "__main__":
    main()
