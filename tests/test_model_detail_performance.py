#!/usr/bin/env python3
"""
Test performance of model detail view operations
"""

import time
import os
import sys
import django

# Set up Django
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'core'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from core.apps.ml_manager.models import MLModel
from core.apps.ml_manager.views import ModelDetailView
from django.test import RequestFactory
from django.contrib.auth.models import User


def time_operation(name, func):
    """Time a function execution"""
    start = time.time()
    try:
        result = func()
        end = time.time()
        print(f"✅ {name}: {end - start:.3f}s")
        return result
    except Exception as e:
        end = time.time()
        print(f"❌ {name}: {end - start:.3f}s (ERROR: {e})")
        return None


def test_model_detail_performance():
    """Test the performance of different parts of model detail view"""
    
    # Get a model to test
    model = MLModel.objects.order_by('-created_at').first()
    if not model:
        print("No models found in database")
        return
    
    print(f"Testing performance for model: {model.name} (ID: {model.id})")
    print("=" * 60)
    
    # Create view instance
    factory = RequestFactory()
    request = factory.get(f'/ml/models/{model.id}/')
    user, created = User.objects.get_or_create(username='testuser')
    request.user = user
    
    view = ModelDetailView()
    view.setup(request)
    view.object = model
    
    # Test individual operations
    total_start = time.time()
    
    # Test 1: Basic model loading
    time_operation("Model loading", lambda: MLModel.objects.get(id=model.id))
    
    # Test 2: MLflow connection
    def test_mlflow():
        import mlflow
        if model.mlflow_run_id:
            client = mlflow.tracking.MlflowClient()
            return client.get_run(model.mlflow_run_id)
        return None
    
    time_operation("MLflow run fetch", test_mlflow)
    
    # Test 3: Training details
    time_operation("Training details", lambda: view._get_training_details())
    
    # Test 4: Training logs
    time_operation("Training logs", lambda: view._get_training_logs())
    
    # Test 5: Architecture details
    time_operation("Architecture details", lambda: view._get_architecture_details())
    
    # Test 6: Full context data
    time_operation("Full context data", lambda: view.get_context_data())
    
    total_end = time.time()
    print("=" * 60)
    print(f"🏁 TOTAL TIME: {total_end - total_start:.3f}s")


def test_directory_walk_performance():
    """Test the performance of directory walking in _get_training_logs"""
    print("\nTesting directory walk performance...")
    print("=" * 60)
    
    # Test walking organized directory
    organized_base = os.path.join("data", "models", "organized")
    
    def count_directories():
        count = 0
        if os.path.exists(organized_base):
            for root, dirs, files in os.walk(organized_base):
                count += len(dirs)
        return count
    
    def count_files():
        count = 0
        if os.path.exists(organized_base):
            for root, dirs, files in os.walk(organized_base):
                count += len(files)
        return count
    
    dir_count = time_operation("Directory counting", count_directories)
    file_count = time_operation("File counting", count_files)
    
    print(f"📁 Total directories: {dir_count}")
    print(f"📄 Total files: {file_count}")


if __name__ == "__main__":
    print("Testing Model Detail View Performance")
    print("=" * 60)
    
    test_model_detail_performance()
    test_directory_walk_performance()
