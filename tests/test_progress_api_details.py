#!/usr/bin/env python3
"""
Test detailed progress API and model display for debugging Progress/Performance columns
"""

import os
import sys
import django
import requests
import json
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core'))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.apps.ml_manager.models import MLModel

def test_model_properties():
    """Test model properties for progress and performance display"""
    print("=== Testing Model Properties ===")
    
    models = MLModel.objects.all().order_by('-created_at')[:5]
    
    for model in models:
        print(f"\nModel: {model.name} (ID: {model.id})")
        print(f"  Status: {model.status}")
        print(f"  Current Epoch: {model.current_epoch}")
        print(f"  Total Epochs: {model.total_epochs}")
        print(f"  Current Batch: {model.current_batch}")
        print(f"  Total Batches per Epoch: {model.total_batches_per_epoch}")
        print(f"  Progress Percentage: {model.progress_percentage:.2f}%")
        print(f"  Batch Progress Percentage: {model.batch_progress_percentage:.2f}%")
        print(f"  Best Val Dice: {model.best_val_dice}")
        print(f"  Best Val IoU: {model.best_val_iou}")
        print(f"  Performance Metrics: {getattr(model, 'performance_metrics', 'Not set')}")
        print(f"  Training Data Info: {getattr(model, 'training_data_info', 'Not set')}")

def test_progress_api():
    """Test the progress API endpoints"""
    print("\n=== Testing Progress API ===")
    
    # Get some models
    models = MLModel.objects.all().order_by('-created_at')[:3]
    
    for model in models:
        print(f"\nTesting API for Model: {model.name} (ID: {model.id})")
        
        try:
            # Test progress API endpoint
            url = f"http://localhost:8000/ml/model/{model.id}/progress/"
            print(f"  Calling: {url}")
            
            response = requests.get(url)
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  API Response:")
                print(f"    Status: {data.get('status')}")
                print(f"    Model Status: {data.get('model_status')}")
                print(f"    Progress: {data.get('progress', {})}")
                print(f"    Metrics: {data.get('metrics', {})}")
            else:
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"  Exception: {e}")

def test_model_list_context():
    """Test what data is passed to model list template"""
    print("\n=== Testing Model List Context ===")
    
    # Import the view
    from core.apps.ml_manager.views import ModelListView
    from django.test import RequestFactory
    from django.contrib.auth.models import User
    
    # Create a fake request
    factory = RequestFactory()
    request = factory.get('/ml/models/')
    
    # Create or get a user for the request
    user, created = User.objects.get_or_create(username='testuser')
    request.user = user
    
    # Create view instance and get context
    view = ModelListView()
    view.setup(request)
    
    queryset = view.get_queryset()
    context = view.get_context_data()
    
    print(f"Number of models in queryset: {queryset.count()}")
    
    # Check first few models
    for model in queryset[:3]:
        print(f"\nModel in context: {model.name} (ID: {model.id})")
        print(f"  Status: {model.status}")
        print(f"  Progress Percentage: {model.progress_percentage:.2f}%")
        print(f"  Best Val Dice: {model.best_val_dice}")
        print(f"  Best Val IoU: {model.best_val_iou}")
        
        # Check template conditions
        has_performance = model.best_val_dice > 0 or model.best_val_iou > 0
        print(f"  Shows Performance Badge: {has_performance}")
        
        # Check classification metrics
        if hasattr(model, 'training_data_info') and model.training_data_info:
            dataset_type = model.training_data_info.get('dataset_type', 'segmentation')
            print(f"  Dataset Type: {dataset_type}")
            
            if dataset_type == 'arcade_classification':
                val_accuracy = model.performance_metrics.get('val_accuracy', 0) if hasattr(model, 'performance_metrics') and model.performance_metrics else 0
                print(f"  Val Accuracy: {val_accuracy}")

def test_template_rendering():
    """Test how templates would render the data"""
    print("\n=== Testing Template Logic ===")
    
    models = MLModel.objects.all().order_by('-created_at')[:5]
    
    for model in models:
        print(f"\nModel: {model.name} (ID: {model.id})")
        print(f"  Status: {model.status}")
        
        # Test Progress column logic
        if model.status in ['training', 'loading']:
            if model.status == 'loading':
                progress_display = "Loading..."
            else:
                progress_display = f"{model.current_epoch}/{model.total_epochs} ({model.progress_percentage:.0f}%)"
        elif model.status == 'pending':
            progress_display = "Preparing..."
        else:
            progress_display = f"{model.current_epoch}/{model.total_epochs}"
        
        print(f"  Progress Display: {progress_display}")
        
        # Test Performance column logic
        training_data_info = getattr(model, 'training_data_info', {}) or {}
        dataset_type = training_data_info.get('dataset_type', 'segmentation')
        
        if dataset_type == 'arcade_classification':
            performance_metrics = getattr(model, 'performance_metrics', {}) or {}
            val_accuracy = performance_metrics.get('val_accuracy', 0)
            if val_accuracy > 0:
                performance_display = f"Accuracy: {val_accuracy:.3f}"
            else:
                performance_display = "-"
        else:
            if model.best_val_dice > 0 or model.best_val_iou > 0:
                if model.best_val_iou > 0 and model.best_val_iou >= model.best_val_dice:
                    performance_display = f"IoU: {model.best_val_iou:.3f}"
                else:
                    performance_display = f"Dice: {model.best_val_dice:.3f}"
            else:
                performance_display = "-"
        
        print(f"  Performance Display: {performance_display}")

if __name__ == "__main__":
    print("Testing Progress and Performance Display Issues")
    print("=" * 60)
    
    test_model_properties()
    test_progress_api()
    test_model_list_context()
    test_template_rendering()
    
    print("\n" + "=" * 60)
    print("Test completed!")
