#!/usr/bin/env python
import os
import sys
import django

# Add the project directory to Python path
sys.path.insert(0, '/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')

try:
    django.setup()
    from ml_manager.models import MLModel
    
    print("Current models in database:")
    models = MLModel.objects.all()
    print(f"Total count: {models.count()}")
    
    for model in models:
        print(f"- ID: {model.id}, Name: {model.name}, Status: {model.status}, Model Type: '{model.model_type}', MLflow Run ID: {model.mlflow_run_id}")
        
    print("\nTesting model creation:")
    test_model = MLModel.objects.create(
        name="Test Model",
        model_type="unet",
        description="Test model to verify functionality"
    )
    print(f"Created test model with ID: {test_model.id}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
