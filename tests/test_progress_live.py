#!/usr/bin/env python3
"""
Test script to check live progress updates during training
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'core'))

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')

import django
django.setup()

from core.apps.ml_manager.models import MLModel

def test_progress_during_training():
    """Test progress updates for training models"""
    print("=== Testing Progress Updates During Training ===")
    
    # Find currently training models
    training_models = MLModel.objects.filter(status__in=['loading', 'training'])
    
    if not training_models.exists():
        print("❌ No training models found")
        print("Available models:")
        for model in MLModel.objects.all().order_by('-created_at')[:5]:
            print(f"  - {model.name} (ID: {model.id}, Status: {model.status})")
        return
    
    for model in training_models:
        print(f"\n🔍 Testing model: {model.name} (ID: {model.id})")
        print(f"   Status: {model.status}")
        print(f"   Current epoch: {model.current_epoch}")
        print(f"   Total epochs: {model.total_epochs}")
        print(f"   Current batch: {model.current_batch}")
        print(f"   Total batches per epoch: {model.total_batches_per_epoch}")
        print(f"   Progress percentage: {model.progress_percentage}%")
        
        # Test API endpoint
        print(f"\n📡 Testing API endpoint...")
        try:
            url = f"http://localhost:8000/ml/model/{model.id}/progress/"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                print(f"   API Response: {json.dumps(data, indent=2)}")
            else:
                print(f"   ❌ API Error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ API Request failed: {e}")

def monitor_training_progress(model_id, duration=60):
    """Monitor progress for a specific model over time"""
    print(f"\n=== Monitoring Model {model_id} for {duration} seconds ===")
    
    start_time = time.time()
    prev_epoch = None
    prev_batch = None
    
    while time.time() - start_time < duration:
        try:
            model = MLModel.objects.get(id=model_id)
            
            # Check if values changed
            epoch_changed = prev_epoch != model.current_epoch
            batch_changed = prev_batch != model.current_batch
            
            if epoch_changed or batch_changed or prev_epoch is None:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Epoch: {model.current_epoch}/{model.total_epochs}, "
                      f"Batch: {model.current_batch}/{model.total_batches_per_epoch}, "
                      f"Progress: {model.progress_percentage:.1f}%")
                
                prev_epoch = model.current_epoch
                prev_batch = model.current_batch
            
            if model.status not in ['loading', 'training']:
                print(f"   Status changed to: {model.status}")
                break
                
        except MLModel.DoesNotExist:
            print(f"   Model {model_id} no longer exists")
            break
        except Exception as e:
            print(f"   Error: {e}")
            
        time.sleep(2)  # Check every 2 seconds

if __name__ == "__main__":
    # Test current progress
    test_progress_during_training()
    
    # Ask user if they want to monitor a specific model
    training_models = MLModel.objects.filter(status__in=['loading', 'training'])
    if training_models.exists():
        print(f"\n📊 Found {training_models.count()} training model(s)")
        model_id = input("Enter model ID to monitor (or press Enter to skip): ").strip()
        if model_id:
            try:
                model_id = int(model_id)
                monitor_training_progress(model_id)
            except ValueError:
                print("Invalid model ID")
