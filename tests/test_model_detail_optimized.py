#!/usr/bin/env python3
"""
Test model detail view performance with and without model summary
"""

import os
import sys
import time
import requests
from pathlib import Path

# Setup Django
project_root = Path(__file__).parent / "core"
sys.path.insert(0, str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from apps.ml_manager.models import MLModel

def test_model_detail_performance():
    """Test model detail view performance"""
    base_url = "http://localhost:8000"
    
    # Get a model
    model = MLModel.objects.first()
    if not model:
        print("❌ No models found in database")
        return
    
    print(f"Testing model detail view for model: {model.name}")
    print(f"Model ID: {model.id}")
    print("-" * 50)
    
    # Test without model summary (default)
    print("1. Testing WITHOUT model summary (default):")
    start_time = time.time()
    response = requests.get(f"{base_url}/ml_manager/models/{model.id}/")
    load_time = time.time() - start_time
    
    if response.status_code == 200:
        print(f"   ✅ SUCCESS: {load_time:.3f}s")
    else:
        print(f"   ❌ FAILED: Status {response.status_code}")
    
    # Test with model summary explicitly requested
    print("\n2. Testing WITH model summary (include_summary=true):")
    start_time = time.time()
    response = requests.get(f"{base_url}/ml_manager/models/{model.id}/?include_summary=true")
    load_time = time.time() - start_time
    
    if response.status_code == 200:
        print(f"   ✅ SUCCESS: {load_time:.3f}s")
    else:
        print(f"   ❌ FAILED: Status {response.status_code}")
    
    # Test multiple calls without summary (should be fast due to caching)
    print("\n3. Testing multiple calls without summary (caching test):")
    times = []
    for i in range(3):
        start_time = time.time()
        response = requests.get(f"{base_url}/ml_manager/models/{model.id}/")
        load_time = time.time() - start_time
        times.append(load_time)
        
        if response.status_code == 200:
            print(f"   Call {i+1}: ✅ {load_time:.3f}s")
        else:
            print(f"   Call {i+1}: ❌ Status {response.status_code}")
    
    avg_time = sum(times) / len(times)
    print(f"   Average: {avg_time:.3f}s")

if __name__ == "__main__":
    test_model_detail_performance()
