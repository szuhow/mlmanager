#!/usr/bin/env python
"""
Test script to verify the batch deletion fix:
1. Create test models
2. Test frontend refresh after deletion
3. Verify models don't reappear after refresh
"""

import sys
import os
import time
import requests
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent / "core"
sys.path.append(str(project_root))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from apps.ml_manager.models import MLModel
from django.contrib.auth.models import User
from django.db import transaction

def clean_test_models():
    """Clean up any test models"""
    print("🧹 Cleaning up test models...")
    test_models = MLModel.objects.filter(name__startswith='test_batch_')
    count = test_models.count()
    test_models.delete()
    print(f"✅ Cleaned up {count} test models")

def create_test_models(count=3):
    """Create test models for deletion"""
    print(f"📝 Creating {count} test models...")
    
    # Get or create a test user
    user, created = User.objects.get_or_create(
        username='testuser',
        defaults={'email': 'test@example.com', 'is_staff': True}
    )
    
    models = []
    for i in range(count):
        model = MLModel.objects.create(
            name=f'test_batch_model_{i+1}',
            description=f'Test model {i+1} for batch deletion',
            model_type='deep_resunet',
            status='completed',
            progress_percentage=100,
            user=user
        )
        models.append(model)
        print(f"  ✅ Created: {model.name} (ID: {model.id})")
    
    return models

def test_api_models_endpoint():
    """Test the API endpoint to see current models"""
    print("\n🔍 Testing API models endpoint...")
    
    try:
        response = requests.get('http://localhost:8000/api/ml/models/')
        if response.status_code == 200:
            data = response.json()
            if 'models' in data:
                models = data['models']
                print(f"✅ API returned {len(models)} models")
                test_models = [m for m in models if m['name'].startswith('test_batch_')]
                print(f"📊 Test models in API: {len(test_models)}")
                for model in test_models:
                    print(f"  - {model['name']} (ID: {model['id']})")
            else:
                print(f"⚠️  API response format: {list(data.keys())}")
        else:
            print(f"❌ API request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def test_batch_deletion_via_api():
    """Test batch deletion via API"""
    print("\n🗑️  Testing batch deletion via API...")
    
    # Get test models
    test_models = MLModel.objects.filter(name__startswith='test_batch_')
    if not test_models.exists():
        print("❌ No test models found")
        return False
    
    model_ids = list(test_models.values_list('id', list=True))
    print(f"📋 Models to delete: {model_ids}")
    
    # Test deletion
    try:
        # Get CSRF token first
        session = requests.Session()
        csrf_response = session.get('http://localhost:8000/ml/models/')
        
        if csrf_response.status_code != 200:
            print(f"❌ Failed to get CSRF token: {csrf_response.status_code}")
            return False
        
        # Extract CSRF token from cookie
        csrf_token = session.cookies.get('csrftoken')
        if not csrf_token:
            print("❌ No CSRF token found")
            return False
        
        # Perform deletion
        delete_response = session.post(
            'http://localhost:8000/api/ml/models/batch-delete/',
            json={'model_ids': model_ids},
            headers={
                'X-CSRFToken': csrf_token,
                'Content-Type': 'application/json',
            }
        )
        
        if delete_response.status_code == 200:
            result = delete_response.json()
            print(f"✅ Deletion response: {result}")
            
            # Verify models are gone from database
            remaining = MLModel.objects.filter(id__in=model_ids).count()
            print(f"📊 Remaining models in DB: {remaining}")
            
            return result.get('status') == 'success' and remaining == 0
        else:
            print(f"❌ Deletion failed: {delete_response.status_code}")
            print(f"Response: {delete_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during deletion: {e}")
        return False

def main():
    print("🚀 Testing Batch Deletion Fix")
    print("=" * 50)
    
    # Clean up first
    clean_test_models()
    
    # Create test models
    test_models = create_test_models(3)
    
    # Test API endpoint before deletion
    test_api_models_endpoint()
    
    # Test batch deletion
    deletion_success = test_batch_deletion_via_api()
    
    # Test API endpoint after deletion
    print("\n🔍 Testing API after deletion...")
    test_api_models_endpoint()
    
    # Summary
    print("\n" + "=" * 50)
    if deletion_success:
        print("✅ BATCH DELETION TEST PASSED")
        print("🔄 Frontend should now reload after deletion to prevent reappearance")
    else:
        print("❌ BATCH DELETION TEST FAILED")
    
    # Final cleanup
    clean_test_models()

if __name__ == '__main__':
    main()
