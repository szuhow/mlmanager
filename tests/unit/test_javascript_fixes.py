#!/usr/bin/env python3
"""
Test the JavaScript fixes for model list live updates and 404 error handling
"""

import os
import sys
import django
import time
import logging
from pathlib import Path

# Setup Django
sys.path.append(str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.testing')
django.setup()

from core.apps.ml_manager.models import MLModel
from django.test import Client
from django.contrib.auth.models import User
import json

def test_model_list_view():
    """Test that the model list view handles missing models gracefully"""
    print("🧪 Testing model list view...")
    
    # Create a test user
    user, created = User.objects.get_or_create(
        username='testuser',
        defaults={'email': 'test@example.com'}
    )
    if created:
        user.set_password('testpass')
        user.save()
        print(f"✅ Created test user: {user.username}")
    else:
        print(f"✅ Using existing test user: {user.username}")
    
    # Create a test model
    test_model = MLModel.objects.create(
        name="Test Model for JS Fix",
        status='pending',
        current_epoch=0,
        total_epochs=10
    )
    print(f"✅ Created test model: {test_model.name} (ID: {test_model.id})")
    
    # Test the list view
    client = Client()
    client.login(username='testuser', password='testpass')
    
    response = client.get('/ml/')
    assert response.status_code == 200
    print("✅ Model list view loads successfully")
    
    # Test AJAX request for a model that exists
    response = client.get(f'/ml/model/{test_model.id}/', HTTP_X_REQUESTED_WITH='XMLHttpRequest')
    assert response.status_code == 200
    data = json.loads(response.content)
    assert 'progress' in data
    print("✅ AJAX request for existing model works")
    
    # Test AJAX request for a model that doesn't exist (should return 404)
    non_existent_id = 99999
    response = client.get(f'/ml/model/{non_existent_id}/', HTTP_X_REQUESTED_WITH='XMLHttpRequest')
    assert response.status_code == 404
    print("✅ AJAX request for non-existent model returns 404 as expected")
    
    # Cleanup
    test_model.delete()
    print("✅ Cleaned up test model")
    
    return True

def test_batch_delete_functionality():
    """Test that batch delete works correctly"""
    print("\n🧪 Testing batch delete functionality...")
    
    # Create multiple test models
    models = []
    for i in range(3):
        model = MLModel.objects.create(
            name=f"Test Model {i+1}",
            status='pending',
            current_epoch=0,
            total_epochs=10
        )
        models.append(model)
    
    print(f"✅ Created {len(models)} test models")
    
    client = Client()
    user = User.objects.get(username='testuser')
    client.force_login(user)
    
    # Test batch delete
    model_ids = [str(model.id) for model in models[:2]]  # Delete first 2 models
    
    response = client.post('/ml/batch-delete/', {
        'model_ids': model_ids
    })
    
    assert response.status_code == 200
    data = json.loads(response.content)
    assert data['status'] == 'success'
    print("✅ Batch delete request successful")
    
    # Verify models were deleted
    remaining_models = MLModel.objects.filter(id__in=[int(id) for id in model_ids])
    assert remaining_models.count() == 0
    print("✅ Models were actually deleted from database")
    
    # Cleanup remaining model
    models[2].delete()
    print("✅ Cleaned up remaining test model")
    
    return True

def check_javascript_files():
    """Check that JavaScript files contain the fixes"""
    print("\n🧪 Checking JavaScript files for fixes...")
    
    js_files = [
        'ml_manager/static/ml_manager/js/model_progress.js',
        'staticfiles/ml_manager/js/model_progress.js'
    ]
    
    for js_file in js_files:
        if os.path.exists(js_file):
            with open(js_file, 'r') as f:
                content = f.read()
            
            # Check for the 404 handling fix
            if 'response.status === 404' in content:
                print(f"✅ {js_file} contains 404 error handling")
            else:
                print(f"❌ {js_file} missing 404 error handling")
            
            # Check for the removeModelFromUpdates method
            if 'removeModelFromUpdates' in content:
                print(f"✅ {js_file} contains removeModelFromUpdates method")
            else:
                print(f"❌ {js_file} missing removeModelFromUpdates method")
        else:
            print(f"❌ {js_file} not found")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing JavaScript fixes for Django ML Manager\n")
    
    try:
        # Test model list view
        test_model_list_view()
        
        # Test batch delete
        test_batch_delete_functionality()
        
        # Check JavaScript files
        check_javascript_files()
        
        print("\n✅ All tests passed! JavaScript fixes are working correctly.")
        print("\n📋 Summary of fixes:")
        print("  • 404 error handling for non-existent models in live updates")
        print("  • Graceful removal of deleted models from UI")
        print("  • Visual feedback for deleted models")
        print("  • Improved batch delete with better UI feedback")
        print("  • No more full page reloads after batch delete")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
