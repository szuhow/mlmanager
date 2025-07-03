#!/usr/bin/env python3
"""
Test batch deletion with proper Django authentication
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
sys.path.insert(0, '/app/core')
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from core.apps.ml_manager.models import MLModel
import uuid

def test_batch_deletion_authenticated():
    """Test batch deletion with authenticated user"""
    
    # Create test user
    try:
        user = User.objects.get(username='testuser')
    except User.DoesNotExist:
        user = User.objects.create_user('testuser', 'test@test.com', 'password')
    
    # Create test model
    test_model = MLModel.objects.create(
        name='Test Batch Delete Model',
        description='Test model for batch deletion',
        status='failed',
        mlflow_run_id=str(uuid.uuid4())
    )
    
    print(f'Created test model with ID: {test_model.id}')
    
    # Create authenticated client
    client = Client()
    client.force_login(user)
    
    # Test batch delete
    response = client.post('/ml/models/batch-delete/', {
        'model_ids': [str(test_model.id)]
    })
    
    print(f'Response status: {response.status_code}')
    print(f'Response content: {response.content.decode()[:500]}')
    
    # Check if model was deleted
    exists = MLModel.objects.filter(id=test_model.id).exists()
    print(f'Model exists after deletion: {exists}')
    
    return response.status_code == 200 and not exists

if __name__ == '__main__':
    success = test_batch_deletion_authenticated()
    print(f'Test {"PASSED" if success else "FAILED"}')
