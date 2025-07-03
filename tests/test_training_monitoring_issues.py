#!/usr/bin/env python3
"""
Test script to verify the two main issues:
1. "Start training & monitor" button should redirect to model detail page (not model list)
2. Auto-refresh should work for models transitioning from "pending" to "training"
"""

import os
import sys
import django
import time
from datetime import datetime

# Setup Django
sys.path.append('core')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from core.apps.ml_manager.models import MLModel
from django.test import Client
from django.contrib.auth.models import User
from django.urls import reverse

# Try to import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

def test_redirect_behavior():
    """Test that 'Start training & monitor' redirects to model detail, not model list"""
    print("🧪 Testing redirect behavior...")
    
    # Use existing admin user
    client = Client()
    login_success = client.login(username='admin', password='admin123')
    if not login_success:
        # Try common passwords for admin
        for password in ['admin', 'password', '123456', 'admin123']:
            if client.login(username='admin', password=password):
                login_success = True
                break
    
    if not login_success:
        print("❌ Could not login as admin user")
        return False
    print("✅ Login successful")
    
    # Test data for creating a training
    form_data = {
        'name': 'Test Redirect Model',
        'model_type': 'unet',
        'epochs': '5',
        'batch_size': '4',
        'learning_rate': '0.001',
        'data_path': '/app/data/datasets/test_coronary_dataset',
        'dataset_type': 'coronary',
        'validation_split': '0.2',
        'resolution': '256',
        'device': 'auto',  # Required field
        'optimizer': 'adam',  # Required field
        'lr_scheduler': 'plateau',  # Required field
        'num_workers': '2',  # Required field
        'loss_function': 'dice',  # Required field
        'checkpoint_strategy': 'best_val_dice',  # Required field
        'max_checkpoints': '3',  # Required field
        'monitor_metric': 'val_dice',  # Required field
        'redirect_to_list': 'false'  # This should redirect to model detail
    }
    
    print("📋 Submitting training form with redirect_to_list=false...")
    response = client.post(reverse('ml_manager:start-training'), form_data, follow=False)
    
    print(f"Response status: {response.status_code}")
    if response.status_code == 302:
        redirect_url = response.url
        print(f"Redirect URL: {redirect_url}")
        
        if '/model/' in redirect_url and '/model/' != redirect_url.split('/model/')[1]:
            print("✅ SUCCESS: Redirects to model detail page")
            return True
        elif redirect_url.endswith('/ml/'):
            print("❌ ISSUE: Redirects to model list instead of model detail")
            return False
        else:
            print(f"⚠️  Unexpected redirect: {redirect_url}")
            return False
    elif response.status_code == 200:
        # Form has errors, check what they are
        content = response.content.decode()
        if 'error' in content.lower() or 'required' in content.lower() or 'invalid' in content.lower():
            print("❌ Form has validation errors")
            # Look for specific error messages
            if BEAUTIFULSOUP_AVAILABLE:
                try:
                    soup = BeautifulSoup(content, 'html.parser')
                    errors = soup.find_all(class_=['alert-danger', 'invalid-feedback', 'error'])
                    for error in errors:
                        print(f"  Error: {error.get_text().strip()}")
                except:
                    print("  Could not parse error details")
            else:
                # Simple text search for errors
                lines = content.split('\n')
                error_lines = [line.strip() for line in lines if any(keyword in line.lower() 
                              for keyword in ['error', 'required', 'invalid', 'field is required'])]
                for line in error_lines[:5]:  # Show first 5 error lines
                    if line:
                        print(f"  Error: {line}")
        else:
            print("❌ Form rendered without redirect (unknown reason)")
        return False
    else:
        print(f"❌ Unexpected response: {response.status_code}")
        print(response.content.decode()[:500])
        return False

def test_auto_refresh_status():
    """Test auto-refresh functionality for pending models"""
    print("\n🧪 Testing auto-refresh for pending models...")
    
    # Create a test model in pending status
    test_model = MLModel.objects.create(
        name="Test Auto Refresh Model",
        status="pending",
        current_epoch=0,
        total_epochs=10,
        train_loss=0.0,
        val_loss=0.0,
        train_dice=0.0,
        val_dice=0.0,
        training_data_info={
            'model_type': 'unet',
            'loss_function': 'combined',
            'segmentation_metric': 'dice',
            'dataset_type': 'coronary',
            'data_path': '/app/data/datasets/test',
            'batch_size': 4,
            'resolution': '256',
            'learning_rate': 0.001,
            'epochs': 10,
            'validation_split': 0.2,
            'device': 'auto',
            'optimizer': 'adam'
        }
    )
    
    print(f"Created test model {test_model.id} with status: {test_model.status}")
    
    # Test the progress API endpoint
    client = Client()
    login_success = client.login(username='admin', password='admin123')
    if not login_success:
        # Try common passwords for admin
        for password in ['admin', 'password', '123456', 'admin123']:
            if client.login(username='admin', password=password):
                login_success = True
                break
    
    if not login_success:
        print("❌ Could not login as admin user")
        return False
    
    progress_url = reverse('ml_manager:model-progress', kwargs={'model_id': test_model.id})
    print(f"Testing progress API: {progress_url}")
    
    response = client.get(progress_url)
    print(f"Progress API response status: {response.status_code}")
    
    if response.status_code == 200:
        import json
        data = json.loads(response.content)
        print(f"Progress API data: {data}")
        
        if data.get('model_status') == 'pending':
            print("✅ SUCCESS: Progress API correctly returns 'pending' status")
            
            # Simulate status change to training
            test_model.status = 'training'
            test_model.current_epoch = 1
            test_model.save()
            
            response2 = client.get(progress_url)
            if response2.status_code == 200:
                data2 = json.loads(response2.content)
                if data2.get('model_status') == 'training':
                    print("✅ SUCCESS: Progress API correctly detects status change to 'training'")
                    result = True
                else:
                    print(f"❌ ISSUE: Expected 'training' status, got: {data2.get('model_status')}")
                    result = False
            else:
                print(f"❌ ISSUE: Second API call failed: {response2.status_code}")
                result = False
        else:
            print(f"❌ ISSUE: Expected 'pending' status, got: {data.get('model_status')}")
            result = False
    else:
        print(f"❌ ISSUE: Progress API failed: {response.status_code}")
        result = False
    
    # Cleanup
    test_model.delete()
    return result

def test_model_detail_page_loading():
    """Test that model detail page loads correctly and includes auto-refresh JS"""
    print("\n🧪 Testing model detail page loading...")
    
    # Create test model
    test_model = MLModel.objects.create(
        name="Test Model Detail",
        status="training",
        current_epoch=3,
        total_epochs=10,
        train_loss=0.5,
        val_loss=0.4,
        train_dice=0.75,
        val_dice=0.72,
        training_data_info={
            'model_type': 'unet',
            'loss_function': 'combined',
            'dataset_type': 'coronary',
            'data_path': '/app/data/datasets/test',
            'batch_size': 4,
            'resolution': '256',
            'learning_rate': 0.001,
            'epochs': 10,
            'validation_split': 0.2,
            'device': 'auto',
            'optimizer': 'adam'
        }
    )
    
    client = Client()
    login_success = client.login(username='admin', password='admin123')
    if not login_success:
        # Try common passwords for admin
        for password in ['admin', 'password', '123456', 'admin123']:
            if client.login(username='admin', password=password):
                login_success = True
                break
    
    if not login_success:
        print("❌ Could not login as admin user")
        return False
    
    detail_url = reverse('ml_manager:model-detail', kwargs={'pk': test_model.id})
    print(f"Testing model detail page: {detail_url}")
    
    response = client.get(detail_url)
    print(f"Model detail response status: {response.status_code}")
    
    success = False
    if response.status_code == 200:
        content = response.content.decode()
        
        # Check for key elements
        checks = [
            ('ModelDetailManager', 'new ModelDetailManager(' in content),
            ('Data Model ID', f'data-model-id="{test_model.id}"' in content),
            ('Auto-refresh JS', 'model_detail_unified.js' in content),
            ('Training progress section', 'Training Progress' in content),
            ('Live indicator', 'update-status' in content)
        ]
        
        all_passed = True
        for check_name, check_result in checks:
            if check_result:
                print(f"  ✅ {check_name}: Found")
            else:
                print(f"  ❌ {check_name}: Missing")
                all_passed = False
        
        success = all_passed
    else:
        print(f"❌ ISSUE: Model detail page failed to load: {response.status_code}")
    
    # Cleanup
    test_model.delete()
    return success

def main():
    print("🚀 Testing Training Monitoring Issues")
    print("=" * 50)
    
    results = {
        'redirect_behavior': test_redirect_behavior(),
        'auto_refresh_api': test_auto_refresh_status(),
        'model_detail_page': test_model_detail_page_loading()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed. Issues found:")
        for test_name, result in results.items():
            if not result:
                print(f"  - {test_name}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
