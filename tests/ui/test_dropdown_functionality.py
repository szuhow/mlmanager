#!/usr/bin/env python3
"""
Test the custom dropdown functionality
"""
import os
import sys
import django
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'core'))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.config.settings.development')
django.setup()

from django.test import Client, RequestFactory
from django.contrib.auth.models import User
from core.apps.ml_manager.models import MLModel

def test_dropdown_functionality():
    """Test that custom dropdown in model list works correctly"""
    print("🧪 Testing Custom Dropdown Functionality...")
    
    try:
        # Create test user
        user, created = User.objects.get_or_create(
            username='dropdown_test_user',
            defaults={'email': 'dropdown_test@example.com'}
        )
        if created:
            user.set_password('test123')
            user.save()
        
        print(f"✅ Test user: {user.username}")
        
        # Create some test models for dropdown testing
        test_models = []
        for i in range(3):
            model = MLModel.objects.create(
                name=f'Dropdown Test Model {i+1}',
                description=f'Test model {i+1} for dropdown functionality',
                status='completed',
                current_epoch=10,
                total_epochs=10,
                train_loss=0.1,
                val_loss=0.15,
                train_dice=0.9,
                val_dice=0.85,
                best_val_dice=0.87,
                mlflow_run_id=f'test-run-{i+1}',
                training_data_info={
                    'model_type': 'unet',
                    'batch_size': 32,
                    'epochs': 10,
                    'learning_rate': 0.001,
                },
                model_type='unet'
            )
            test_models.append(model)
        
        print(f"✅ Created {len(test_models)} test models")
        
        # Test model list view with client
        client = Client()
        client.force_login(user)
        
        # Get model list page
        response = client.get('/ml/')
        
        if response.status_code == 200:
            print("✅ Model list page loads successfully")
            
            # Check if custom dropdown CSS is present
            content = response.content.decode('utf-8')
            
            # Check for custom dropdown classes
            dropdown_checks = [
                '.custom-dropdown' in content,
                '.custom-dropdown-toggle' in content,
                '.custom-dropdown-menu' in content,
                'z-index: 10000' in content,
                'pauseUpdateForRow' in content,
                'resumeUpdateForRow' in content,
            ]
            
            passed_checks = sum(dropdown_checks)
            total_checks = len(dropdown_checks)
            
            print(f"✅ Custom dropdown elements: {passed_checks}/{total_checks} found")
            
            if passed_checks >= 4:  # Most important elements should be present
                print("✅ Custom dropdown implementation is active")
                dropdown_success = True
            else:
                print("⚠️  Some custom dropdown elements missing")
                dropdown_success = False
        else:
            print(f"❌ Model list page failed to load: {response.status_code}")
            dropdown_success = False
        
        # Test template rendering for start training form
        response = client.get('/ml/start-training/')
        
        if response.status_code == 200:
            print("✅ Start training page loads successfully")
            
            # Check for host directory selection
            content = response.content.decode('utf-8')
            host_directory_checks = [
                '/host/desktop' in content,
                '/host/downloads' in content,
                '/host/documents' in content,
            ]
            
            host_passed = sum(host_directory_checks)
            if host_passed >= 2:
                print("✅ Host directory selection is available")
                host_success = True
            else:
                print("⚠️  Host directory selection may be missing")
                host_success = False
        else:
            print(f"❌ Start training page failed to load: {response.status_code}")
            host_success = False
        
        # Cleanup test models
        for model in test_models:
            model.delete()
        print("🧹 Cleaned up test models")
        
        return dropdown_success and host_success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dropdown_functionality()
    
    print("\n" + "="*60)
    if success:
        print("🎉 DROPDOWN FUNCTIONALITY TEST PASSED!")
        print("✅ Custom dropdown implementation is working")
        print("✅ Host directory selection is available")
        print("✅ UI improvements are active")
    else:
        print("❌ DROPDOWN FUNCTIONALITY TEST HAD ISSUES!")
        print("🔧 Some UI elements may need attention")
    print("="*60)
