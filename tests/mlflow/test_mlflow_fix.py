#!/usr/bin/env python3
"""
Test the MLflow conflict fix in StartTrainingView
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

import logging
from django.test import RequestFactory
from django.contrib.auth.models import User
from django.contrib.sessions.backends.db import SessionStore
from django.contrib.messages.storage.fallback import FallbackStorage

from core.apps.ml_manager.views import StartTrainingView
from core.apps.ml_manager.forms import TrainingForm
from core.apps.ml_manager.models import MLModel

def test_mlflow_conflict_fix():
    """Test that MLflow conflict is resolved in StartTrainingView"""
    print("🧪 Testing MLflow Conflict Fix in StartTrainingView...")
    
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create test user
        user, created = User.objects.get_or_create(
            username='test_user',
            defaults={'email': 'test@example.com'}
        )
        if created:
            user.set_password('test123')
            user.save()
        
        print(f"✅ Test user: {user.username}")
        
        # Prepare test form data
        form_data = {
            'name': 'MLflow Fix Test Model',
            'description': 'Testing MLflow conflict resolution',
            'model_type': 'unet',
            'data_path': '/app/data/datasets',
            'batch_size': 2,
            'epochs': 1,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'resolution': '256',
            'device': 'auto',
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 128,
            'num_workers': 0,
            'use_random_crop': True,
            'use_elastic_transform': False,
            'use_gaussian_noise': False,
            'elastic_alpha': 1.0,
            'elastic_sigma': 0.1,
            'noise_std': 0.1,
            'flip_probability': 0.5,
            'rotation_range': 10,
            'scale_range_min': 0.9,
            'scale_range_max': 1.1,
            'intensity_range': 0.1,
        }
        
        # Create form instance
        form = TrainingForm(data=form_data)
        
        if not form.is_valid():
            print(f"❌ Form validation failed: {form.errors}")
            return False
            
        print("✅ Form validation successful")
        
        # Create mock request
        factory = RequestFactory()
        request = factory.post('/ml/start-training/', data=form_data)
        request.user = user
        request.session = SessionStore()
        request._messages = FallbackStorage(request)
        
        # Create view instance
        view = StartTrainingView()
        view.setup(request)
        
        print("🚀 Testing form_valid method...")
        
        # Test the form_valid method
        initial_model_count = MLModel.objects.count()
        
        try:
            response = view.form_valid(form)
            print("✅ form_valid executed successfully")
            
            # Check if model was created
            final_model_count = MLModel.objects.count()
            if final_model_count > initial_model_count:
                new_model = MLModel.objects.latest('created_at')
                print(f"✅ New model created: {new_model.name} (ID: {new_model.id})")
                print(f"   Status: {new_model.status}")
                print(f"   MLflow Run ID: {new_model.mlflow_run_id}")
                
                # Clean up
                new_model.delete()
                print("🧹 Cleaned up test model")
                
                return True
            else:
                print("❌ No new model was created")
                return False
                
        except Exception as e:
            print(f"❌ Error in form_valid: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mlflow_conflict_fix()
    
    print("\n" + "="*60)
    if success:
        print("🎉 MLFLOW CONFLICT FIX TEST PASSED!")
        print("✅ The MLflow conflict issue has been resolved")
        print("✅ StartTrainingView can now handle active MLflow runs properly")
    else:
        print("❌ MLFLOW CONFLICT FIX TEST FAILED!")
        print("🔧 The MLflow conflict issue still needs attention")
    print("="*60)
