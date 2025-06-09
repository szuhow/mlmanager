#!/usr/bin/env python3
"""
Test the rerun training feature implementation
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from django.test import RequestFactory
from django.contrib.auth.models import User
from ml_manager.models import MLModel
from ml_manager.views import StartTrainingView

def test_rerun_training_feature():
    """Test the rerun training feature"""
    print("🧪 Testing Rerun Training Feature...")
    
    try:
        # Check if we have existing models
        models = MLModel.objects.all()[:3]
        print(f"📊 Found {models.count()} models in database")
        
        if models.count() == 0:
            print("⚠️  No models found. Creating a test model...")
            # Create a test model with training data
            test_model = MLModel.objects.create(
                name="Test Model for Rerun",
                description="Test model to verify rerun functionality",
                status="completed",
                total_epochs=50,
                training_data_info={
                    'model_type': 'unet',
                    'data_path': 'shared/datasets/test_data',
                    'batch_size': 16,
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'validation_split': 0.2,
                    'use_random_flip': True,
                    'use_random_rotate': True,
                    'use_random_scale': False,
                    'use_random_intensity': True,
                    'crop_size': 128,
                    'num_workers': 4,
                }
            )
            models = [test_model]
            print(f"✅ Created test model {test_model.id}")
        
        # Test the rerun functionality
        test_model = models[0]
        print(f"🎯 Testing rerun with model {test_model.id}: {test_model.name}")
        
        # Create a mock request factory
        factory = RequestFactory()
        
        # Create a GET request with rerun parameter
        request = factory.get(f'/ml_manager/start-training/?rerun={test_model.id}')
        
        # Create a user for the request
        user = User.objects.get_or_create(username='testuser', defaults={'email': 'test@example.com'})[0]
        request.user = user
        
        # Initialize the view
        view = StartTrainingView()
        view.request = request
        
        # Test get_initial method
        initial_data = view.get_initial()
        print("📋 Initial data from rerun:")
        for key, value in initial_data.items():
            print(f"  {key}: {value}")
        
        # Verify the data matches expected values
        expected_checks = [
            ('name', f"{test_model.name} (Rerun)"),
            ('model_type', 'unet'),
            ('batch_size', 16),
            ('epochs', 50),
            ('learning_rate', 0.001),
            ('validation_split', 0.2),
            ('use_random_flip', True),
            ('crop_size', 128),
        ]
        
        success = True
        for field, expected in expected_checks:
            actual = initial_data.get(field)
            if actual == expected:
                print(f"  ✅ {field}: {actual} (correct)")
            else:
                print(f"  ❌ {field}: {actual} (expected {expected})")
                success = False
        
        # Test get_context_data method
        context = view.get_context_data()
        if 'rerun_model' in context:
            print(f"✅ Context includes rerun_model: {context['rerun_model'].name}")
        else:
            print("❌ Context missing rerun_model")
            success = False
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rerun_training_feature()
    
    print("\n" + "="*60)
    if success:
        print("🎉 RERUN TRAINING FEATURE TEST PASSED!")
        print("✅ The rerun training functionality is working correctly")
        print("✅ Form pre-population from existing model works")
        print("✅ Context data includes rerun model information")
    else:
        print("❌ RERUN TRAINING FEATURE TEST FAILED!")
        print("🔧 Some functionality needs fixing")
    print("="*60)
