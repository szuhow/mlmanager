#!/usr/bin/env python3
"""
Simple test to verify model status transitions work.
"""

import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

def test_model_creation_and_status():
    """Test that we can create a model and update its status"""
    
    from ml_manager.models import MLModel
    
    print("🧪 Testing Model Status Transitions...")
    
    # Create test model
    test_model_data = {
        'name': 'Simple Status Test Model',
        'description': 'Test model for status transitions',
        'status': 'pending',
        'current_epoch': 0,
        'total_epochs': 5,
        'train_loss': 0.0,
        'val_loss': 0.0,
        'train_dice': 0.0,
        'val_dice': 0.0,
        'best_val_dice': 0.0,
        'training_data_info': {
            'model_type': 'unet',
            'data_path': 'test/data',
            'batch_size': 16,
        }
    }
    
    # Create model
    model = MLModel.objects.create(**test_model_data)
    print(f"✅ Created model {model.id} with status: {model.status}")
    
    # Test status transitions
    model.status = 'loading'
    model.save()
    print(f"✅ Updated status to: {model.status}")
    
    model.status = 'training'
    model.save()
    print(f"✅ Updated status to: {model.status}")
    
    model.status = 'completed'
    model.save()
    print(f"✅ Updated status to: {model.status}")
    
    # Cleanup
    model.delete()
    print(f"🧹 Cleaned up test model {model.id}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_model_creation_and_status()
        print("\n" + "="*50)
        if success:
            print("🎉 MODEL STATUS TEST PASSED!")
        else:
            print("❌ MODEL STATUS TEST FAILED!")
        print("="*50)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
