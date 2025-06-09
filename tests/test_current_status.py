#!/usr/bin/env python3
"""
Test current status of the training system
"""

import os
import sys
import django
import time

# Setup Django
sys.path.append('/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coronary_experiments.settings')
django.setup()

from ml_manager.models import MLModel

def test_current_status():
    """Test what models exist and their current status"""
    print("=== CURRENT STATUS TEST ===")
    
    # Check existing models
    print("\n1. Checking existing models...")
    models = MLModel.objects.all().order_by('-id')
    print(f"Found {len(models)} total models")
    
    if models.exists():
        print("\nRecent models:")
        for model in models[:5]:
            print(f"  ID: {model.id}, Status: {model.status}, Name: {model.name}")
            print(f"      Created: {model.created_at}")
    else:
        print("No models found in database")
    
    # Check for training models
    training_models = models.filter(status='training')
    pending_models = models.filter(status='pending')
    loading_models = models.filter(status='loading')
    
    print(f"\n2. Status breakdown:")
    print(f"   Training: {len(training_models)}")
    print(f"   Pending: {len(pending_models)}")
    print(f"   Loading: {len(loading_models)}")
    
    # Check dataset
    print("\n3. Checking dataset...")
    data_path = "/app/shared/datasets/data"
    if os.path.exists(data_path):
        imgs_path = os.path.join(data_path, "imgs")
        masks_path = os.path.join(data_path, "masks")
        
        img_count = len(os.listdir(imgs_path)) if os.path.exists(imgs_path) else 0
        mask_count = len(os.listdir(masks_path)) if os.path.exists(masks_path) else 0
        
        print(f"   Images: {img_count}")
        print(f"   Masks: {mask_count}")
        print(f"   Match: {'✅' if img_count == mask_count else '❌'}")
    else:
        print("   Dataset path not found")
    
    # Test model creation
    print("\n4. Testing model creation...")
    try:
        import torch
        from shared.train import create_model_from_registry
        
        device = torch.device("cpu")
        model, arch_info = create_model_from_registry(
            model_type='unet',
            device=device,
            spatial_dims=2,
            in_channels=1,
            out_channels=1
        )
        print("   Model creation: ✅ Working")
        print(f"   Architecture: {arch_info.display_name}")
    except Exception as e:
        print(f"   Model creation: ❌ Failed - {e}")
    
    # Test callback system
    print("\n5. Testing callback system...")
    try:
        from shared.utils.training_callback import TrainingCallback
        
        # Create test model
        test_model = MLModel.objects.create(
            name="Status Test Model",
            description="Test callback system",
            status="pending"
        )
        
        callback = TrainingCallback(test_model.id, "test-run")
        callback.on_training_start()
        
        test_model.refresh_from_db()
        print(f"   Callback test: ✅ Working - Status changed to '{test_model.status}'")
        
        # Cleanup
        test_model.delete()
        
    except Exception as e:
        print(f"   Callback test: ❌ Failed - {e}")
    
    print("\n=== STATUS TEST COMPLETE ===")

if __name__ == "__main__":
    test_current_status()
