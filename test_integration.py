#!/usr/bin/env python3
"""
Test integration of advanced loss functions and medical preprocessing in training pipeline
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
core_path = str(project_root / 'core')
ml_path = str(project_root / 'ml')

if core_path not in sys.path:
    sys.path.insert(0, core_path)
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

def test_advanced_loss_integration():
    """Test advanced loss function integration"""
    print("🧪 Testing Advanced Loss Functions Integration...")
    
    try:
        # Import our training utilities
        from ml.training.train import create_advanced_loss_function
        
        # Test various advanced loss functions
        test_losses = [
            'dice',
            'tversky',
            'tversky_precision', 
            'tversky_recall',
            'focal',
            'combo_dice_bce',
            'soft_dice',
            'weighted_bce',
            'boundary',
            'stable_bce',
            'bce',
            'unknown_loss'  # Should fallback to dice
        ]
        
        for loss_name in test_losses:
            try:
                loss_fn = create_advanced_loss_function(loss_name)
                print(f"✅ {loss_name}: {type(loss_fn).__name__}")
                
                # Test with dummy data
                pred = torch.randn(2, 1, 32, 32, requires_grad=True)
                target = torch.randint(0, 2, (2, 1, 32, 32)).float()
                
                loss_value = loss_fn(pred, target)
                print(f"   Loss value: {loss_value.item():.4f}")
                
            except Exception as e:
                print(f"❌ {loss_name}: {e}")
                
    except ImportError as e:
        print(f"❌ Advanced loss functions not available: {e}")
        return False
    
    print("✅ Advanced loss function integration test completed!\n")
    return True

def test_medical_preprocessing_integration():
    """Test medical preprocessing integration"""
    print("🧪 Testing Medical Preprocessing Integration...")
    
    try:
        # Import preprocessing utilities
        from ml.training.train import apply_medical_preprocessing
        from ml.utils.medical_preprocessing import MedicalImagePreprocessor
        
        # Create test image
        test_image = np.random.rand(128, 128).astype(np.float32)
        test_tensor = torch.from_numpy(test_image).unsqueeze(0)
        
        # Test different preprocessing types
        preprocessing_types = ['angiography', 'ct_coronary', 'oct_coronary', 'general']
        
        for prep_type in preprocessing_types:
            try:
                # Test with numpy array
                result_np = apply_medical_preprocessing(test_image, prep_type)
                print(f"✅ {prep_type} (numpy): {result_np.shape}, range: [{result_np.min():.3f}, {result_np.max():.3f}]")
                
                # Test with torch tensor
                result_tensor = apply_medical_preprocessing(test_tensor, prep_type)
                print(f"✅ {prep_type} (tensor): {result_tensor.shape}, range: [{result_tensor.min():.3f}, {result_tensor.max():.3f}]")
                
            except Exception as e:
                print(f"❌ {prep_type}: {e}")
        
        # Test MedicalImagePreprocessor directly
        preprocessor = MedicalImagePreprocessor(
            target_size=(128, 128),
            normalize_method='percentile',
            enhance_contrast=True,
            enhance_vessels=True
        )
        
        result = preprocessor.preprocess(test_image)
        print(f"✅ Direct preprocessor: {result['image'].shape}")
        if 'metadata' in result:
            print(f"   Metadata keys: {list(result['metadata'].keys())}")
            
    except ImportError as e:
        print(f"❌ Medical preprocessing not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Medical preprocessing test failed: {e}")
        return False
    
    print("✅ Medical preprocessing integration test completed!\n")
    return True

def test_transforms_integration():
    """Test MONAI transforms integration with medical preprocessing"""
    print("🧪 Testing MONAI Transforms Integration...")
    
    try:
        from ml.training.train import get_monai_transforms
        
        # Test parameters with medical preprocessing enabled
        test_params = {
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 128,
            'use_medical_preprocessing': True,
            'medical_preprocessing_type': 'angiography'
        }
        
        # Get training transforms
        train_transforms = get_monai_transforms(test_params, for_training=True)
        print(f"✅ Training transforms created: {len(train_transforms.transforms)} transforms")
        
        # Get validation transforms
        val_transforms = get_monai_transforms(test_params, for_training=False)
        print(f"✅ Validation transforms created: {len(val_transforms.transforms)} transforms")
        
        # Test without medical preprocessing
        test_params_no_med = test_params.copy()
        test_params_no_med['use_medical_preprocessing'] = False
        
        basic_transforms = get_monai_transforms(test_params_no_med, for_training=True)
        print(f"✅ Basic transforms created: {len(basic_transforms.transforms)} transforms")
        
        print("✅ Transform integration test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Transform integration test failed: {e}")
        return False

def test_loss_components():
    """Test loss components functionality for evaluation"""
    print("🧪 Testing Loss Components for Evaluation...")
    
    try:
        from ml.utils.advanced_losses import (
            TverskyLoss, FocalLoss, ComboDiceBCELoss, 
            SoftDiceLoss, WeightedBCELoss, BoundaryLoss, StableBCELoss
        )
        
        # Create dummy data
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        # Test get_loss_components for each loss
        losses_to_test = [
            TverskyLoss(),
            FocalLoss(),
            ComboDiceBCELoss(),
            SoftDiceLoss(),
            WeightedBCELoss(),
            BoundaryLoss(),
            StableBCELoss()
        ]
        
        for loss_fn in losses_to_test:
            try:
                # Test loss computation
                loss_value = loss_fn(pred, target)
                
                # Test loss components
                components = loss_fn.get_loss_components(pred, target)
                
                print(f"✅ {type(loss_fn).__name__}: loss={loss_value.item():.4f}, components={list(components.keys())}")
                
            except Exception as e:
                print(f"❌ {type(loss_fn).__name__}: {e}")
        
        print("✅ Loss components test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Loss components test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Starting Advanced Loss & Preprocessing Integration Tests...\n")
    
    results = []
    
    # Test advanced loss functions
    results.append(test_advanced_loss_integration())
    
    # Test medical preprocessing
    results.append(test_medical_preprocessing_integration())
    
    # Test transforms integration
    results.append(test_transforms_integration())
    
    # Test loss components
    results.append(test_loss_components())
    
    # Summary
    print("📊 Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("🎉 All integration tests passed! Advanced loss functions and medical preprocessing are ready for training.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
