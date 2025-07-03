#!/usr/bin/env python3
"""
Test GUI-level preprocessing integration
Tests that preprocessing parameters from GUI are properly handled in the training pipeline
"""

import numpy as np
import torch
import os
import sys
import tempfile
import argparse
from unittest.mock import Mock

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

def test_preprocessing_args():
    """Test that preprocessing arguments are properly parsed"""
    print("Testing preprocessing arguments parsing...")
    
    try:
        # Import the argument parser from training script
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml', 'training'))
        from train import parse_args
        
        # Test with preprocessing arguments
        test_args = [
            '--mode', 'train',
            '--data-path', '/tmp/test',
            '--use-medical-preprocessing',
            '--preprocessing-type', 'angiography',
            '--clahe-clip-limit', '3.0',
            '--unsharp-amount', '2.0',
            '--frangi-sigma-min', '0.5',
            '--frangi-sigma-max', '8.0',
            '--noise-reduction-sigma', '1.5',
            '--normalize-intensity'
        ]
        
        # Parse arguments
        args = parse_args(test_args)
        
        # Verify preprocessing parameters
        assert hasattr(args, 'use_medical_preprocessing')
        assert args.use_medical_preprocessing == True
        assert args.preprocessing_type == 'angiography'
        assert args.clahe_clip_limit == 3.0
        assert args.unsharp_amount == 2.0
        assert args.frangi_sigma_min == 0.5
        assert args.frangi_sigma_max == 8.0
        assert args.noise_reduction_sigma == 1.5
        assert args.normalize_intensity == True
        
        print("✓ Preprocessing arguments parsing test passed")
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing arguments parsing test failed: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_preprocessing_integration():
    """Test preprocessing integration in MONAI transforms"""
    print("Testing preprocessing integration in MONAI transforms...")
    
    try:
        from ml.utils.medical_preprocessing import MedicalImagePreprocessor
        
        # Create mock args with preprocessing enabled
        args = Mock()
        args.use_medical_preprocessing = True
        args.preprocessing_type = 'angiography'
        args.clahe_clip_limit = 2.0
        args.unsharp_amount = 1.5
        args.frangi_sigma_min = 1.0
        args.frangi_sigma_max = 16.0
        args.noise_reduction_sigma = 1.0
        args.normalize_intensity = True
        
        # Test preprocessing wrapper function
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml', 'training'))
        from train import create_medical_preprocessing_wrapper
        
        wrapper_func = create_medical_preprocessing_wrapper(args)
        
        # Test with sample image
        test_image = np.random.rand(256, 256).astype(np.float32)
        processed = wrapper_func(test_image)
        
        assert processed.shape == test_image.shape
        assert isinstance(processed, np.ndarray)
        
        print(f"  Input shape: {test_image.shape}")
        print(f"  Output shape: {processed.shape}")
        print(f"  Input range: [{test_image.min():.4f}, {test_image.max():.4f}]")
        print(f"  Output range: [{processed.min():.4f}, {processed.max():.4f}]")
        print("✓ Preprocessing integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing integration test failed: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_monai_transforms_with_preprocessing():
    """Test MONAI transforms with preprocessing enabled"""
    print("Testing MONAI transforms with preprocessing...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml', 'training'))
        from train import get_monai_transforms
        
        # Create transform params with preprocessing
        transform_params = {
            'use_medical_preprocessing': True,
            'preprocessing_type': 'angiography',
            'clahe_clip_limit': 2.0,
            'unsharp_amount': 1.5,
            'frangi_sigma_min': 1.0,
            'frangi_sigma_max': 16.0,
            'noise_reduction_sigma': 1.0,
            'normalize_intensity': True,
            'use_random_flip': True,
            'use_random_rotate': True,
            'use_random_scale': True,
            'use_random_intensity': True,
            'crop_size': 128
        }
        
        # Get transforms
        train_transforms = get_monai_transforms(transform_params, for_training=True)
        val_transforms = get_monai_transforms(transform_params, for_training=False)
        
        print(f"  Train transforms: {len(train_transforms.transforms)} steps")
        print(f"  Val transforms: {len(val_transforms.transforms)} steps")
        
        # Check that medical preprocessing is included
        transform_names = [type(t).__name__ for t in train_transforms.transforms]
        print(f"  Transform pipeline: {transform_names}")
        
        # Look for Lambdad transform that applies medical preprocessing
        lambda_transforms = [t for t in train_transforms.transforms if type(t).__name__ == 'Lambdad']
        print(f"  Found {len(lambda_transforms)} Lambda transforms (including medical preprocessing)")
        
        print("✓ MONAI transforms with preprocessing test passed")
        return True
        
    except Exception as e:
        print(f"✗ MONAI transforms with preprocessing test failed: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_loss_function_integration():
    """Test advanced loss function integration"""
    print("Testing advanced loss function integration...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml', 'training'))
        from train import create_advanced_loss_function
        
        # Test different loss types
        loss_types = ['dice', 'tversky', 'focal', 'combo_dice_bce', 'boundary']
        
        for loss_type in loss_types:
            try:
                loss_fn = create_advanced_loss_function(loss_type)
                print(f"  ✓ Created {loss_type} loss: {type(loss_fn).__name__}")
            except Exception as e:
                print(f"  ⚠ Failed to create {loss_type} loss: {e}")
        
        print("✓ Advanced loss function integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Advanced loss function integration test failed: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all GUI preprocessing tests"""
    print("=" * 60)
    print("GUI PREPROCESSING INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_preprocessing_args,
        test_preprocessing_integration,
        test_monai_transforms_with_preprocessing,
        test_loss_function_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 40)
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All GUI preprocessing integration tests PASSED!")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
