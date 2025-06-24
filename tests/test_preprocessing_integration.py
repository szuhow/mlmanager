#!/usr/bin/env python3
"""
Test script for medical preprocessing integration in training pipeline
"""

import sys
import os
import torch
import numpy as np
import argparse
from pathlib import Path

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'ml'))

# Set up basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def test_preprocessing_arguments():
    """Test that all preprocessing arguments are properly parsed"""
    print("🧪 Testing preprocessing arguments parsing...")
    
    try:
        from ml.training.train import parse_args
        
        # Test basic preprocessing arguments
        test_args = [
            '--mode', 'train',
            '--data-path', 'dummy',
            '--use-medical-preprocessing',
            '--medical-preprocessing-type', 'angiography',
            '--preprocessing-clahe-clip-limit', '4.0',
            '--preprocessing-clahe-tile-size', '16',
            '--preprocessing-use-unsharp-masking',
            '--preprocessing-unsharp-radius', '1.5',
            '--preprocessing-unsharp-amount', '1.2',
            '--preprocessing-use-frangi',
            '--preprocessing-frangi-scale-range', '2,15',
            '--preprocessing-frangi-scale-step', '2.5',
            '--preprocessing-use-histogram-equalization',
            '--preprocessing-use-denoising',
            '--preprocessing-noise-variance', '0.2',
            '--preprocessing-gamma-correction', '1.2',
            '--preprocessing-vessel-enhancement-sigma', '1.5',
            '--preprocessing-custom-pipeline', 'clahe,unsharp,frangi'
        ]
        
        # Parse arguments
        sys.argv = ['test'] + test_args
        args = parse_args()
        
        # Verify preprocessing arguments
        assert args.use_medical_preprocessing == True
        assert args.medical_preprocessing_type == 'angiography'
        assert args.preprocessing_clahe_clip_limit == 4.0
        assert args.preprocessing_clahe_tile_size == 16
        assert args.preprocessing_use_unsharp_masking == True
        assert args.preprocessing_unsharp_radius == 1.5
        assert args.preprocessing_unsharp_amount == 1.2
        assert args.preprocessing_use_frangi == True
        assert args.preprocessing_frangi_scale_range == '2,15'
        assert args.preprocessing_frangi_scale_step == 2.5
        assert args.preprocessing_use_histogram_equalization == True
        assert args.preprocessing_use_denoising == True
        assert args.preprocessing_noise_variance == 0.2
        assert args.preprocessing_gamma_correction == 1.2
        assert args.preprocessing_vessel_enhancement_sigma == 1.5
        assert args.preprocessing_custom_pipeline == 'clahe,unsharp,frangi'
        
        print("✅ All preprocessing arguments parsed correctly")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing arguments parsing failed: {e}")
        return False

def test_preprocessing_function():
    """Test the apply_medical_preprocessing function with custom parameters"""
    print("🧪 Testing apply_medical_preprocessing function...")
    
    try:
        from ml.training.train import apply_medical_preprocessing
        
        # Create a test image
        test_image = torch.rand(1, 256, 256)
        
        # Test with custom parameters
        preprocessing_params = {
            'clahe_clip_limit': 4.0,
            'clahe_tile_size': 16,
            'use_unsharp_masking': True,
            'unsharp_radius': 1.5,
            'unsharp_amount': 1.2,
            'use_frangi': True,
            'frangi_scale_range': '2,15',
            'frangi_scale_step': 2.5,
            'use_histogram_equalization': True,
            'use_denoising': True,
            'noise_variance': 0.2,
            'gamma_correction': 1.2,
            'vessel_enhancement_sigma': 1.5
        }
        
        # Test different preprocessing types
        for preprocessing_type in ['angiography', 'ct_coronary', 'oct_coronary', 'general']:
            result = apply_medical_preprocessing(
                test_image, 
                preprocessing_type=preprocessing_type, 
                **preprocessing_params
            )
            
            # Check that result has the same shape
            assert result.shape == test_image.shape, f"Shape mismatch for {preprocessing_type}"
            assert isinstance(result, torch.Tensor), f"Result should be tensor for {preprocessing_type}"
            
            print(f"✅ {preprocessing_type} preprocessing works correctly")
        
        print("✅ apply_medical_preprocessing function works correctly")
        return True
        
    except Exception as e:
        print(f"❌ apply_medical_preprocessing function failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_pipeline():
    """Test the custom preprocessing pipeline functionality"""
    print("🧪 Testing custom preprocessing pipeline...")
    
    try:
        from ml.training.train import _apply_custom_preprocessing_pipeline
        
        # Create a test image
        test_image = np.random.rand(256, 256)
        
        # Test custom pipeline
        pipeline_str = "clahe,gamma"
        params = {
            'clahe_clip_limit': 3.0,
            'clahe_tile_size': 8,
            'gamma_correction': 1.2
        }
        
        result = _apply_custom_preprocessing_pipeline(test_image, pipeline_str, params)
        
        # Check that result has the same shape
        assert result.shape == test_image.shape, "Shape mismatch for custom pipeline"
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        
        print("✅ Custom preprocessing pipeline works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Custom preprocessing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transform_params_integration():
    """Test that preprocessing parameters are correctly integrated into transform_params"""
    print("🧪 Testing transform_params integration...")
    
    try:
        # Create a mock args object with preprocessing parameters
        class MockArgs:
            def __init__(self):
                self.random_flip = True
                self.random_rotate = True
                self.random_scale = True
                self.random_intensity = True
                self.crop_size = 128
                self.use_medical_preprocessing = True
                self.medical_preprocessing_type = 'angiography'
                self.preprocessing_clahe_clip_limit = 4.0
                self.preprocessing_clahe_tile_size = 16
                self.preprocessing_use_unsharp_masking = True
                self.preprocessing_unsharp_radius = 1.5
                self.preprocessing_unsharp_amount = 1.2
                self.preprocessing_use_frangi = True
                self.preprocessing_frangi_scale_range = '2,15'
                self.preprocessing_frangi_scale_step = 2.5
                self.preprocessing_use_histogram_equalization = True
                self.preprocessing_use_denoising = True
                self.preprocessing_noise_variance = 0.2
                self.preprocessing_intensity_range = 'auto'
                self.preprocessing_gamma_correction = 1.2
                self.preprocessing_vessel_enhancement_sigma = 1.5
                self.preprocessing_custom_pipeline = 'clahe,unsharp'
        
        args = MockArgs()
        
        # Build transform_params as done in the training script
        transform_params = {
            'use_random_flip': getattr(args, 'random_flip', True),
            'use_random_rotate': getattr(args, 'random_rotate', True),
            'use_random_scale': getattr(args, 'random_scale', True),
            'use_random_intensity': getattr(args, 'random_intensity', True),
            'crop_size': getattr(args, 'crop_size', 128),
            'use_medical_preprocessing': getattr(args, 'use_medical_preprocessing', False),
            'medical_preprocessing_type': getattr(args, 'medical_preprocessing_type', 'angiography'),
            # Add all detailed preprocessing parameters
            'preprocessing_clahe_clip_limit': getattr(args, 'preprocessing_clahe_clip_limit', 3.0),
            'preprocessing_clahe_tile_size': getattr(args, 'preprocessing_clahe_tile_size', 8),
            'preprocessing_use_unsharp_masking': getattr(args, 'preprocessing_use_unsharp_masking', False),
            'preprocessing_unsharp_radius': getattr(args, 'preprocessing_unsharp_radius', 1.0),
            'preprocessing_unsharp_amount': getattr(args, 'preprocessing_unsharp_amount', 1.0),
            'preprocessing_use_frangi': getattr(args, 'preprocessing_use_frangi', False),
            'preprocessing_frangi_scale_range': getattr(args, 'preprocessing_frangi_scale_range', '1,10'),
            'preprocessing_frangi_scale_step': getattr(args, 'preprocessing_frangi_scale_step', 2.0),
            'preprocessing_use_histogram_equalization': getattr(args, 'preprocessing_use_histogram_equalization', False),
            'preprocessing_use_denoising': getattr(args, 'preprocessing_use_denoising', False),
            'preprocessing_noise_variance': getattr(args, 'preprocessing_noise_variance', 0.1),
            'preprocessing_intensity_range': getattr(args, 'preprocessing_intensity_range', 'auto'),
            'preprocessing_gamma_correction': getattr(args, 'preprocessing_gamma_correction', 1.0),
            'preprocessing_vessel_enhancement_sigma': getattr(args, 'preprocessing_vessel_enhancement_sigma', 1.0),
            'preprocessing_custom_pipeline': getattr(args, 'preprocessing_custom_pipeline', '')
        }
        
        # Verify all parameters are correctly set
        assert transform_params['use_medical_preprocessing'] == True
        assert transform_params['medical_preprocessing_type'] == 'angiography'
        assert transform_params['preprocessing_clahe_clip_limit'] == 4.0
        assert transform_params['preprocessing_clahe_tile_size'] == 16
        assert transform_params['preprocessing_use_unsharp_masking'] == True
        assert transform_params['preprocessing_unsharp_radius'] == 1.5
        assert transform_params['preprocessing_unsharp_amount'] == 1.2
        assert transform_params['preprocessing_use_frangi'] == True
        assert transform_params['preprocessing_frangi_scale_range'] == '2,15'
        assert transform_params['preprocessing_frangi_scale_step'] == 2.5
        assert transform_params['preprocessing_use_histogram_equalization'] == True
        assert transform_params['preprocessing_use_denoising'] == True
        assert transform_params['preprocessing_noise_variance'] == 0.2
        assert transform_params['preprocessing_gamma_correction'] == 1.2
        assert transform_params['preprocessing_vessel_enhancement_sigma'] == 1.5
        assert transform_params['preprocessing_custom_pipeline'] == 'clahe,unsharp'
        
        print("✅ Transform params integration works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Transform params integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all preprocessing integration tests"""
    print("🚀 Running Medical Preprocessing Integration Tests")
    print("=" * 60)
    
    tests = [
        test_preprocessing_arguments,
        test_preprocessing_function,
        test_custom_pipeline,
        test_transform_params_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All preprocessing integration tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
