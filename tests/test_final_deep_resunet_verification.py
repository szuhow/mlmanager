#!/usr/bin/env python3
"""
Final comprehensive test for DeepResUNet with attention.
Tests all parameter combinations to ensure the fix is complete.
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append('/app')
sys.path.append('/app/ml')
sys.path.append('/app/ml/training')

from ml.training.models.resunet_model import DeepResUNet

def test_comprehensive():
    """Test all combinations of parameters for DeepResUNet with attention."""
    
    print("🧪 Comprehensive DeepResUNet with Attention Test")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        # (n_channels, n_classes, bilinear, use_attention)
        (1, 1, False, True),   # Grayscale, binary segmentation, standard conv
        (1, 1, True, True),    # Grayscale, binary segmentation, bilinear
        (3, 1, False, True),   # RGB, binary segmentation, standard conv
        (3, 1, True, True),    # RGB, binary segmentation, bilinear
        (1, 2, False, True),   # Grayscale, 2-class segmentation, standard conv
        (1, 2, True, True),    # Grayscale, 2-class segmentation, bilinear
        (3, 3, False, True),   # RGB, 3-class segmentation, standard conv
        (3, 3, True, True),    # RGB, 3-class segmentation, bilinear
        
        # Also test without attention for comparison
        (1, 1, False, False),  # Standard DeepResUNet without attention
        (3, 1, True, False),   # Bilinear without attention
    ]
    
    input_sizes = [
        (256, 256),  # Standard size
        (512, 512),  # Larger size
        (128, 128),  # Smaller size
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    print()
    
    total_tests = 0
    passed_tests = 0
    
    for i, (n_channels, n_classes, bilinear, use_attention) in enumerate(test_configs, 1):
        for j, (h, w) in enumerate(input_sizes, 1):
            total_tests += 1
            
            config_desc = f"channels={n_channels}, classes={n_classes}, bilinear={bilinear}, attention={use_attention}"
            size_desc = f"{h}x{w}"
            
            print(f"Test {i}.{j}: {config_desc}, size={size_desc}")
            
            try:
                # Create model
                model = DeepResUNet(
                    n_channels=n_channels,
                    n_classes=n_classes,
                    bilinear=bilinear,
                    use_attention=use_attention
                ).to(device)
                
                # Create test input
                batch_size = 2
                test_input = torch.randn(batch_size, n_channels, h, w).to(device)
                
                # Forward pass
                with torch.no_grad():
                    output = model(test_input)
                
                # Check output shape
                expected_shape = (batch_size, n_classes, h, w)
                actual_shape = tuple(output.shape)
                
                if actual_shape == expected_shape:
                    print(f"  ✅ Input: {tuple(test_input.shape)} → Output: {actual_shape}")
                    passed_tests += 1
                else:
                    print(f"  ❌ Shape mismatch! Expected: {expected_shape}, Got: {actual_shape}")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
            
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! DeepResUNet with attention is fully functional.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

def test_model_registry_integration():
    """Test that the fixed model works with the model registry."""
    print("\n🔧 Testing model registry integration...")
    
    try:
        from ml.utils.architecture_registry import registry
        
        # Test that deep_resunet_attention is properly registered
        deep_resunet_info = registry.get_architecture('deep_resunet_attention')
        if deep_resunet_info:
            print(f"✅ Model registered: {deep_resunet_info.display_name}")
            
            # Try to create the model through registry
            model_class = deep_resunet_info.model_class
            test_params = {
                'n_channels': 1,
                'n_classes': 1,
                'bilinear': False,
                'use_attention': True
            }
            
            model = model_class(**test_params)
            test_input = torch.randn(1, 1, 256, 256)
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"✅ Registry integration test passed!")
            print(f"   Input: {tuple(test_input.shape)} → Output: {tuple(output.shape)}")
            return True
            
        else:
            print("❌ deep_resunet_attention not found in registry")
            return False
            
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_comprehensive()
    success2 = test_model_registry_integration()
    
    if success1 and success2:
        print("\n🚀 All verification tests completed successfully!")
        print("DeepResUNet with attention is ready for production use.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please review the issues above.")
        sys.exit(1)
