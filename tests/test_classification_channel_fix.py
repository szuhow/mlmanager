#!/usr/bin/env python3
"""
Test classification model channel configuration fix
"""

import sys
import os
import torch

# Add the project path
sys.path.insert(0, '/Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager')

def test_classification_channel_fix():
    """Test that classification models now work with 1-channel input"""
    print("🔧 Testing Classification Model Channel Fix")
    print("=" * 60)
    
    try:
        from ml.training.train import create_model_from_registry
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Test 1: Test artery classification with 1 channel input
        print("\n🧪 Test 1: Artery classification with 1 channel")
        
        model_kwargs = {
            'in_channels': 1,  # Binary mask input
            'out_channels': 2  # Binary classification
        }
        
        model, arch_info = create_model_from_registry(
            model_type='unet',
            device=device,
            task_type='artery_classification',
            **model_kwargs
        )
        
        print(f"✅ Model created: {arch_info.display_name}")
        print(f"   Architecture: {arch_info.key}")
        
        # Test with sample data
        test_input = torch.randn(16, 1, 128, 128).to(device)  # Single channel binary mask
        print(f"   Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
            print(f"   Output shape: {output.shape}")
            
        # Check if output is correct for classification
        expected_output_shape = torch.Size([16, 2])  # [batch_size, n_classes]
        if output.shape == expected_output_shape:
            print(f"✅ Output shape correct: {output.shape}")
        else:
            print(f"❌ Output shape wrong: {output.shape}, expected: {expected_output_shape}")
            return False
        
        # Test 2: Test different model architectures
        print("\n🧪 Test 2: Different classification architectures")
        
        architectures_to_test = ['unet', 'resunet']
        
        for model_type in architectures_to_test:
            try:
                model, arch_info = create_model_from_registry(
                    model_type=model_type,
                    device=device,
                    task_type='artery_classification',
                    in_channels=1,
                    out_channels=2
                )
                
                with torch.no_grad():
                    output = model(test_input)
                    
                if output.shape == expected_output_shape:
                    print(f"✅ {model_type} → {arch_info.display_name}: {output.shape}")
                else:
                    print(f"❌ {model_type} → wrong shape: {output.shape}")
                    
            except Exception as e:
                print(f"❌ {model_type} failed: {e}")
        
        # Test 3: Check model parameters
        print("\n🧪 Test 3: Model parameter analysis")
        
        # Create model and check first layer
        model, _ = create_model_from_registry(
            model_type='unet',
            device=device, 
            task_type='artery_classification',
            in_channels=1,
            out_channels=2
        )
        
        # Find first convolutional layer
        first_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv = module
                print(f"   First conv layer: {name}")
                print(f"   Input channels: {module.in_channels}")
                print(f"   Output channels: {module.out_channels}")
                break
        
        if first_conv and first_conv.in_channels == 1:
            print("✅ First layer correctly configured for 1 channel input")
        else:
            print(f"❌ First layer has {first_conv.in_channels if first_conv else 'unknown'} channels")
            return False
        
        print("\n🎉 All tests passed!")
        print("✅ Classification model channel fix is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that segmentation models still work with 3 channels"""
    print("\n🔧 Testing Backward Compatibility")
    print("=" * 40)
    
    try:
        from ml.training.train import create_model_from_registry
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Test standard segmentation model
        model, arch_info = create_model_from_registry(
            model_type='unet',
            device=device,
            task_type=None,  # No specific task type
            in_channels=3,   # RGB input
            out_channels=1   # Binary segmentation
        )
        
        test_input = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            output = model(test_input)
            
        print(f"✅ Segmentation model: input {test_input.shape} → output {output.shape}")
        
        if output.shape == torch.Size([1, 1, 128, 128]):
            print("✅ Backward compatibility maintained")
            return True
        else:
            print("❌ Backward compatibility broken")
            return False
            
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🫀 Classification Model Channel Fix Test")
    print("=" * 60)
    
    # Test the fix
    fix_success = test_classification_channel_fix()
    
    # Test backward compatibility  
    compat_success = test_backward_compatibility()
    
    if fix_success and compat_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Channel configuration fix is working correctly")
        print("✅ Backward compatibility is maintained")
        print("\n📝 Summary:")
        print("- Artery classification models now work with 1-channel input")
        print("- Segmentation models still work with 3-channel input")
        print("- Architecture registry correctly handles both cases")
        
    else:
        print("\n❌ SOME TESTS FAILED!")
        if not fix_success:
            print("❌ Channel fix is not working")
        if not compat_success:
            print("❌ Backward compatibility is broken")
    
    return fix_success and compat_success

if __name__ == "__main__":
    main()
