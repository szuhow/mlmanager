#!/usr/bin/env python3
"""
Debug script to test the exact model configuration flow from training.
"""
import torch
import sys
import os
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_config_flow():
    """Test the exact model configuration flow from training"""
    print("🧪 Testing model configuration flow from training...")
    
    try:
        from ml.training.train import get_default_model_config, create_model_from_registry
        
        # Simulate the exact flow from training
        model_type = 'unet'  # This is probably what's being used
        
        print(f"\n1. Getting default config for model_type: '{model_type}'")
        model_config = get_default_model_config(model_type)
        print(f"   Default config: {model_config}")
        
        # Set input channels (as done in training)
        input_channels = 3  # RGB
        model_config["in_channels"] = input_channels
        print(f"   After setting input channels: {model_config}")
        
        # Simulate binary class detection result
        class_info = {
            'num_classes': 1,
            'class_type': 'binary',
            'unique_values': [0.0, 1.0],
            'max_channels': 1
        }
        
        # Apply class detection (as done in training)
        if class_info:
            output_channels = class_info['num_classes']
            print(f"\n2. Applying class detection result:")
            print(f"   Detected {output_channels} output classes ({class_info['class_type']})")
            print(f"   Class values found: {class_info['unique_values']}")
            print(f"   Max channels in masks: {class_info['max_channels']}")
            
            if class_info['class_type'] == 'semantic_onehot':
                model_config["out_channels"] = class_info['max_channels']
                print(f"   Using {class_info['max_channels']} output channels for one-hot semantic segmentation")
            else:
                model_config["out_channels"] = output_channels
                print(f"   Using {output_channels} output channels for {class_info['class_type']} segmentation")
        else:
            default_out_channels = 1
            model_config["out_channels"] = default_out_channels
            print(f"   Using default {default_out_channels} output channels")
        
        print(f"\n3. Final model configuration: {model_config}")
        
        # Create model (as done in training)
        device = torch.device('cpu')
        print(f"\n4. Creating model with configuration...")
        model, arch_info = create_model_from_registry(model_type, device, **model_config)
        
        # Test the model
        test_input = torch.randn(2, 3, 128, 128)  # Same as in error: batch=2, channels=3
        
        with torch.no_grad():
            output = model(test_input)
            print(f"\n5. Model test results:")
            print(f"   Architecture: {arch_info.display_name}")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Expected output channels: {model_config['out_channels']}")
            print(f"   Actual output channels: {output.shape[1]}")
            
            if output.shape[1] == model_config['out_channels']:
                print("   ✅ Output channels match configuration!")
                return True
            else:
                print(f"   ❌ Output channels mismatch! Expected {model_config['out_channels']}, got {output.shape[1]}")
                
                # Additional debugging
                print(f"\n6. Additional debugging:")
                print(f"   Model config channels: {model_config.get('channels', 'Not set')}")
                print(f"   Model config spatial_dims: {model_config.get('spatial_dims', 'Not set')}")
                print(f"   Model config strides: {model_config.get('strides', 'Not set')}")
                print(f"   Model config num_res_units: {model_config.get('num_res_units', 'Not set')}")
                
                return False
                
    except Exception as e:
        print(f"❌ Error testing model config flow: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_class_detection():
    """Test what happens when class detection fails (None result)"""
    print("\n🧪 Testing fallback behavior when class detection fails...")
    
    try:
        from ml.training.train import get_default_model_config, create_model_from_registry
        
        model_type = 'unet'
        
        print(f"\n1. Getting default config for model_type: '{model_type}'")
        model_config = get_default_model_config(model_type)
        print(f"   Default config: {model_config}")
        
        # Set input channels (as done in training)
        input_channels = 3
        model_config["in_channels"] = input_channels
        
        # Simulate class detection failure (class_info = None)
        class_info = None
        
        if class_info:
            # This won't run
            pass
        else:
            # Fallback to default
            default_out_channels = 1
            model_config["out_channels"] = default_out_channels
            print(f"   Using default {default_out_channels} output channels (class detection failed)")
        
        print(f"\n2. Final model configuration: {model_config}")
        
        # Create model
        device = torch.device('cpu')
        model, arch_info = create_model_from_registry(model_type, device, **model_config)
        
        # Test the model
        test_input = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(test_input)
            print(f"\n3. Model test results:")
            print(f"   Architecture: {arch_info.display_name}")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Expected output channels: {model_config['out_channels']}")
            print(f"   Actual output channels: {output.shape[1]}")
            
            if output.shape[1] == model_config['out_channels']:
                print("   ✅ Output channels match configuration!")
                return True
            else:
                print(f"   ❌ Output channels mismatch! Expected {model_config['out_channels']}, got {output.shape[1]}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing fallback behavior: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 100)
    print("🔍 DEBUGGING MODEL CONFIGURATION FLOW ISSUE")
    print("=" * 100)
    
    success1 = test_model_config_flow()
    success2 = test_without_class_detection()
    
    print("\n" + "=" * 100)
    print("📋 SUMMARY")
    print("=" * 100)
    print(f"Model config flow test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Fallback behavior test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The model configuration flow should work correctly.")
        print("💡 The issue might be elsewhere in the training pipeline.")
    else:
        print("\n⚠️  Some tests failed. There's an issue with model configuration.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
