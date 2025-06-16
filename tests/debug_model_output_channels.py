#!/usr/bin/env python3
"""
Debug script to check model output channels issue.
"""
import torch
import sys
import os
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_monai_unet_configuration():
    """Test MONAI UNet configuration to see why it's outputting 256 channels"""
    print("🧪 Testing MONAI UNet configuration...")
    
    try:
        from monai.networks.nets import UNet as MonaiUNet
        
        # Test with explicit configuration that should give 1 output channel
        print("\n1. Testing explicit configuration (should output 1 channel):")
        model_config = {
            'spatial_dims': 2,
            'in_channels': 3,
            'out_channels': 1,  # This should be the final output
            'channels': (16, 32, 64, 128, 256),  # These are internal channel progressions
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
        }
        print(f"Config: {model_config}")
        
        model = MonaiUNet(**model_config)
        
        # Test with dummy input
        test_input = torch.randn(2, 3, 128, 128)  # Batch=2, Channels=3, H=128, W=128
        
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Model created successfully")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Expected output channels: {model_config['out_channels']}")
            print(f"   Actual output channels: {output.shape[1]}")
            
            if output.shape[1] == model_config['out_channels']:
                print("   ✅ Output channels match configuration!")
            else:
                print(f"   ❌ Output channels mismatch! Expected {model_config['out_channels']}, got {output.shape[1]}")
                
        # Check model's final layer
        print(f"\n2. Inspecting model architecture:")
        if hasattr(model, 'conv_last'):
            print(f"   Final conv layer out_channels: {model.conv_last.out_channels}")
        elif hasattr(model, 'out_conv'):
            print(f"   Final conv layer out_channels: {model.out_conv.out_channels}")
        elif hasattr(model, 'segmentation_head'):
            print(f"   Segmentation head out_channels: {model.segmentation_head.out_channels}")
        else:
            print("   Could not find final layer")
            print(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error testing MONAI UNet: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_model_creation():
    """Test the same model creation as used in training"""
    print("\n🧪 Testing dynamic model creation from training script...")
    
    try:
        from ml.training.train import create_model_from_registry
        
        # Test with binary segmentation config (similar to training)
        model_kwargs = {
            'spatial_dims': 2,
            'in_channels': 3,
            'out_channels': 1,  # Binary segmentation
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
        }
        
        print(f"Creating model with kwargs: {model_kwargs}")
        
        device = torch.device('cpu')
        model, arch_info = create_model_from_registry('unet', device, **model_kwargs)
        
        # Test with dummy input
        test_input = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ Dynamic model created successfully")
            print(f"   Architecture: {arch_info.display_name}")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Expected output channels: {model_kwargs['out_channels']}")
            print(f"   Actual output channels: {output.shape[1]}")
            
            if output.shape[1] == model_kwargs['out_channels']:
                print("   ✅ Output channels match configuration!")
                return True
            else:
                print(f"   ❌ Output channels mismatch! Expected {model_kwargs['out_channels']}, got {output.shape[1]}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing dynamic model creation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("🔍 DEBUGGING MODEL OUTPUT CHANNELS ISSUE")
    print("=" * 80)
    
    success1 = test_monai_unet_configuration()
    success2 = test_dynamic_model_creation()
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"MONAI UNet direct test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Dynamic model creation test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The model configuration should work correctly.")
    else:
        print("\n⚠️  Some tests failed. There's an issue with model configuration.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
